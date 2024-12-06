import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from math import ceil, log2
from sklearn.cluster import MiniBatchKMeans, Birch
from tqdm import tqdm

from utils import batch_distances
from index import SearchResult, HNSW, HNSWConfig


@dataclass
class ClusterConfig:
    """Configuration for clustering parameters"""
    average_cluster_size: int = 128
    maximum_cluster_size: Optional[int] = None
    insert_method: str = "ClusterThenInsert"
    algorithm_name: str = "MiniBatchKMeans"
    algorithm_params: dict = None

    def __str__(self):
        return f"ClusterConfig(avg_size={self.average_cluster_size}, max_size={self.maximum_cluster_size}, " \
               f"insert_method={self.insert_method}, algorithm_params={self.algorithm_params})"

    def __post_init__(self):
        if self.algorithm_params is None:
            self.algorithm_params = {}


@dataclass
class ClusterInfo:
    """Information about a single cluster"""
    centroid: np.ndarray
    vectors: np.ndarray
    indices: np.ndarray

    @property
    def size(self) -> int:
        return len(self.vectors)


class ClusteredHNSW(HNSW):
    def __init__(self, dim, hnsw_config: HNSWConfig, cluster_config: Optional[ClusterConfig] = None):
        super().__init__(dim, hnsw_config)

        self.cumulative_sizes = None
        self.cluster_config = cluster_config or ClusterConfig()
        self.clusters: Dict[int, ClusterInfo] = {}
        self.cluster_sizes: Dict[int, int] = {}
        self._original_vectors: Optional[np.ndarray] = None
        self.max_candidates = 0

    def __str__(self):
        return f"(M={self.config.M}, C={self.cluster_config.average_cluster_size}, {self.cluster_config.insert_method})"

    def _create_clusterer(self, n_clusters: int):
        """Create the clustering algorithm instance based on configuration"""
        params = self.cluster_config.algorithm_params
        if self.cluster_config.algorithm_name == "MiniBatchKMeans":
            return MiniBatchKMeans(
                n_init=params.get('n_init', 64),
                max_no_improvement=params.get('max_no_improvement', 64),
                max_iter=params.get('max_iter', 16),
                init='random',
                n_clusters=n_clusters,
                batch_size=params.get('batch_size', max(1024, n_clusters * 2)),
                reassignment_ratio=params.get('reassignment_ratio', 0.05),
                **params
            )
        # TODO: optimize Birch parameters; Binary Search for threshold?
        # Here as a placeholder till we integrate with rest of the code
        elif self.cluster_config.algorithm_name == "Birch":
            return Birch(
                n_clusters=n_clusters,
                threshold=params.get('threshold', 1),
                branching_factor=params.get('branching_factor', 64),
                **params
            )

    def batch_insert_cluster_method(self, vectors: np.ndarray, indices: Optional[np.ndarray] = None,
                                    verbose: bool = True):
        vectors_offset = 0
        if indices is None:
            if self._original_vectors is None:
                self._original_vectors = vectors
            else:
                self._original_vectors = np.vstack((self._original_vectors, vectors))
            vectors_offset = len(self._original_vectors) - len(vectors)
        n_clusters = len(vectors) // self.cluster_config.average_cluster_size
        self.nodes_in_memory += n_clusters
        clusterer = self._create_clusterer(n_clusters)
        cluster_labels = clusterer.fit_predict(vectors)
        unique_labels = np.unique(cluster_labels)
        centroids = clusterer.cluster_centers_
        if self.vectors is None:
            self.vectors = centroids
        else:
            self.vectors = np.vstack((self.vectors, centroids))
        pbar = tqdm(unique_labels) if verbose else unique_labels
        for i, label in enumerate(pbar):
            mask = cluster_labels == label
            cluster_vectors = vectors[mask]
            if indices is None:
                cluster_indices = np.where(mask)[0]
                # add offset to indices
                cluster_indices += vectors_offset
            else:
                cluster_indices = indices[mask]
            centroid = centroids[i]
            node = super().insert(centroid)
            self.clusters[node] = ClusterInfo(
                centroid=centroid,
                vectors=cluster_vectors,
                indices=cluster_indices,
            )
            self.cluster_sizes[node] = len(cluster_vectors)

    def compute_recommended_efs(self):
        # sort clusters by size
        sorted_clusters = sorted(self.cluster_sizes, key=self.cluster_sizes.get)
        # compute cumulative sum of cluster sizes
        self.cumulative_sizes = np.cumsum([self.cluster_sizes[cluster_id] for cluster_id in sorted_clusters])

    def batch_insert(self, vectors: np.ndarray):
        start_time = time.time()

        if self.cluster_config.insert_method == "ClusterThenInsert" or self._original_vectors is None:
            self.batch_insert_cluster_method(vectors)
        elif self.cluster_config.insert_method == "InsertBlindly":
            self.batch_insert_only(vectors)
        elif self.cluster_config.maximum_cluster_size is not None:
            cluster_indices = self.batch_insert_only(vectors)
            to_split = set()
            for cluster_index in cluster_indices:
                if len(self.clusters[cluster_index].vectors) > self.cluster_config.maximum_cluster_size:
                    to_split.add(cluster_index)
                    super().delete(cluster_index)
            for cluster_index in tqdm(to_split, desc="Splitting Clusters"):
                new_vectors = self.clusters[cluster_index].vectors
                new_indices = self.clusters[cluster_index].indices
                self.batch_insert_cluster_method(new_vectors, new_indices, verbose=False)
                self.clusters.pop(cluster_index)
                self.cluster_sizes.pop(cluster_index)
        else:
            raise ValueError(
                f"Invalid insert method: {self.cluster_config.insert_method} | max_cluster_size: {self.cluster_config.maximum_cluster_size}")

        self.construction_time += time.time() - start_time
        self.compute_recommended_efs()

    def insert(self, vector: np.ndarray):
        nearest_centroid = super().search(vector, 1, ef=self.config.ef_construction).indices[0]
        self.clusters[nearest_centroid].vectors = np.vstack((self.clusters[nearest_centroid].vectors, vector))
        self.cluster_sizes[nearest_centroid] += 1
        new_index = len(self._original_vectors)
        self.clusters[nearest_centroid].indices = np.append(self.clusters[nearest_centroid].indices, new_index)
        self._original_vectors = np.vstack((self._original_vectors, vector))
        return nearest_centroid

    def batch_insert_only(self, vectors: np.ndarray):
        new_indices = np.arange(len(self._original_vectors), len(self._original_vectors) + len(vectors))
        self._original_vectors = np.vstack((self._original_vectors, vectors))
        nearest_centroids = []
        for vector in tqdm(vectors, desc="Finding Nearest Centroids"):
            nearest_centroids.append(super().search(vector, 1, ef=self.config.ef_construction).indices[0])
        unique_centroids = np.unique(nearest_centroids)
        for centroid in tqdm(unique_centroids, desc="Inserting Vectors"):
            mask = nearest_centroids == centroid
            cluster_vectors = vectors[mask]
            cluster_indices = new_indices[mask]
            self.clusters[centroid].vectors = np.vstack((self.clusters[centroid].vectors, cluster_vectors))
            self.clusters[centroid].indices = np.append(self.clusters[centroid].indices, cluster_indices)
            self.cluster_sizes[centroid] += len(cluster_vectors)
        return unique_centroids

    def search(self, query: np.ndarray, k: int, ef_search: int) -> SearchResult:
        """
        Search for k nearest neighbors of query vector
        """
        # get the ef smallest clusters
        recommended_ef = np.searchsorted(self.cumulative_sizes, k, side='left')

        if recommended_ef > ef_search:
            logging.warning(f"Recommended ef: {recommended_ef} > ef_search: {ef_search}")
            logging.warning(f"Recommended ef will be used instead")
            ef_search = recommended_ef

        start_time = time.time()
        # First find ef nearest cluster centroids
        centroid_results = super().search(query, ef_search, ef_search)

        # Check if we have enough clusters to search
        actual_case_size = sum(self.cluster_sizes[cluster_id] for cluster_id in centroid_results.indices)
        if actual_case_size < k:
            raise ValueError(f"Search failed: {actual_case_size} < {k}")

        # Get all vectors from the ef nearest clusters
        candidates = []
        candidate_indices = []

        for centroid_id in centroid_results.indices:
            if centroid_id not in self.clusters:
                raise ValueError(f"Cluster {centroid_id} not found in index")
            cluster = self.clusters[centroid_id]
            candidates.append(cluster.vectors)
            candidate_indices.extend(cluster.indices)

        # Concatenate all candidates
        all_candidates = np.vstack(candidates)

        # Compute distances to all candidates
        distances = batch_distances(query, np.arange(len(all_candidates)), all_candidates)
        self.distance_computations += len(all_candidates)
        self.max_candidates = max(self.max_candidates, len(all_candidates))
        # Get top k results
        top_k_idx = np.argpartition(distances, min(k, len(distances) - 1))[:k]
        top_k_distances = distances[top_k_idx]
        top_k_indices = [candidate_indices[i] for i in top_k_idx]

        # Sort by distance
        sorted_idx = np.argsort(top_k_distances)

        return SearchResult(
            indices=[top_k_indices[i] for i in sorted_idx],
            distances=[float(top_k_distances[i]) for i in sorted_idx],
            query_time=time.time() - start_time
        )

    def compute_stats(self) -> dict:
        self.nodes_in_memory += self.max_candidates
        results = super().compute_stats()
        self.nodes_in_memory -= self.max_candidates
        self.max_candidates = 0
        return results

    def compute_cluster_stats(self) -> dict:
        """Compute statistics about the clustering"""
        sizes = [c.size for c in self.clusters.values()]

        return {
            "num_clusters": len(self.clusters),
            "total_vectors": sum(sizes),
            "avg_cluster_size": np.mean(sizes),
            "min_cluster_size": np.min(sizes),
            "max_cluster_size": np.max(sizes),
            "std_cluster_size": np.std(sizes),
        }


if __name__ == "__main__":
    np.random.seed(42)
    dim = 2
    vectors = np.random.randn(6400, dim).astype(np.float32)
    k = 400
    m = 5
    cluster_config = ClusterConfig(
        average_cluster_size=128,
        maximum_cluster_size=256,
        insert_method="InsertWithMitosis"
    )
    hnsw_config = HNSWConfig(
        M=m,
        M0=2 * m,
        mL=1 / np.log(m),
        ef_construction=4 * m,
        max_level=ceil(log2(m))
    )
    index = ClusteredHNSW(
        dim=dim,
        hnsw_config=hnsw_config,
        cluster_config=cluster_config
    )
    index.batch_insert(vectors)
    for vector in vectors:
        result = index.search(vector, k, m)
    print(index.compute_stats())
    print(index.compute_cluster_stats())
    index.batch_insert(vectors)
    index.batch_insert(vectors)
    for vector in vectors[:50]:
        result = index.search(vector, k, m)
        print(result)
    print(index.compute_stats())
    print(index.compute_cluster_stats())
