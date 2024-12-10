import logging
import time
from typing import Dict, Optional

import numpy as np
from math import ceil, log2
from sklearn.cluster import MiniBatchKMeans, Birch
from tqdm import tqdm

from hnsw import HNSW
from utils import batch_distances
from classes import ClusterConfig, ClusterInfo, HNSWConfig, SearchResult


class ClusteredHNSW(HNSW):
    """
    ClusteredHNSW is an extension of the HNSW algorithm that integrates clustering functionality for better management
    and search performance in high-dimensional data.

    Attributes:
        cumulative_sizes (Optional[np.ndarray]): Cumulative sizes of clusters for ef calculation.
        cluster_config (ClusterConfig): Configuration parameters for clustering.
        clusters (Dict[int, ClusterInfo]): Information about clusters.
        _original_vectors (Optional[np.ndarray]): Original vectors inserted into the structure.
        max_candidates (int): Maximum number of candidates evaluated during searches.
    """

    def __init__(self, dim, hnsw_config: HNSWConfig, cluster_config: Optional[ClusterConfig] = None):
        """
        Initialize a ClusteredHNSW instance.

        Args:
            dim (int): Dimensionality of the vectors.
            hnsw_config (HNSWConfig): Configuration for HNSW.
            cluster_config (Optional[ClusterConfig]): Configuration for clustering. Defaults to None.
        """
        super().__init__(dim, hnsw_config)

        self.cumulative_sizes = None  # Placeholder for cumulative cluster sizes used in recommended ef calculation.
        self.cluster_config = cluster_config or ClusterConfig()  # Use provided or default cluster configuration.
        self.clusters: Dict[int, ClusterInfo] = {}  # Dictionary to store cluster information.
        self._original_vectors: Optional[np.ndarray] = None  # Cache for the original vectors.
        self.max_candidates = 0  # Tracks the maximum number of candidates in memory during search.

    def __str__(self):
        """
        Provide a string representation of the ClusteredHNSW instance.

        Returns:
            str: A string summarizing the configuration.
        """
        return f"(M={self.config.M}, C={self.cluster_config.average_cluster_size}, {self.cluster_config.insert_method})"

    def _create_clusterer(self, n_clusters: int):
        """
        Create the clustering algorithm instance based on configuration.

        Args:
            n_clusters (int): Number of clusters to create.

        Returns:
            Any: Initialized clustering algorithm.
        """
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
        """
        Insert vectors into the HNSW index using a clustering method.

        Args:
            vectors (np.ndarray): Vectors to be inserted.
            indices (Optional[np.ndarray]): Indices of the vectors. Defaults to None.
            verbose (bool): Flag to enable progress display. Defaults to True.
        """
        vectors_offset = 0  # Offset for indices if original vectors are being appended.
        if indices is None:
            if self._original_vectors is None:
                self._original_vectors = vectors  # Initialize original vectors if not present.
            else:
                self._original_vectors = np.vstack((self._original_vectors, vectors))  # Append to original vectors.
            vectors_offset = len(self._original_vectors) - len(vectors)  # Calculate offset for new vectors.
        n_clusters = len(vectors) // self.cluster_config.average_cluster_size  # Compute number of clusters needed.
        self.nodes_in_memory += n_clusters  # Account for additional nodes in memory.
        clusterer = self._create_clusterer(n_clusters)  # Initialize clusterer based on config.
        cluster_labels = clusterer.fit_predict(vectors)  # Perform clustering.
        unique_labels = np.unique(cluster_labels)  # Unique cluster labels.
        centroids = clusterer.cluster_centers_  # Centroids of clusters.
        if self.vectors is None:
            self.vectors = centroids  # Initialize vectors with centroids if none exist.
        else:
            self.vectors = np.vstack((self.vectors, centroids))  # Append centroids to existing vectors.
        pbar = tqdm(unique_labels) if verbose else unique_labels  # Initialize progress bar if verbose.
        for i, label in enumerate(pbar):
            mask = cluster_labels == label  # Mask for vectors belonging to the current cluster.
            cluster_vectors = vectors[mask]  # Extract cluster vectors.
            if indices is None:
                cluster_indices = np.where(mask)[0] + vectors_offset  # Calculate indices with offset.
            else:
                cluster_indices = indices[mask]  # Use provided indices.
            centroid = centroids[i]  # Current centroid.
            node = super().insert(centroid)  # Insert centroid into HNSW structure.
            self.clusters[node] = ClusterInfo(
                centroid=centroid,
                vectors=cluster_vectors,
                indices=cluster_indices,
            )  # Store cluster information.

    def prepare_recommended_ef(self):
        """
        Prepare cumulative sizes for recommended ef computation.
        """
        cluster_sizes = {cluster_id: self.clusters[cluster_id].size for cluster_id in self.clusters}  # Get sizes of clusters.
        sorted_clusters = sorted(cluster_sizes, key=cluster_sizes.get)  # Sort clusters by size.
        self.cumulative_sizes = np.cumsum([cluster_sizes[cluster_id] for cluster_id in sorted_clusters])  # Compute cumulative sizes.

    def get_recommended_ef(self, k: int):
        """
        Get the recommended ef value for a given k.

        Args:
            k (int): Number of nearest neighbors.

        Returns:
            int: Recommended ef value.
        """
        recommended_ef = np.searchsorted(self.cumulative_sizes, k, side='left')  # Find where k fits in cumulative sizes.
        return recommended_ef

    def batch_insert(self, vectors: np.ndarray):
        """
        Batch insert vectors into the HNSW structure.

        Args:
            vectors (np.ndarray): Vectors to insert.
        """
        start_time = time.time()  # Track start time.

        if self.cluster_config.insert_method == "ClusterThenInsert" or self._original_vectors is None:
            self.batch_insert_cluster_method(vectors)  # Use cluster-based insertion.
        elif self.cluster_config.insert_method == "InsertBlindly":
            self.batch_insert_only(vectors)  # Direct insertion without clustering.
        elif self.cluster_config.maximum_cluster_size is not None:
            cluster_indices = self.batch_insert_only(vectors)  # Perform direct insertion.
            to_split = set()  # Track clusters to split.
            for cluster_index in cluster_indices:
                if len(self.clusters[cluster_index].vectors) > self.cluster_config.maximum_cluster_size:
                    to_split.add(cluster_index)  # Mark oversized clusters for splitting.
                    super().delete(cluster_index)  # Remove oversized cluster centroid.
            for cluster_index in tqdm(to_split, desc="Splitting Clusters"):
                new_vectors = self.clusters[cluster_index].vectors  # Extract vectors from oversized cluster.
                new_indices = self.clusters[cluster_index].indices  # Extract indices from oversized cluster.
                self.batch_insert_cluster_method(new_vectors, new_indices, verbose=False)  # Reinsert vectors using clustering.
                self.clusters.pop(cluster_index)  # Remove old cluster.
        else:
            raise ValueError(
                f"Invalid insert method: {self.cluster_config.insert_method} | max_cluster_size: {self.cluster_config.maximum_cluster_size}")

        self.construction_time += time.time() - start_time  # Update construction time.
        self.prepare_recommended_ef()  # Prepare ef recommendations.

    def insert(self, vector: np.ndarray):
        """
        Insert a single vector into the nearest cluster.

        Args:
            vector (np.ndarray): The vector to insert.

        Returns:
            int: The ID of the nearest centroid.
        """
        nearest_centroid = super().search(vector, 1, ef=self.config.ef_construction).indices[0]  # Find nearest centroid.
        self.clusters[nearest_centroid].vectors = np.vstack((self.clusters[nearest_centroid].vectors, vector))  # Add vector to cluster.
        new_index = len(self._original_vectors)  # Calculate new index.
        self.clusters[nearest_centroid].indices = np.append(self.clusters[nearest_centroid].indices, new_index)  # Update indices.
        self._original_vectors = np.vstack((self._original_vectors, vector))  # Add vector to original set.
        return nearest_centroid

    def batch_insert_only(self, vectors: np.ndarray):
        """
        Insert vectors without clustering.

        Args:
            vectors (np.ndarray): Vectors to insert.

        Returns:
            np.ndarray: Unique centroids affected.
        """
        new_indices = np.arange(len(self._original_vectors), len(self._original_vectors) + len(vectors))  # Generate indices for new vectors.
        self._original_vectors = np.vstack((self._original_vectors, vectors))  # Append vectors to original set.
        nearest_centroids = []  # Track nearest centroids.
        for vector in tqdm(vectors, desc="Finding Nearest Centroids"):
            nearest_centroids.append(super().search(vector, 1, ef=self.config.ef_construction).indices[0])  # Find nearest centroid for each vector.
        unique_centroids = np.unique(nearest_centroids)  # Get unique centroids.
        for centroid in tqdm(unique_centroids, desc="Inserting Vectors"):
            mask = nearest_centroids == centroid  # Mask for vectors assigned to the current centroid.
            cluster_vectors = vectors[mask]  # Extract vectors for centroid.
            cluster_indices = new_indices[mask]  # Extract corresponding indices.
            self.clusters[centroid].vectors = np.vstack((self.clusters[centroid].vectors, cluster_vectors))  # Add vectors to cluster.
            self.clusters[centroid].indices = np.append(self.clusters[centroid].indices, cluster_indices)  # Update cluster indices.
        return unique_centroids

    def search(self, query: np.ndarray, k: int, ef_search: int) -> SearchResult:
        """
        Search for the k-nearest neighbors to a query vector.

        Args:
            query (np.ndarray): Query vector.
            k (int): Number of neighbors to retrieve.
            ef_search (int): ef parameter for the search.

        Returns:
            SearchResult: The search results containing indices and distances.
        """
        recommended_ef = self.get_recommended_ef(k)  # Get recommended ef based on k.

        if recommended_ef > ef_search:
            logging.warning(f"Recommended ef: {recommended_ef} > ef_search: {ef_search}")  # Warn if ef_search is less than recommended.
            logging.warning(f"Recommended ef will be used instead")
            ef_search = recommended_ef  # Override ef_search.

        start_time = time.time()  # Track start time.
        centroid_results = super().search(query, ef_search, ef_search)  # Search centroids.

        actual_case_size = sum(self.clusters[cluster_id].size for cluster_id in centroid_results.indices)  # Compute total candidate size.
        if actual_case_size < k:
            raise ValueError(f"Search failed: {actual_case_size} < {k}")

        candidate_distances = []  # Distances for candidates.
        candidate_indices = []  # Indices for candidates.

        for centroid_id in centroid_results.indices:
            if centroid_id not in self.clusters:
                raise ValueError(f"Cluster {centroid_id} not found in index")  # Raise error if cluster is missing.
            cluster = self.clusters[centroid_id]  # Get cluster information.
            cluster_size = cluster.size  # Cluster size.
            cluster_distances = batch_distances(query, np.arange(cluster_size), cluster.vectors)  # Compute distances to cluster vectors.
            candidate_distances.extend(cluster_distances)  # Add distances to candidates.
            candidate_indices.extend(cluster.indices)  # Add indices to candidates.
            self.distance_computations += cluster_size  # Update distance computations.
            self.max_candidates = max(self.max_candidates, cluster_size)  # Track maximum candidates.
        top_k_idx = np.argpartition(candidate_distances, min(k, len(candidate_distances) - 1))[:k]  # Get top-k indices.
        top_k_distances = candidate_distances[top_k_idx]  # Extract top-k distances.
        top_k_indices = [candidate_indices[i] for i in top_k_idx]  # Extract top-k indices.

        sorted_idx = np.argsort(top_k_distances)  # Sort by distance.

        return SearchResult(
            indices=[top_k_indices[i] for i in sorted_idx],  # Sort indices by distance.
            distances=[float(top_k_distances[i]) for i in sorted_idx],  # Sort distances.
            query_time=time.time() - start_time  # Calculate query time.
        )

    def compute_stats(self) -> dict:
        """
        Compute and return statistics about the HNSW structure.

        Returns:
            dict: Dictionary containing the statistics.
        """
        self.nodes_in_memory += self.max_candidates  # Adjust memory usage for candidates.
        results = super().compute_stats()  # Compute base HNSW stats.
        self.nodes_in_memory -= self.max_candidates  # Reset memory usage adjustment.
        self.max_candidates = 0  # Reset max candidates.
        return results

    def compute_cluster_stats(self) -> dict:
        """
        Compute and return statistics about the clusters.

        Returns:
            dict: Dictionary containing cluster statistics.
        """
        sizes = [c.size for c in self.clusters.values()]  # Collect cluster sizes.

        return {
            "num_clusters": len(self.clusters),  # Number of clusters.
            "total_vectors": sum(sizes),  # Total number of vectors.
            "avg_cluster_size": np.mean(sizes),  # Average cluster size.
            "min_cluster_size": np.min(sizes),  # Minimum cluster size.
            "max_cluster_size": np.max(sizes),  # Maximum cluster size.
            "std_cluster_size": np.std(sizes),  # Standard deviation of cluster sizes.
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
