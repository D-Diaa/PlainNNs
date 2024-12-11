import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Any

import numpy as np
from math import ceil, log2

@dataclass
class HNSWConfig:
    """
    Configuration for HNSW index.

    Attributes:
        M (int): Maximum number of connections per layer.
        M0 (int): Maximum number of connections for the 0-th layer.
        mL (float): Level multiplier based on logarithmic function.
        max_level (int): Maximum level in the hierarchy.
        ef_construction (int): Size of the dynamic list used during construction.
    """
    M: int
    M0: int
    mL: float
    max_level: int
    ef_construction: int = 256

    @staticmethod
    def create(m):
        """
        Create an HNSWConfig instance with calculated parameters based on M.

        Args:
            m (int): Maximum number of connections per layer.

        Returns:
            HNSWConfig: Configured HNSWConfig instance.
        """
        return HNSWConfig(M=m, M0=2 * m, mL=1 / np.log(m), max_level=ceil(log2(m)))

    def __str__(self):
        """
        Return a string representation of the HNSWConfig instance.

        Returns:
            str: String representation of the configuration.
        """
        return (f"HNSWConfig(M={self.M}, M0={self.M0}, mL={self.mL}, "
                f"ef_construction={self.ef_construction}, max_level={self.max_level})")

@dataclass
class Node:
    """
    Representation of a node in the HNSW graph.

    Attributes:
        id (int): Unique identifier for the node.
        level (int): Level of the node in the hierarchy.
        neighbors (Dict[int, List[int]]): Mapping of levels to lists of neighbor node IDs.
    """
    id: int
    level: int
    neighbors: Dict[int, List[int]]

    def degree(self, level: int) -> int:
        """
        Calculate the degree of the node at a specific level.

        Args:
            level (int): Level to calculate the degree for.

        Returns:
            int: Number of neighbors at the specified level.
        """
        return len(self.neighbors[level])

    def __eq__(self, other):
        """
        Compare equality with another Node based on ID.

        Args:
            other (Node): Node to compare against.

        Returns:
            bool: True if IDs are equal, False otherwise.
        """
        return self.id == other.id

    def __hash__(self):
        """
        Compute a hash for the node based on its ID.

        Returns:
            int: Hash value.
        """
        return hash(self.id)

    def __lt__(self, other):
        """
        Compare this node with another node for ordering based on ID.

        Args:
            other (Node): Node to compare against.

        Returns:
            bool: True if this node's ID is less than the other node's ID.
        """
        return self.id < other.id

@dataclass
class SearchResult:
    """
    Representation of a search result in the HNSW index.

    Attributes:
        indices (List[int]): List of indices of the nearest neighbors.
        distances (List[float]): List of distances corresponding to the nearest neighbors.
        query_time (float): Time taken to execute the query.
    """
    indices: List[int]
    distances: List[float]
    query_time: float

@dataclass
class ClusterConfig:
    """
    Configuration for clustering parameters.

    Attributes:
        average_cluster_size (int): Desired average size of clusters.
        maximum_cluster_size (Optional[int]): Maximum size of a single cluster. None means no limit.
        insert_method (str): Method used to insert new items into clusters.
        algorithm_name (str): Name of the clustering algorithm to use.
        algorithm_params (dict): Additional parameters for the clustering algorithm.
    """
    average_cluster_size: int = 128
    maximum_cluster_size: Optional[int] = None
    insert_method: str = "ClusterThenInsert"
    algorithm_name: str = "MiniBatchKMeans"
    algorithm_params: dict = None

    def __str__(self):
        """
        Return a string representation of the ClusterConfig instance.

        Returns:
            str: String representation of the configuration.
        """
        return f"ClusterConfig(avg_size={self.average_cluster_size}, max_size={self.maximum_cluster_size}, " \
               f"insert_method={self.insert_method}, algorithm_params={self.algorithm_params})"

    def __post_init__(self):
        """
        Initialize default values for algorithm_params if not provided.
        """
        if self.algorithm_params is None:
            self.algorithm_params = {}

@dataclass
class ClusterInfo:
    """
    Information about a single cluster.

    Attributes:
        centroid (np.ndarray): Centroid of the cluster.
        vectors (np.ndarray): Data points belonging to the cluster.
        indices (np.ndarray): Indices of the data points in the original dataset.
    """
    centroid: np.ndarray
    vectors: np.ndarray
    indices: np.ndarray

    @property
    def size(self) -> int:
        """
        Compute the size of the cluster.

        Returns:
            int: Number of data points in the cluster.
        """
        return len(self.vectors)

@dataclass
class EvaluationMetrics:
    """
    Metrics for evaluating the performance of the HNSW index.

    Attributes:
        name (str): Name of the evaluation or dataset.
        recall (float): Recall achieved during search.
        precision (float): Precision achieved during search.
        f1_score (float): F1 score combining recall and precision.
        average_query_time (float): Average time per query.
        median_query_time (float): Median time per query.
        query_time_95th_percentile (float): 95th percentile of query times.
        queries_per_second (float): Number of queries processed per second.
        distance_computations_per_query (int): Average distance computations per query.
        distance_ratio (float): Ratio of computed distances to possible distances.
        memory_usage_bytes (int): Memory used by the index in bytes.
        level_distribution (Dict[int, int]): Distribution of nodes across levels in the index.
        average_out_degree (float): Average number of neighbors per node.
        construction_time (float): Total time taken to construct the index.
    """
    name: str
    recall: float
    precision: float
    f1_score: float
    average_query_time: float
    median_query_time: float
    query_time_95th_percentile: float
    queries_per_second: float
    distance_computations_per_query: int
    distance_ratio: float
    memory_usage_bytes: int
    level_distribution: Dict[int, int]
    average_out_degree: float
    construction_time: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the EvaluationMetrics instance to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the metrics.
        """
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'EvaluationMetrics':
        """
        Create an EvaluationMetrics instance from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing evaluation metrics.

        Returns:
            EvaluationMetrics: Configured EvaluationMetrics instance.
        """
        return EvaluationMetrics(**data)

    def __str__(self):
        """
        Return a string representation of the EvaluationMetrics instance.

        Returns:
            str: String representation of the metrics.
        """
        return (f"EvaluationMetrics(name={self.name}, recall={self.recall}, precision={self.precision}, "
                f"f1_score={self.f1_score}, avg_query_time={self.average_query_time}, "
                f"median_query_time={self.median_query_time}, queries_per_second={self.queries_per_second}, "
                f"dist_computations_per_query={self.distance_computations_per_query}, "
                f"dist_ratio={self.distance_ratio}, memory_usage={self.memory_usage_bytes}, "
                f"level_dist={self.level_distribution}, avg_out_degree={self.average_out_degree}, "
                f"construction_time={self.construction_time})")

    def log(self):
        """
        Log evaluation metrics in a structured format.
        """
        logging.info("\nEvaluation Results:")
        logging.info(f"Recall: {self.recall:.4f}")
        logging.info(f"Precision: {self.precision:.4f}")
        logging.info(f"F1 Score: {self.f1_score:.4f}")
        logging.info(f"Average Query Time: {self.average_query_time * 1000:.2f}ms")
        logging.info(f"Median Query Time: {self.median_query_time * 1000:.2f}ms")
        logging.info(f"95th Percentile Query Time: {self.query_time_95th_percentile * 1000:.2f}ms")
        logging.info(f"Queries per Second: {self.queries_per_second:.2f}")
        logging.info(f"Average Distance Computations per Query: {self.distance_computations_per_query:.2f}")
        logging.info(f"Memory Usage: {self.memory_usage_bytes / (1024 * 1024):.2f}MB")
        logging.info(f"Construction Time: {self.construction_time:.2f}s")
        logging.info("\nLevel Distribution:")
        for level, count in sorted(self.level_distribution.items()):
            logging.info(f"Level {level}: {count} nodes")
