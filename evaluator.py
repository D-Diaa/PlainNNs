import logging
from typing import Optional, List, Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from classes import EvaluationMetrics
from hnsw import HNSW

logging.basicConfig(level=logging.INFO)

class HNSWEvaluator:
    """
    A class to evaluate the performance of the Hierarchical Navigable Small World (HNSW) graph.

    Attributes:
        name (str): The name of the evaluation task.
        base_vectors (np.ndarray): The base dataset vectors.
        query_vectors (np.ndarray): The query vectors for evaluation.
        groundtruth_indices (Optional[np.ndarray]): Precomputed ground truth indices.
        recompute_ground_truth (bool): Whether to recompute ground truth indices.
        k (int): Number of nearest neighbors to consider.
    """

    def __init__(
            self,
            name: str,
            base_vectors: np.ndarray,
            query_vectors: np.ndarray,
            groundtruth_vectors: Optional[np.ndarray] = None,
            recompute_ground_truth: bool = False,
            k: int = 10
    ):
        """
        Initializes the HNSWEvaluator with the required data and parameters.

        Args:
            name (str): Name of the evaluation task.
            base_vectors (np.ndarray): Base dataset vectors.
            query_vectors (np.ndarray): Query vectors for evaluation.
            groundtruth_vectors (Optional[np.ndarray]): Precomputed ground truth indices.
            recompute_ground_truth (bool): Flag to recompute ground truth.
            k (int): Number of nearest neighbors.
        """
        # Store parameters
        self.base_vectors = base_vectors
        self.query_vectors = query_vectors
        self.k = k
        self.name = name
        self.recompute_ground_truth = recompute_ground_truth
        self.groundtruth_indices = groundtruth_vectors

        # Compute ground truth if necessary
        if self.groundtruth_indices is None or self.recompute_ground_truth:
            self._compute_ground_truth()

    def _compute_ground_truth(self):
        """
        Computes the exact nearest neighbors for the query vectors using brute-force search.
        """
        logging.info("Computing exact nearest neighbors...")
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='brute').fit(self.base_vectors)
        self.ground_truth_distances, self.groundtruth_indices = nbrs.kneighbors(self.query_vectors)
        logging.info("Ground truth computation complete")

    def evaluate(self, index: HNSW, ef_search: int) -> EvaluationMetrics:
        """
        Evaluates the HNSW index for a specific ef_search parameter.

        Args:
            index (HNSW): The HNSW index to evaluate.
            ef_search (int): The ef_search parameter value for evaluation.

        Returns:
            EvaluationMetrics: The computed evaluation metrics.
        """
        # Initialize metrics
        total_time = 0.0
        total_distance_computations = 0
        recalls: List[float] = []
        precisions: List[float] = []
        distance_ratios: List[float] = []
        query_times: List[float] = []

        logging.info(f"Evaluating queries with ef_search={ef_search}...")

        for i in range(len(self.query_vectors)):
            # Perform a search in the HNSW index
            result = index.search(self.query_vectors[i], self.k, ef_search)

            # Record query execution time
            query_times.append(result.query_time)
            total_time += result.query_time
            total_distance_computations += index.distance_computations

            # Compute recall and precision by comparing with ground truth
            ground_truth_set = set(self.groundtruth_indices[i][:self.k])
            result_set = set(result.indices)

            intersection = len(ground_truth_set.intersection(result_set))
            recalls.append(intersection / len(ground_truth_set))
            precisions.append(intersection / len(result_set))

            # Calculate distance ratios for quality assessment if ground truth distances exist
            if hasattr(self, 'ground_truth_distances'):
                hnsw_distances = np.sqrt(result.distances)
                exact_distances = self.ground_truth_distances[i]
                exact_distances = np.maximum(exact_distances, 1e-10)  # Avoid division by zero
                ratios = hnsw_distances / exact_distances
                distance_ratios.append(np.mean(ratios))

        # Compute aggregate metrics based on collected data
        stats = index.compute_stats()
        metrics = EvaluationMetrics(
            name=self.name,
            recall=float(np.mean(recalls)),
            precision=float(np.mean(precisions)),
            f1_score=self._compute_f1(float(np.mean(recalls)), float(np.mean(precisions))),
            average_query_time=float(np.mean(query_times)),
            median_query_time=float(np.median(query_times)),
            query_time_95th_percentile=float(np.percentile(query_times, 95)),
            queries_per_second=len(self.query_vectors) / total_time if total_time > 0 else 0.0,
            distance_computations_per_query=total_distance_computations // len(self.query_vectors) if len(
                self.query_vectors) > 0 else 0,
            distance_ratio=np.mean(distance_ratios) if distance_ratios else 0.0,
            memory_usage_bytes=stats.get("memory_usage_bytes", 0),
            level_distribution=stats.get("level_distribution", {}),
            average_out_degree=stats.get("average_out_degree", 0.0),
            construction_time=index.get_construction_time()
        )

        # Log the evaluation results
        metrics.log()
        return metrics

    @staticmethod
    def _compute_f1(recall: float, precision: float) -> float:
        """
        Computes the F1 score given recall and precision.

        Args:
            recall (float): The recall value.
            precision (float): The precision value.

        Returns:
            float: The F1 score.
        """
        # Calculate F1 score only if both precision and recall are non-zero
        if recall + precision > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0

def evaluate_end_to_end(
        algorithm_name: str,
        algorithm_index: Any,
        config: dict,
        k: int,
        base_vectors: np.ndarray,
        query_vectors: np.ndarray,
        groundtruth: np.ndarray,
        ef_values: Optional[List[int]] = None
):
    """
    Evaluates the HNSW index from start to end, including building and querying.

    Args:
        algorithm_name (str): Name of the algorithm.
        algorithm_index (Any): The HNSW index class.
        config (dict): Configuration parameters for the index.
        k (int): Number of nearest neighbors to consider.
        base_vectors (np.ndarray): Base dataset vectors.
        query_vectors (np.ndarray): Query dataset vectors.
        groundtruth (np.ndarray): Ground truth indices.
        ef_values (Optional[List[int]]): List of ef_search values for evaluation.

    Returns:
        tuple: A list of results and the index as a string.
    """
    # Initialize the index with the specified dimensions and configuration
    dim = base_vectors.shape[1]
    index = algorithm_index(dim=dim, **config)

    if ef_values is None:
        # Default ef_search values if none provided
        ef_values = [10, 24, 32, 48, 64, 128]

    # Build the index with all base vectors
    logging.info("Building index...")
    index.batch_insert(base_vectors)

    # Initialize evaluator for the complete dataset
    evaluator = HNSWEvaluator(algorithm_name, base_vectors, query_vectors, groundtruth, k=k)
    results = []
    for ef in ef_values:
        # Evaluate the index for each ef_search value
        logging.info(f"\nEvaluating with ef_search = {ef}")
        result = evaluator.evaluate(index, ef)
        result_dict = result.to_dict()
        result_dict["ef_search"] = ef
        results.append(result_dict)

    return results, str(index)

def evaluate_batched(
        algorithm_name: str,
        algorithm_index: Any,
        config: dict,
        k: int,
        base_vectors: np.ndarray,
        query_vectors: np.ndarray,
        groundtruth: np.ndarray,
        ef_values: Optional[List[int]] = None
):
    """
    Evaluates the HNSW index in batches, simulating incremental indexing.

    Args:
        algorithm_name (str): Name of the algorithm.
        algorithm_index (Any): The HNSW index class.
        config (dict): Configuration parameters for the index.
        k (int): Number of nearest neighbors to consider.
        base_vectors (np.ndarray): Base dataset vectors.
        query_vectors (np.ndarray): Query dataset vectors.
        groundtruth (np.ndarray): Ground truth indices.
        ef_values (Optional[List[int]]): List of ef_search values for evaluation.

    Returns:
        tuple: Results for each batch and the index as a string.
    """
    # Determine dimensionality and initialize the index
    dim = base_vectors.shape[1]
    index = algorithm_index(dim=dim, **config)

    if ef_values is None:
        # Use default ef_search values if none provided
        ef_values = [10, 24, 32, 48, 64, 128]

    # Split base vectors into two batches for incremental building
    first_batch_size = len(base_vectors) // 2
    first_batch_vectors = base_vectors[:first_batch_size]
    second_batch_vectors = base_vectors[first_batch_size:]

    # Build the index with the first batch of vectors
    logging.info("Building index (1/2)...")
    index.batch_insert(first_batch_vectors)

    # Evaluate the index after the first batch
    evaluator_1 = HNSWEvaluator(f"before: {algorithm_name}", first_batch_vectors, query_vectors, k=k)
    results_1 = []
    for ef in ef_values:
        # Evaluate with each ef_search value
        logging.info(f"\nEvaluating with ef_search = {ef}")
        result = evaluator_1.evaluate(index, ef)
        result_dict = result.to_dict()
        result_dict["ef_search"] = ef
        results_1.append(result_dict)

    # Build the index with the second batch of vectors
    logging.info("Building index (2/2)...")
    index.batch_insert(second_batch_vectors)

    # Evaluate the index after the second batch
    evaluator_2 = HNSWEvaluator(f"after: {algorithm_name}", base_vectors, query_vectors, k=k,
                                groundtruth_vectors=groundtruth)
    results_2 = []
    for ef in ef_values:
        # Evaluate with each ef_search value
        logging.info(f"\nEvaluating with ef_search = {ef}")
        result = evaluator_2.evaluate(index, ef)
        result_dict = result.to_dict()
        result_dict["ef_search"] = ef
        results_2.append(result_dict)

    return results_1, results_2, str(index)
