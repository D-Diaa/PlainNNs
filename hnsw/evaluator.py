import copy
import json
import logging
import os
from dataclasses import dataclass, asdict
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from clustered import ClusteredHNSW, ClusterConfig
from index import HNSW, HNSWConfig
from utils import DATASET_CONFIGS, DatasetLoader

logging.basicConfig(level=logging.INFO)


@dataclass
class EvaluationMetrics:
    """
    Dataclass to store evaluation metrics for HNSW indices.
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
        """Convert the EvaluationMetrics instance to a dictionary."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'EvaluationMetrics':
        """Create an EvaluationMetrics instance from a dictionary."""
        return EvaluationMetrics(**data)


def _log_metrics(metrics: EvaluationMetrics):
    """
    Log evaluation metrics in a structured format.

    Args:
        metrics (EvaluationMetrics): The metrics to log.
    """
    logging.info("\nEvaluation Results:")
    logging.info(f"Recall: {metrics.recall:.4f}")
    logging.info(f"Precision: {metrics.precision:.4f}")
    logging.info(f"F1 Score: {metrics.f1_score:.4f}")
    logging.info(f"Average Query Time: {metrics.average_query_time * 1000:.2f}ms")
    logging.info(f"Median Query Time: {metrics.median_query_time * 1000:.2f}ms")
    logging.info(f"95th Percentile Query Time: {metrics.query_time_95th_percentile * 1000:.2f}ms")
    logging.info(f"Queries per Second: {metrics.queries_per_second:.2f}")
    logging.info(f"Average Distance Computations per Query: {metrics.distance_computations_per_query:.2f}")
    logging.info(f"Memory Usage: {metrics.memory_usage_bytes / (1024 * 1024):.2f}MB")
    logging.info(f"Construction Time: {metrics.construction_time:.2f}s")
    logging.info("\nLevel Distribution:")
    for level, count in sorted(metrics.level_distribution.items()):
        logging.info(f"Level {level}: {count} nodes")


class HNSWEvaluator:
    """
    Class to evaluate the performance of HNSW indices.
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
        Initialize the HNSWEvaluator.

        Args:
            name (str): Name of the evaluator.
            base_vectors (np.ndarray): Base vectors for indexing.
            query_vectors (np.ndarray): Query vectors for evaluation.
            groundtruth_vectors (Optional[np.ndarray]): Ground truth indices.
            recompute_ground_truth (bool): Whether to recompute ground truth.
            k (int): Number of nearest neighbors to consider.
        """
        self.base_vectors = base_vectors
        self.query_vectors = query_vectors
        self.k = k
        self.name = name
        self.recompute_ground_truth = recompute_ground_truth
        self.groundtruth_indices = groundtruth_vectors

        if self.groundtruth_indices is None or self.recompute_ground_truth:
            self._compute_ground_truth()

    def _compute_ground_truth(self):
        """
        Compute exact nearest neighbors using brute-force.
        """
        logging.info("Computing exact nearest neighbors...")
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='brute').fit(self.base_vectors)
        self.ground_truth_distances, self.groundtruth_indices = nbrs.kneighbors(self.query_vectors)
        logging.info("Ground truth computation complete")

    def evaluate(self, index: HNSW, ef_search: int) -> EvaluationMetrics:
        """
        Evaluate the HNSW index performance.

        Args:
            index (HNSW): Trained HNSW index.
            ef_search (int): Size of dynamic candidate list for search.

        Returns:
            EvaluationMetrics: Object containing various performance metrics.
        """
        total_time = 0.0
        total_distance_computations = 0
        recalls: List[float] = []
        precisions: List[float] = []
        distance_ratios: List[float] = []
        query_times: List[float] = []

        logging.info(f"Evaluating queries with ef_search={ef_search}...")

        for i in range(len(self.query_vectors)):
            # Search with HNSW
            result = index.search(self.query_vectors[i], self.k, ef_search)

            # Record metrics
            query_times.append(result.query_time)
            total_time += result.query_time
            total_distance_computations += index.distance_computations

            # Compute recall and precision
            ground_truth_set = set(self.groundtruth_indices[i][:self.k])
            result_set = set(result.indices)

            intersection = len(ground_truth_set.intersection(result_set))
            recalls.append(intersection / len(ground_truth_set))
            precisions.append(intersection / len(result_set))

            # Compute distance ratio if ground truth distances available
            if hasattr(self, 'ground_truth_distances'):
                hnsw_distances = np.sqrt(result.distances)
                exact_distances = self.ground_truth_distances[i]
                exact_distances = np.maximum(exact_distances, 1e-10)  # Avoid division by zero
                ratios = hnsw_distances / exact_distances
                distance_ratios.append(np.mean(ratios))

        # Compute aggregate metrics
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

        _log_metrics(metrics)
        return metrics

    @staticmethod
    def _compute_f1(recall: float, precision: float) -> float:
        """
        Compute F1 score from recall and precision.

        Args:
            recall (float): Recall value.
            precision (float): Precision value.

        Returns:
            float: Computed F1 score.
        """
        if recall + precision > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0


def evaluate(
        algorithm_name: str,
        algorithm_index: Any,
        config: dict,
        k: int,
        base_vectors: np.ndarray,
        query_vectors: np.ndarray,
        groundtruth: np.ndarray
):
    dim = base_vectors.shape[1]
    index = algorithm_index(dim=dim, **config)

    # Build index
    logging.info("Building index...")
    index.batch_insert(base_vectors)

    # Initialize evaluator
    evaluator = HNSWEvaluator(algorithm_name, base_vectors, query_vectors, groundtruth, k=k)
    results = []

    # Evaluate with different ef_search values
    ef_search_values = [10, 24, 32, 48, 64, 128]
    for ef in ef_search_values:
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
        groundtruth: np.ndarray
):
    dim = base_vectors.shape[1]
    index = algorithm_index(dim=dim, **config)

    # Split base vectors into two batches
    first_batch_size = len(base_vectors) // 2
    first_batch_vectors = base_vectors[:first_batch_size]
    second_batch_vectors = base_vectors[first_batch_size:]

    # Build index with the first batch
    logging.info("Building index (1/2)...")
    index.batch_insert(first_batch_vectors)

    # Initialize evaluator for the first batch
    evaluator_1 = HNSWEvaluator(f"before: {algorithm_name}", first_batch_vectors, query_vectors, k=k)
    results_1 = []
    ef_search_values = [10, 24, 32, 48, 64, 128]
    for ef in ef_search_values:
        logging.info(f"\nEvaluating with ef_search = {ef}")
        result = evaluator_1.evaluate(index, ef)
        result_dict = result.to_dict()
        result_dict["ef_search"] = ef
        results_1.append(result_dict)

    # Build index with the second batch
    logging.info("Building index (2/2)...")
    index.batch_insert(second_batch_vectors)

    # Initialize evaluator for the second batch
    evaluator_2 = HNSWEvaluator(f"after: {algorithm_name}", base_vectors, query_vectors, k=k,
                                groundtruth_vectors=groundtruth)
    results_2 = []
    for ef in ef_search_values:
        logging.info(f"\nEvaluating with ef_search = {ef}")
        result = evaluator_2.evaluate(index, ef)
        result_dict = result.to_dict()
        result_dict["ef_search"] = ef
        results_2.append(result_dict)

    return results_1, results_2, str(index)


def evaluate_task_full(
        dataset: str,
        algorithm_name: str,
        algorithm_class: Any,
        config: dict,
        k: int,
        base_vectors: np.ndarray,
        query_vectors: np.ndarray,
        groundtruth: np.ndarray,
        file_lock: Any
):
    filename = f'results/{dataset}/summary.json'
    if is_in_json(filename, algorithm_name, str(algorithm_class(dim=base_vectors.shape[1], **config)), file_lock):
        logging.info(f"Skipping {algorithm_name} on {dataset} with config={config}...")
        return
    logging.info(f"\nEvaluating {algorithm_name} on {dataset} with config={config}...")
    result, instance_str = evaluate(algorithm_name, algorithm_class, config, k, base_vectors, query_vectors,
                                    groundtruth)

    # Update JSON file immediately
    os.makedirs(f'results/{dataset}', exist_ok=True)
    update_json_file(filename, algorithm_name, instance_str, result, file_lock)


def evaluate_task_batched(
        dataset: str,
        algorithm_name: str,
        algorithm_class: Any,
        config: dict,
        k: int,
        base_vectors: np.ndarray,
        query_vectors: np.ndarray,
        groundtruth: np.ndarray,
        file_lock: Any
):
    filename = f'results/{dataset}/summary.json'
    dim = base_vectors.shape[1]
    before = is_in_json(
        filename,
        f"before: {algorithm_name}",
        str(algorithm_class(dim=dim, **config)),
        file_lock
    )
    after = is_in_json(
        filename,
        f"after: {algorithm_name}",
        str(algorithm_class(dim=dim, **config)),
        file_lock
    )
    if before and after:
        logging.info(f"Skipping {algorithm_name} on {dataset} with config={config}, using batched...")
        return
    logging.info(f"\nEvaluating {algorithm_name} on {dataset} with config={config}, using batched...")
    results_1, results_2, instance_str = evaluate_batched(algorithm_name, algorithm_class, config, k, base_vectors,
                                                          query_vectors,
                                                          groundtruth)

    os.makedirs(f'results/{dataset}', exist_ok=True)
    update_json_file(filename, f"before: {algorithm_name}", instance_str, results_1, file_lock)
    update_json_file(filename, f"after: {algorithm_name}", instance_str, results_2, file_lock)


def is_in_json(
        filename: str,
        algorithm_name: str,
        instance_str: str,
        lock: Any
):
    with lock:
        # Read existing data or create new structure
        if Path(filename).exists():
            with open(filename, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logging.warning(f"JSON decode error in {filename}. Starting with empty data.")
                    return False
        else:
            return False
        if algorithm_name not in data:
            return False
        if instance_str not in data[algorithm_name]:
            return False
        return True


def update_json_file(
        filename: str,
        algorithm_name: str,
        instance_str: str,
        results: List[Dict[str, Any]],
        lock: Any
):
    with lock:
        # Read existing data or create new structure
        if Path(filename).exists():
            with open(filename, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logging.warning(f"JSON decode error in {filename}. Starting with empty data.")
                    data = {}
        else:
            data = {}

        # Update data structure
        if algorithm_name not in data:
            data[algorithm_name] = {}
        data[algorithm_name][instance_str] = results

        # Write updated data back to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    algorithms: Dict[str, Any] = {
        'HNSW': HNSW,
        'ClusteredHNSW': ClusteredHNSW
    }
    datasets: List[str] = ['siftsmall', 'sift']
    alg_confs: Dict[str, List[dict]] = {
        "HNSW": [
            {"config": HNSWConfig.create(50)}
        ],
        "ClusteredHNSW": [
            {"hnsw_config": HNSWConfig.create(50),
             "cluster_config": ClusterConfig(average_cluster_size=1000, insert_method="ClusterThenInsert")},
            {"hnsw_config": HNSWConfig.create(50),
             "cluster_config": ClusterConfig(average_cluster_size=500, insert_method="ClusterThenInsert")},
            {"hnsw_config": HNSWConfig.create(50),
             "cluster_config": ClusterConfig(average_cluster_size=100, insert_method="ClusterThenInsert")},
            {"hnsw_config": HNSWConfig.create(50),
             "cluster_config": ClusterConfig(average_cluster_size=50, insert_method="ClusterThenInsert")},
            {"hnsw_config": HNSWConfig.create(50),
             "cluster_config": ClusterConfig(average_cluster_size=20, insert_method="ClusterThenInsert")},
        ]
    }
    extra_tasks_batched = [
        {"hnsw_config": HNSWConfig.create(50),
         "cluster_config": ClusterConfig(average_cluster_size=1000, insert_method="InsertBlindly")},
        {"hnsw_config": HNSWConfig.create(50),
         "cluster_config": ClusterConfig(average_cluster_size=500, insert_method="InsertBlindly")},
        {"hnsw_config": HNSWConfig.create(50),
         "cluster_config": ClusterConfig(average_cluster_size=100, insert_method="InsertBlindly")},
        {"hnsw_config": HNSWConfig.create(50),
         "cluster_config": ClusterConfig(average_cluster_size=50, insert_method="InsertBlindly")},
        {"hnsw_config": HNSWConfig.create(50),
         "cluster_config": ClusterConfig(average_cluster_size=20, insert_method="InsertBlindly")},
        {"hnsw_config": HNSWConfig.create(50),
         "cluster_config": ClusterConfig(average_cluster_size=1000, maximum_cluster_size=2000,
                                         insert_method="InsertWithMitosis")},
        {"hnsw_config": HNSWConfig.create(50),
         "cluster_config": ClusterConfig(average_cluster_size=500, maximum_cluster_size=1000,
                                         insert_method="InsertWithMitosis")},
        {"hnsw_config": HNSWConfig.create(50),
         "cluster_config": ClusterConfig(average_cluster_size=100, maximum_cluster_size=200,
                                         insert_method="InsertWithMitosis")},
        {"hnsw_config": HNSWConfig.create(50),
         "cluster_config": ClusterConfig(average_cluster_size=50, maximum_cluster_size=100,
                                         insert_method="InsertWithMitosis")},
        {"hnsw_config": HNSWConfig.create(50),
         "cluster_config": ClusterConfig(average_cluster_size=20, maximum_cluster_size=40,
                                         insert_method="InsertWithMitosis")},
    ]

    k: int = 10

    # Use Manager to share locks across processes
    manager = Manager()
    dataset_loader = DatasetLoader()
    file_locks: Dict[str, Any] = {dataset: manager.Lock() for dataset in datasets}

    # Load datasets once
    loaded_datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dataset in datasets:
        config = DATASET_CONFIGS.get(dataset)
        if config is None:
            logging.error(f"No configuration found for dataset '{dataset}'. Skipping.")
            continue
        loaded_datasets[dataset] = dataset_loader.load_dataset(
            config['url'],
            config['name']
        )
    all_tasks = []
    for task in ['batched', 'full']:
        task_fn = evaluate_task_full if task == 'full' else evaluate_task_batched
        for dataset in datasets:
            if dataset not in loaded_datasets:
                logging.warning(f"Dataset '{dataset}' not loaded. Skipping evaluation.")
                continue

            base_vectors, query_vectors, groundtruth = loaded_datasets[dataset]
            for algorithm_name, algorithm_class in algorithms.items():
                if task == 'batched' and algorithm_name == 'HNSW':
                    continue
                algorithm_configs = copy.deepcopy(alg_confs[algorithm_name])
                if algorithm_name == "ClusteredHNSW" and task == 'batched':
                    algorithm_configs += extra_tasks_batched
                for alg_conf in algorithm_configs:
                    all_tasks.append((
                        task_fn,
                        dataset,
                        algorithm_name,
                        algorithm_class,
                        alg_conf,
                        k,
                        base_vectors,
                        query_vectors,
                        groundtruth,
                        file_locks[dataset],
                    ))
    all_tasks = sorted(all_tasks, key=lambda x: (x[0].__name__, len(loaded_datasets[x[1]]), x[2]))
    n_processes = min(64, len(all_tasks), os.cpu_count())
    with Pool(processes=n_processes) as pool:
        pool.starmap(execute_task_wrapper, all_tasks)


def execute_task_wrapper(task_fn, *args):
    """Wrapper function to unpack arguments and execute the task function"""
    try:
        task_fn(*args)
    except Exception as e:
        logging.error(f"Error executing task: {str(e)}")
        raise


if __name__ == "__main__":
    main()
