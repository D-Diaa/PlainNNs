import argparse
import copy
import logging
import os
from multiprocessing import Manager, Pool
from typing import Dict, Tuple, List, Any, Optional

import numpy as np

from clustered_hnsw import ClusteredHNSW, ClusterConfig
from evaluator import evaluate_end_to_end, evaluate_batched
from hnsw import HNSW, HNSWConfig
from utils import DATASET_CONFIGS, DatasetLoader, update_json_file, is_in_json, execute_task_wrapper


def evaluate_end_to_end_wrapper(
        dataset: str,
        algorithm_name: str,
        algorithm_class: Any,
        config: dict,
        k: int,
        ef_values: Optional[List[int]],
        base_vectors: np.ndarray,
        query_vectors: np.ndarray,
        groundtruth: np.ndarray,
        file_lock: Any
):
    """
    Wrapper for evaluating an algorithm in an end-to-end fashion.

    Parameters:
        dataset (str): Name of the dataset.
        algorithm_name (str): Name of the algorithm.
        algorithm_class (Any): Class of the algorithm to be evaluated.
        config (dict): Configuration dictionary for the algorithm.
        k (int): Number of nearest neighbors to consider.
        ef_values (Optional[List[int]]): List of ef values to evaluate.
        base_vectors (np.ndarray): Base dataset vectors.
        query_vectors (np.ndarray): Query dataset vectors.
        groundtruth (np.ndarray): Ground truth nearest neighbors.
        file_lock (Any): Lock object to manage file access.

    Returns:
        None
    """
    # Check if results already exist in the summary file and skip if so
    filename = f'results/{dataset}/summary.json'
    if is_in_json(filename, algorithm_name, str(algorithm_class(dim=base_vectors.shape[1], **config)), file_lock):
        logging.info(f"Skipping {algorithm_name} on {dataset} with config={config}...")
        return

    logging.info(f"\nEvaluating {algorithm_name} on {dataset} with config={config}...")
    # Perform evaluation and store results
    result, instance_str = evaluate_end_to_end(algorithm_name, algorithm_class, config, k, base_vectors, query_vectors,
                                               groundtruth, ef_values)

    # Update the summary file with the evaluation results
    os.makedirs(f'results/{dataset}', exist_ok=True)
    update_json_file(filename, algorithm_name, instance_str, result, file_lock)


def evaluate_batched_wrapper(
        dataset: str,
        algorithm_name: str,
        algorithm_class: Any,
        config: dict,
        k: int,
        ef_values: Optional[List[int]],
        base_vectors: np.ndarray,
        query_vectors: np.ndarray,
        groundtruth: np.ndarray,
        file_lock: Any
):
    """
    Wrapper for evaluating an algorithm in batched mode.

    Parameters:
        dataset (str): Name of the dataset.
        algorithm_name (str): Name of the algorithm.
        algorithm_class (Any): Class of the algorithm to be evaluated.
        config (dict): Configuration dictionary for the algorithm.
        k (int): Number of nearest neighbors to consider.
        ef_values (Optional[List[int]]): List of ef values to evaluate.
        base_vectors (np.ndarray): Base dataset vectors.
        query_vectors (np.ndarray): Query dataset vectors.
        groundtruth (np.ndarray): Ground truth nearest neighbors.
        file_lock (Any): Lock object to manage file access.

    Returns:
        None
    """
    # Check if the "before" and "after" results are already in the summary file
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
    # Perform batched evaluation
    results_1, results_2, instance_str = evaluate_batched(algorithm_name, algorithm_class, config, k, base_vectors,
                                                          query_vectors, groundtruth, ef_values)

    # Update the summary file with "before" and "after" results
    os.makedirs(f'results/{dataset}', exist_ok=True)
    update_json_file(filename, f"before: {algorithm_name}", instance_str, results_1, file_lock)
    update_json_file(filename, f"after: {algorithm_name}", instance_str, results_2, file_lock)


def main(datasets=None, k: int = 10, ef_values=None):
    """
    Main function for managing the evaluation process.

    Loads datasets, prepares algorithm configurations, and evaluates
    algorithms in both batched and full modes using multiprocessing.

    Returns:
        None
    """
    # Set default values for datasets and ef_values
    if ef_values is None:
        ef_values = [10, 24, 32, 48, 64, 128]
    if datasets is None:
        datasets = ['siftsmall', 'sift']
    # Define the algorithms to evaluate
    algorithms: Dict[str, Any] = {
        'HNSW': HNSW,
        'ClusteredHNSW': ClusteredHNSW
    }

    # Define algorithm-specific configurations
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

    # Define additional batched tasks for ClusteredHNSW
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

    # Use Manager to share locks across processes
    manager = Manager()
    dataset_loader = DatasetLoader()
    file_locks: Dict[str, Any] = {dataset: manager.Lock() for dataset in datasets}

    # Load datasets once
    loaded_datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dataset in datasets:
        # Get the dataset configuration
        config = DATASET_CONFIGS.get(dataset)
        if config is None:
            logging.error(f"No configuration found for dataset '{dataset}'. Skipping.")
            continue
        # Load dataset and store it in the loaded_datasets dictionary
        loaded_datasets[dataset] = dataset_loader.load_dataset(
            config['url'],
            config['name']
        )

    all_tasks = []
    for task in ['batched', 'full']:
        # Select the appropriate evaluation function based on the task type
        task_fn = evaluate_end_to_end_wrapper if task == 'full' else evaluate_batched_wrapper
        for dataset in datasets:
            # Skip datasets that failed to load
            if dataset not in loaded_datasets:
                logging.warning(f"Dataset '{dataset}' not loaded. Skipping evaluation.")
                continue

            # Retrieve dataset vectors and ground truth
            base_vectors, query_vectors, groundtruth = loaded_datasets[dataset]
            for algorithm_name, algorithm_class in algorithms.items():
                # Skip batched evaluation for HNSW
                if task == 'batched' and algorithm_name == 'HNSW':
                    continue

                # Copy configurations and append extra tasks for batched evaluation
                algorithm_configs = copy.deepcopy(alg_confs[algorithm_name])
                if algorithm_name == "ClusteredHNSW" and task == 'batched':
                    algorithm_configs += extra_tasks_batched

                for alg_conf in algorithm_configs:
                    # Add task details to the task list
                    all_tasks.append((
                        task_fn,
                        dataset,
                        algorithm_name,
                        algorithm_class,
                        alg_conf,
                        k,
                        ef_values,
                        base_vectors,
                        query_vectors,
                        groundtruth,
                        file_locks[dataset],
                    ))

    # Sort tasks to balance load and ensure determinism
    all_tasks = sorted(all_tasks, key=lambda x: (x[0].__name__, len(loaded_datasets[x[1]]), x[2]))
    n_processes = min(64, len(all_tasks), os.cpu_count())

    # Use multiprocessing to execute tasks in parallel
    with Pool(processes=n_processes) as pool:
        pool.starmap(execute_task_wrapper, all_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HNSW and ClusteredHNSW on multiple datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="List of datasets to evaluate. Defaults to ['siftsmall', 'sift']. Choices: ['siftsmall', 'sift', gist]",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors to consider. Defaults to 10.",
    )
    parser.add_argument(
        "--ef_values",
        nargs="+",
        type=int,
        default=None,
        help="List of ef values to evaluate. Defaults to [10, 24, 32, 48, 64, 128].",
    )
    main(**vars(parser.parse_args()))
