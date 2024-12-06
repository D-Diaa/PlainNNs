import json
import logging
import os
import time
from math import ceil, log2, floor

import numpy as np
from tqdm import tqdm

from utils import download_and_extract, read_fvecs, read_ivecs, calculate_mean_and_ci

# Configure logging for better readability
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


# Benchmark nearest neighbor search algorithm

def benchmark_nn(dataset_name, dataset_url, nn_algorithm, num_queries=10, k=10, evaluate_pruned=False):
    # Create path to store datasets
    datasets_path = os.path.join(os.getcwd(), 'datasets')
    os.makedirs(datasets_path, exist_ok=True)  # Ensure the directory exists

    # Download and extract dataset
    logging.info(f"Downloading and extracting dataset {dataset_name}...")
    local_name = download_and_extract(dataset_url, datasets_path)

    # Load base vectors used to build the graph
    base_file = os.path.join(datasets_path, local_name, local_name + '_base.fvecs')
    logging.info(f"Loading base vectors from {base_file}...")
    base_vectors = np.array(list(read_fvecs(base_file)))  # Reading all vectors from file
    logging.info(f"Loaded {len(base_vectors)} base vectors.")

    # Load query vectors to use for the nearest neighbor search
    query_file = os.path.join(datasets_path, local_name, local_name + '_query.fvecs')
    logging.info(f"Loading query vectors from {query_file}...")
    query_vectors = np.array(list(read_fvecs(query_file)))  # Reading all query vectors
    logging.info(f"Loaded {len(query_vectors)} query vectors.")

    # Load ground truth values for evaluating recall and accuracy
    groundtruth_file = os.path.join(datasets_path, local_name, local_name + '_groundtruth.ivecs')
    logging.info(f"Loading ground truth from {groundtruth_file}...")
    groundtruth_vectors = np.array(list(read_ivecs(groundtruth_file)))  # Reading ground truth vectors
    logging.info(f"Loaded {len(groundtruth_vectors)} ground truth vectors.")
    log_N = ceil(log2(len(base_vectors)))
    kwargs = {
        "m": log_N,
    }
    if nn_algorithm.__name__ == "HypercubeNN":
        # TODO: implement different initialization strategies
        kwargs["init_strategy"] = "random"
        kwargs["permutation_strategy"] = "bihalf"
        # deleting nodes bigger than bucket size makes this an important parameter, the bigger the better
        # but now we push these nodes to nearby neighbors so bigger than log_N is not necessarily better
        kwargs["bucket_size"] = log_N
        # Bigger m implies sparser code implies less connected graph (even at bigger degree)
        # So, m should not be too big
        # For exactly a power of 2 nodes, m = log_N is the best
        # Otherwise, m = log_N - 1 or m = log_N - 2 is the best
        # setting it as floor for now
        kwargs["m"] = floor(log2(len(base_vectors)))

    nn_algorithm = nn_algorithm(**kwargs)

    # Initialize the nearest neighbor algorithm
    logging.info(f"Initializing {nn_algorithm.__class__.__name__}...")
    nn_algorithm.batch_insert(base_vectors, bfs_width=log_N)  # Build the graph using base vectors

    def evaluate(key_prefix=""):
        # Perform benchmark on the queries
        query_times, recalls, ratios, num_hops_list = [], [], [], []
        for i, query in enumerate(tqdm(query_vectors[:num_queries], desc="Benchmarking Queries")):
            # Measure the time to find k nearest neighbors
            start_time = time.time()  # Start the timer
            neighbors, num_hops = nn_algorithm.find(query, k=k, bfs_width=log_N)  # Find k nearest neighbors
            elapsed_time = time.time() - start_time  # Calculate time taken
            query_times.append(elapsed_time)

            # Calculate recall: how many of the retrieved neighbors are correct
            ground_truth = groundtruth_vectors[i][:k]  # Get the true nearest neighbors
            neighbor_indices = set()
            for neighbor in neighbors:
                neighbor_indices.update(nn_algorithm.graph.nodes[neighbor]['label'])
            recall = len(neighbor_indices.intersection(set(ground_truth))) / len(ground_truth)
            recalls.append(recall)

            # Calculate ratio of distances (distance of retrieved neighbor / distance of true neighbor)
            true_distance = np.linalg.norm(query - base_vectors[ground_truth[0]])  # Distance to true neighbor
            neighbor_vectors = [nn_algorithm.graph.nodes[neighbor]['value'] for neighbor in neighbors]
            returned_distance = min([np.linalg.norm(vector - query, axis=1).min() for vector in neighbor_vectors])
            num_hops_list.append(num_hops)  # Track number of queries made
            ratios.append(returned_distance / true_distance)

        # Calculate average metrics with 95% confidence intervals
        average_time, time_ci = calculate_mean_and_ci(query_times)
        recall_mean, recall_ci = calculate_mean_and_ci(recalls)
        ratio_mean, ratio_ci = calculate_mean_and_ci(ratios)
        query_mean, query_ci = calculate_mean_and_ci(num_hops_list)
        ratio_max = float(np.max(ratios)) if ratios else 0.0
        query_max = float(np.max(num_hops_list)) if num_hops_list else 0.0

        key = key_prefix + nn_algorithm.__class__.__name__

        # Log benchmark results
        logging.info(f"Benchmark Results for {dataset_name} with {key}:")
        logging.info(f"Average Query Time: {average_time:.4f} seconds, 95% CI: ({time_ci[0]:.4f}, {time_ci[1]:.4f})")
        logging.info(f"Recall Mean@{k}: {recall_mean:.4f}, 95% CI: ({recall_ci[0]:.4f}, {recall_ci[1]:.4f})")
        logging.info(f"Ratio Mean: {ratio_mean:.4f}, 95% CI: ({ratio_ci[0]:.4f}, {ratio_ci[1]:.4f})")
        logging.info(f"Ratio Max: {ratio_max:.4f}")
        logging.info(f"Query Mean: {query_mean:.4f}, 95% CI: ({query_ci[0]:.4f}, {query_ci[1]:.4f})")
        logging.info(f"Query Max: {query_max:.4f}")

        logging.info(f"Graph Summary:")
        graph_stats = nn_algorithm.summary()

        # Save results to a JSON file for reference
        results = {
            "average_query_time": average_time,
            "time_95_ci": time_ci,
            "recall_mean": recall_mean,
            "recall_95_ci": recall_ci,
            "ratio_mean": ratio_mean,
            "ratio_95_ci": ratio_ci,
            "ratio_max": ratio_max,
            "query_mean": query_mean,
            "query_95_ci": query_ci,
            "query_max": query_max
        }
        results.update(graph_stats)
        result_file_path = os.path.join(datasets_path, local_name, 'benchmark_results.json')
        # Load existing results if the JSON file already exists
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as json_file:
                existing_results = json.load(json_file)
            existing_results[key] = results

        else:
            existing_results = {key: results}

        # Save updated results
        with open(result_file_path, 'w') as json_file:
            json.dump(existing_results, json_file, indent=4)  # Write the results to a JSON file

    evaluate()
    if evaluate_pruned:
        nn_algorithm.prune()  # Prune the graph to ensure each node has at most m neighbors
        evaluate("Pruned ")


if __name__ == "__main__":
    from nsw_nn import NSWNN
    from graph_nn import LinearNN, RandomNN
    from hypercube_nn import HypercubeNN

    algorithms_dict = {
        "LinearNN": LinearNN,
        "RandomNN": RandomNN,
        "NSWNN": NSWNN,
        "HypercubeNN": HypercubeNN
    }

    # Define datasets to be benchmarked with their URLs
    datasets = {
        "ANN_SIFT10K": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
        # "ANN_SIFT1M": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        # "ANN_GIST1M": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
    }
    algorithms = [
        # "LinearNN",
        # "RandomNN",
        # "NSWNN",
        "HypercubeNN"
    ]
    # Run benchmark for each dataset
    for dataset_name, dataset_url in datasets.items():
        for algorithm in algorithms:
            algorithm_class = algorithms_dict[algorithm]
            evaluate_pruned = algorithm == "NSWNN"  # Evaluate pruned version for NSWNN
            benchmark_nn(dataset_name, dataset_url, algorithm_class,
                         num_queries=10000, k=10, evaluate_pruned=evaluate_pruned)
