import logging
import os
import tarfile
import urllib.request
from typing import Tuple

import numpy as np
from numba import njit, prange, literal_unroll

DATASET_CONFIGS = {
    'siftsmall': {
        'url': "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
        'name': 'siftsmall'
    },
    'sift': {
        'url': "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        'name': 'sift'
    },
    'gist': {
        'url': "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        'name': 'gist'
    },
}


class DatasetLoader:
    """
    Class responsible for downloading, extracting, and loading datasets.
    """

    def __init__(self, datasets_path: str = 'datasets'):
        """
        Initialize the DatasetLoader.

        Args:
            datasets_path (str): Path to store datasets.
        """
        self.datasets_path = datasets_path
        os.makedirs(datasets_path, exist_ok=True)

    def load_dataset(self, dataset_url: str, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Download, extract, and load dataset vectors.

        Args:
            dataset_url (str): URL to download the dataset.
            dataset_name (str): Name of the dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Base vectors, query vectors, and ground truth indices.
        """
        logging.info(f"Loading dataset {dataset_name}...")

        # Download and extract if needed
        local_name = download_and_extract(dataset_url, self.datasets_path)
        dataset_dir = os.path.join(self.datasets_path, local_name)

        # Load base vectors
        base_file = os.path.join(dataset_dir, f"{local_name}_base.fvecs")
        logging.info(f"Loading base vectors from {base_file}...")
        base_vectors = np.array(list(read_fvecs(base_file)))

        # Load query vectors
        query_file = os.path.join(dataset_dir, f"{local_name}_query.fvecs")
        logging.info(f"Loading query vectors from {query_file}...")
        query_vectors = np.array(list(read_fvecs(query_file)))

        # Load ground truth
        groundtruth_file = os.path.join(dataset_dir, f"{local_name}_groundtruth.ivecs")
        logging.info(f"Loading ground truth from {groundtruth_file}...")
        groundtruth_vectors = np.array(list(read_ivecs(groundtruth_file)))

        logging.info(
            f"Dataset loaded: {len(base_vectors)} base vectors, "
            f"{len(query_vectors)} query vectors, "
            f"{len(groundtruth_vectors)} ground truth vectors"
        )

        return base_vectors, query_vectors, groundtruth_vectors


def get_base_data(local_name):
    datasets_path = "datasets"
    base_file = os.path.join(datasets_path, local_name, local_name + '_base.fvecs')
    base_vectors = np.array(list(read_fvecs(base_file)))  # Reading all vectors from file
    return base_vectors


def download_and_extract(url, extract_path):
    local_filename = url.split('/')[-1]
    local_filepath = os.path.join(extract_path, local_filename)
    local_name = local_filename.split('.tar.gz')[0]
    local_extracted_path = os.path.join(extract_path, local_name)
    if not os.path.exists(local_filepath):
        with urllib.request.urlopen(url) as response:
            with open(local_filepath, 'wb') as f:
                f.write(response.read())

    if local_filename.endswith('.tar.gz') and not os.path.exists(local_extracted_path):
        with tarfile.open(local_filepath, "r:gz") as tar:
            tar.extractall(path=extract_path)
    return local_name


# Helper function to read .fvecs file format
def read_fvecs(filepath):
    with open(filepath, 'rb') as f:
        while True:
            bytes_ = f.read(4)
            if not bytes_:
                break
            dim = int(np.frombuffer(bytes_, dtype=np.int32)[0])
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            yield vec


# Helper function to read .ivecs file format
def read_ivecs(filepath):
    with open(filepath, 'rb') as f:
        while True:
            bytes_ = f.read(4)
            if not bytes_:
                break
            dim = int(np.frombuffer(bytes_, dtype=np.int32)[0])
            vec = np.frombuffer(f.read(dim * 4), dtype=np.int32)
            yield vec


@njit(fastmath=True)
def batch_distances(query_vec, indices, all_vectors):
    n = len(indices)
    distances = np.empty(n, dtype=np.float32)

    # Parallelize the outer loop
    for i in prange(n):
        idx = indices[i]
        dist = 0.0
        vec = all_vectors[idx]

        # Unroll the inner dimension for better vectorization
        for j in literal_unroll(range(0, len(query_vec), 4)):
            # Process 4 elements at a time when possible
            if j + 4 <= len(query_vec):
                diff0 = query_vec[j] - vec[j]
                diff1 = query_vec[j + 1] - vec[j + 1]
                diff2 = query_vec[j + 2] - vec[j + 2]
                diff3 = query_vec[j + 3] - vec[j + 3]

                dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3
            else:
                # Handle remaining elements
                for jj in range(j, len(query_vec)):
                    diff = query_vec[jj] - vec[jj]
                    dist += diff * diff

        distances[i] = dist

    return distances
