import tarfile
import urllib.request
import numpy as np
import os

import torch
from scipy import stats
def pdist(vectors):
    # Step 1: Compute the squared norms of the vectors (N,)
    norms_squared = (vectors ** 2).sum(dim=1, keepdim=True)

    # Step 2: Compute the distance matrix using the formula:
    # d(i,j)^2 = ||vi||^2 + ||vj||^2 - 2 * vi · vj
    distances_squared = norms_squared + norms_squared.T - 2 * vectors @ vectors.T

    # Step 3: Clip small negative values that may arise due to floating-point precision errors
    distances_squared = torch.clamp(distances_squared, min=1e-7)

    # Step 4: Take the square root to get the Euclidean distances
    # pairwise_distances = torch.sqrt(distances_squared)

    return distances_squared

def pdist_upper(vectors):
    pairwise_distances = pdist(vectors)
    # Step 5: Extract the upper triangular part (without the diagonal)
    i, j = torch.triu_indices(vectors.size(0), vectors.size(0), offset=1)
    return pairwise_distances[i, j]

def get_base_data(local_name):
    datasets_path = "datasets"
    base_file = os.path.join(datasets_path, local_name, local_name + '_base.fvecs')
    base_vectors = np.array(list(read_fvecs(base_file)))  # Reading all vectors from file
    return base_vectors
def prepare_data(base_vectors, batch_size=128):
    num_dataset = base_vectors.shape[0]
    dim = base_vectors.shape[1]
    dataset_loader = torch.utils.data.DataLoader(
        dataset=torch.from_numpy(base_vectors).float(),
        batch_size=batch_size, shuffle=False, drop_last=True)
    return dataset_loader, num_dataset, dim
# Helper function for grey code conversion
def grey(n):
    return n ^ (n >> 1)

# Calculate metrics with 95% confidence intervals
def calculate_mean_and_ci(data):
    mean = float(np.mean(data)) if data else 0.0
    ci = tuple(float(x) for x in
               (stats.norm.interval(0.95, loc=mean, scale=stats.sem(data)) if len(data) > 1 else (mean, mean)))
    return mean, ci
def euclidean_distance(a, b):
    """
    Compute the Euclidean distance between two points.

    Parameters:
    - a: The first point
    - b: The second point

    Returns:
    - Euclidean distance between points a and b
    """
    return sum((x - y) ** 2 for x, y in zip(a, b))


# Helper function to download and extract datasets
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
