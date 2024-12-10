import torch


def euclidean_matrix_chunked(vectors, chunk_size=1000):
    """
    Memory-efficient computation of pairwise Euclidean distances using chunking.
    """
    n = vectors.size(0)
    distances = torch.zeros(n, n, device=vectors.device)

    # Compute squared norms once
    norms_squared = (vectors ** 2).sum(dim=1, keepdim=True)

    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk_i = vectors[i:end_i]
        norms_i = norms_squared[i:end_i]

        # Compute distances for this chunk against all vectors
        # Using the formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        prod = chunk_i @ vectors.T
        distances[i:end_i] = norms_i + norms_squared.T - 2 * prod

    # Mirror the upper triangular part
    distances = (distances + distances.T) / 2

    return torch.clamp(distances, min=1e-7)


def hamming_matrix_chunked(codes, chunk_size=1000):
    """
    Memory-efficient computation of Hamming distances using chunking.
    """
    n = codes.size(0)
    hamming = torch.zeros(n, n, device=codes.device)

    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk_i = codes[i:end_i]

        for j in range(0, n, chunk_size):
            end_j = min(j + chunk_size, n)
            chunk_j = codes[j:end_j]

            # Compute XOR for this chunk pair
            xor = chunk_i.unsqueeze(1) ^ chunk_j.unsqueeze(0)
            hamming[i:end_i, j:end_j] = xor.float().count_ones()

    return hamming


def hamming_matrix(codes):
    # Step 1: Compute the bitwise XOR between all pairs of codes
    xor = codes.unsqueeze(1) ^ codes.unsqueeze(0)

    # Step 2: Count the number of set bits in the XOR result
    hamming = xor.float().count_ones()

    return hamming


def hamming_triu(codes):
    hamming = hamming_matrix(codes)
    # Step 3: Extract the upper triangular part (without the diagonal)
    i, j = torch.triu_indices(codes.size(0), codes.size(0), offset=1)
    return hamming[i, j]


def euclidean_matrix(vectors):
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


def euclidean_triu(vectors):
    pairwise_distances = euclidean_matrix(vectors)
    # Step 5: Extract the upper triangular part (without the diagonal)
    i, j = torch.triu_indices(vectors.size(0), vectors.size(0), offset=1)
    return pairwise_distances[i, j]


def prepare_data(base_vectors, batch_size=128):
    num_dataset = base_vectors.shape[0]
    dim = base_vectors.shape[1]
    # base_vectors = (base_vectors - base_vectors.mean(axis=0)) / base_vectors.std(axis=0)
    dataset_loader = torch.utils.data.DataLoader(
        dataset=torch.from_numpy(base_vectors).float(),
        batch_size=batch_size, shuffle=False, drop_last=True)
    return dataset_loader, num_dataset, dim
