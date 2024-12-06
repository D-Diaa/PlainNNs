from abc import abstractmethod

import torch
from torch import nn

from utils import euclidean_matrix

class HashLoss(nn.Module):
    def __init__(self):
        super(HashLoss, self).__init__()

    @abstractmethod
    def forward(self, codes, info):
        pass

    @abstractmethod
    def prepare_batch(self, batch_vectors):
        pass


class MSELoss(HashLoss):
    def forward(self, codes, info):
        distances = info
        # calculate pairwise distances in hash space
        hash_distances = euclidean_matrix(codes)
        if distances.max() > 0:
            distances = distances / distances.max()
        return torch.mean((hash_distances - distances) ** 2)

    def prepare_batch(self, batch_vectors):
        distances = euclidean_matrix(batch_vectors)
        return batch_vectors, distances


class TripletLoss(HashLoss):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, codes, info):
        anchor_idx, positive_idx, negative_idx = info
        anchor = codes[anchor_idx]
        positive = codes[positive_idx]
        negative = codes[negative_idx]

        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)

        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

    def prepare_batch(self, batch_vectors):
        # Compute pairwise distance matrix
        distances = euclidean_matrix(batch_vectors)

        # each sample is an anchor, closest is positive, furthest is negative
        anchor_idx = torch.arange(batch_vectors.size(0))
        # Create a mask for the diagonal (self-distances)
        mask = torch.eye(batch_vectors.size(0), dtype=torch.bool, device=batch_vectors.device)

        # For Positive Indices (Closest)
        distance_matrix_pos = distances.clone()
        distance_matrix_pos.masked_fill_(mask, float('inf'))  # Ignore self by setting to infinity
        positive_idx = torch.argmin(distance_matrix_pos, dim=1)

        # For Negative Indices (Furthest)
        distance_matrix_neg = distances.clone()
        distance_matrix_neg.masked_fill_(mask, -float('inf'))  # Ignore self by setting to -infinity
        negative_idx = torch.argmax(distance_matrix_neg, dim=1)
        return batch_vectors, (anchor_idx, positive_idx, negative_idx)


class ContrastiveLoss(HashLoss):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, codes, info):
        pairs, labels = info
        code1, code2 = codes[pairs[:, 0]], codes[pairs[:, 1]]
        hash_distances = torch.sum((code1 - code2) ** 2, dim=1)

        positive_loss = labels * hash_distances
        negative_loss = (1 - labels) * torch.clamp(self.margin - hash_distances, min=0.0)

        return (positive_loss + negative_loss).mean()

    def prepare_batch(self, batch_vectors):
        distances = euclidean_matrix(batch_vectors)
        i, j = torch.triu_indices(batch_vectors.size(0), batch_vectors.size(0), offset=1)
        distances_upper = distances[i, j]
        # median of distances
        median_distance = torch.median(distances_upper)
        pairs = torch.stack([i, j], dim=1)
        # labels are 1 if distance < median distance of batch, 0 otherwise
        labels = (distances_upper < median_distance).float()
        return batch_vectors, (pairs, labels)


class BiHalfLoss(HashLoss):
    def __init__(self):
        super(BiHalfLoss, self).__init__()

    def forward(self, codes, info):
        x = info
        batch_size = codes.size(0)
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even for BiHalf loss")
        half_size = batch_size // 2
        b1, b2 = codes[:half_size], codes[half_size:]
        x1, x2 = x[:half_size], x[half_size:]

        target_b = torch.cosine_similarity(b1, b2)
        target_x = torch.cosine_similarity(x1, x2)

        return torch.mean((target_b - target_x) ** 2)

    def prepare_batch(self, batch_vectors):
        return batch_vectors, None


class BitIndependenceLoss(HashLoss):
    def __init__(self, weight=1.0):
        super(BitIndependenceLoss, self).__init__()
        self.weight = weight

    def forward(self, codes, info):
        if self.weight == 0:
            return torch.tensor(0.0, device=codes.device)
        # Center the codes
        centered_codes = codes - codes.mean(dim=0)

        # Compute correlation matrix
        n_samples = codes.size(0)
        correlation_matrix = torch.matmul(centered_codes.t(), centered_codes) / n_samples

        # Get off-diagonal elements
        mask = torch.ones_like(correlation_matrix) - torch.eye(correlation_matrix.size(0)).to(correlation_matrix.device)
        correlations = correlation_matrix * mask

        return self.weight * torch.mean(correlations ** 2)

    def prepare_batch(self, batch_vectors):
        return batch_vectors, None


class HashingLoss(HashLoss):
    def __init__(self, alpha=0, beta=1.0):
        super(HashingLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, codes, info):
        distances = info
        # Quantization loss: encourage outputs to be binary (-1 or 1)
        quantization_loss = 1 - torch.pow(codes, 2).mean()

        # Normalize distances to [0, 1]
        if distances.max() > 0:
            distances = distances / distances.max()

        # Compute pairwise hamming-like distances in hash space
        hash_pdist = euclidean_matrix(codes)
        hash_pdist = hash_pdist / codes.size(1)  # Normalize by hash dimension

        # Calculate similarity preservation loss
        similarity_loss = torch.mean(torch.pow(distances - hash_pdist, 2))

        return self.alpha * quantization_loss + self.beta * similarity_loss

    def prepare_batch(self, batch_vectors):
        distances = euclidean_matrix(batch_vectors)
        return batch_vectors, distances


class CombinedHashLoss(HashLoss):
    def __init__(self, alpha=1.0, beta=2.0, gamma=0.5):
        super(CombinedHashLoss, self).__init__()
        self.alpha = alpha  # Weight for quantization
        self.beta = beta  # Weight for similarity preservation
        self.gamma = gamma  # Weight for ranking preservation

    def forward(self, codes, info):
        distances = info
        batch_size = codes.size(0)

        # 1. Quantization loss to encourage binary codes
        quantization_loss = 1 - torch.pow(codes, 2).mean()

        # 2. Similarity preservation loss
        if distances.max() > 0:
            distances = distances / distances.max()
        hash_pdist = euclidean_matrix(codes)
        hash_pdist = hash_pdist / codes.size(1)
        similarity_loss = torch.mean(torch.pow(distances - hash_pdist, 2))

        # 3. Ranking preservation loss - FIXED
        # Get relative orderings while avoiding numerical issues
        distances_sorted, distances_ranks = torch.sort(distances.view(-1))
        hash_sorted, hash_ranks = torch.sort(hash_pdist.view(-1))

        # Convert to normalized positions (0 to 1)
        n_pairs = distances_ranks.size(0)
        dist_positions = distances_ranks.float() / (n_pairs - 1)
        hash_positions = hash_ranks.float() / (n_pairs - 1)

        # Compute ranking loss using position differences
        # Sort both sets of positions to compare corresponding quantiles
        ranking_loss = torch.mean(torch.pow(
            torch.sort(dist_positions)[0] - torch.sort(hash_positions)[0],
            2
        ))

        # Combine losses with weights
        total_loss = (
                self.alpha * quantization_loss +
                self.beta * similarity_loss +
                self.gamma * ranking_loss
        )

        # For debugging
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Loss components:")
            print(f"Quantization: {quantization_loss.item():.4f}")
            print(f"Similarity: {similarity_loss.item():.4f}")
            print(f"Ranking: {ranking_loss.item():.4f}")
            raise ValueError("NaN or Inf in loss calculation")

        return total_loss

    def prepare_batch(self, batch_vectors):
        distances = euclidean_matrix(batch_vectors)
        return batch_vectors, distances
def build_loss(config):
    loss_type = config["loss_type"]
    if loss_type == "mse":
        return MSELoss()
    elif loss_type == "triplet":
        return TripletLoss(margin=config["margin"])
    elif loss_type == "contrastive":
        return ContrastiveLoss(margin=config["margin"])
    elif loss_type == "bihalf":
        return BiHalfLoss()
    elif loss_type == "combined":
        return CombinedHashLoss(alpha=config["alpha"], beta=config["beta"], gamma=config["gamma"])
    elif loss_type == "independence":
        return BitIndependenceLoss(weight=config["independence_weight"])
    elif loss_type == "hashing":
        return HashingLoss(alpha=config["alpha"], beta=config["beta"])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")