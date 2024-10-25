import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import *


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BiHalfEstimator(torch.autograd.Function):
    gamma = 6

    @staticmethod
    def forward(ctx, U):
        # Yunqiang for half and half (optimal transport)
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        B_creat = torch.cat((torch.ones([int(N / 2), D]), -torch.ones([N - int(N / 2), D]))).to(U.device)
        B = torch.zeros(U.shape).to(U.device).scatter_(0, index, B_creat)
        ctx.save_for_backward(U, B)
        return B

    @staticmethod
    def backward(ctx, g):
        U, B = ctx.saved_tensors
        add_g = (U - B) / (B.numel())
        grad = g + BiHalfEstimator.gamma * add_g
        return grad


class BinaryActivation(nn.Module):
    def __init__(self, activation_type="tanh", bihalf_gamma=6):
        super(BinaryActivation, self).__init__()
        self.activation_type = activation_type
        self.bihalf_gamma = bihalf_gamma
        BiHalfEstimator.gamma = bihalf_gamma

    def forward(self, x):
        if self.activation_type == "tanh":
            return torch.tanh(x)
        elif self.activation_type == "ste":
            return StraightThroughEstimator.apply(x)
        elif self.activation_type == "sigmoid":
            return torch.sigmoid(x) - 0.5
        elif self.activation_type == "bihalf":
            return BiHalfEstimator.apply(x)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")


class DeepHash(nn.Module):
    def __init__(self, input_dim, hash_dim, hidden_dims=None, activation_type="tanh"):
        """
        Neural network that learns to generate binary hash codes.

        Args:
            input_dim (int): Dimension of input vectors
            hash_dim (int): Number of bits in output hash
            hidden_dims (list): Dimensions of hidden layers
        """
        super(DeepHash, self).__init__()

        # Build encoder network
        if hidden_dims is None:
            hidden_dims = [512, 256]
        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim

        # Add final layer to produce hash_dim outputs
        layers.append(nn.Linear(prev_dim, hash_dim))
        self.encoder = nn.Sequential(*layers)
        self.binary_activation = BinaryActivation(activation_type)
        self.hash_dim = hash_dim

    def forward(self, x):
        """
        Forward pass using smooth approximation of sign function
        """
        h = self.encoder(x)
        return self.binary_activation(h)

    def get_binary_hash(self, x):
        """
        Get discrete binary hash codes
        """
        with torch.no_grad():
            h = self.forward(x)
            return (h >= 0).float()


class DefaultLoss(nn.Module):
    def __init__(self):
        super(DefaultLoss, self).__init__()

    def forward(self, codes, distances):
        dot_products = torch.matmul(codes, codes.t())
        hash_distances = 0.5 * (codes.size(1) - dot_products)
        distances = distances / distances.max()
        return torch.mean((hash_distances - distances) ** 2)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, codes, anchor_idx, positive_idx, negative_idx):
        anchor = codes[anchor_idx]
        positive = codes[positive_idx]
        negative = codes[negative_idx]

        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)

        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, codes, pairs, labels):
        code1, code2 = codes[pairs[:, 0]], codes[pairs[:, 1]]
        distances = torch.sum((code1 - code2) ** 2, dim=1)

        positive_loss = labels * distances
        negative_loss = (1 - labels) * torch.clamp(self.margin - distances, min=0.0)

        return (positive_loss + negative_loss).mean()


class BitIndependenceLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(BitIndependenceLoss, self).__init__()
        self.weight = weight

    def forward(self, codes):
        # Center the codes
        centered_codes = codes - codes.mean(dim=0)

        # Compute correlation matrix
        n_samples = codes.size(0)
        correlation_matrix = torch.matmul(centered_codes.t(), centered_codes) / n_samples

        # Get off-diagonal elements
        mask = torch.ones_like(correlation_matrix) - torch.eye(correlation_matrix.size(0)).to(correlation_matrix.device)
        correlations = correlation_matrix * mask

        return self.weight * torch.mean(correlations ** 2)


class HashingLoss(nn.Module):
    def __init__(self, alpha=0, beta=1.0):
        super(HashingLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, codes, distances):
        # Quantization loss: encourage outputs to be binary
        quantization_loss = 1 - torch.pow(codes, 2).mean()

        # Normalize distances to [0, 1]
        distances = distances / distances.max()

        # Compute pairwise hamming-like distances in hash space
        hash_pdist = pdist(codes)
        hash_pdist = hash_pdist / codes.size(1)  # Normalize by hash dimension

        # Calculate similarity preservation loss
        similarity_loss = torch.mean(torch.pow(distances - hash_pdist, 2))

        return self.alpha * quantization_loss + self.beta * similarity_loss


class DeepHashTrainer:
    def __init__(self, model, learning_rate=0.001, loss_type="contrastive", margin=1.0, alpha=0, beta=1.0,
                 independence_weight=0.0):
        """
        Trainer for DeepHash model.

        Args:
            model: DeepHash model instance
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_type = loss_type
        self.independence_weight = independence_weight
        self.ind_loss = BitIndependenceLoss(weight=independence_weight)

        if loss_type == "default":
            self.criterion = DefaultLoss()
        elif loss_type == "triplet":
            self.criterion = TripletLoss(margin)
        elif loss_type == "contrastive":
            self.criterion = ContrastiveLoss(margin)
        elif loss_type == "hashing":
            self.criterion = HashingLoss(alpha, beta)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def prepare_batch(self, batch_vectors):
        """
        Prepare batch for training
        """
        distances = pdist(batch_vectors)
        i, j = torch.triu_indices(batch_vectors.size(0), batch_vectors.size(0), offset=1)
        distances_upper = distances[i, j]
        if self.loss_type == "triplet":
            # each sample is an anchor, closest is positive, furthest is negative (from the upper triangle)
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

        elif self.loss_type == "contrastive":
            # labels are 1 if distance < median distance, 0 otherwise
            # median of distances
            median_distance = torch.median(distances_upper)
            pairs = torch.stack([i, j], dim=1)
            labels = (distances_upper < median_distance).float()
            return batch_vectors, (pairs, labels)

        else:
            return batch_vectors, distances

    def train_step(self, batch_vectors, optimizer):
        optimizer.zero_grad()

        batch_vectors, batch_info = self.prepare_batch(batch_vectors)
        codes = self.model(batch_vectors)

        if self.loss_type == "contrastive":
            pairs, labels = batch_info
            loss = self.criterion(codes, pairs, labels)
        elif self.loss_type == "triplet":
            anchor_idx, positive_idx, negative_idx = batch_info
            loss = self.criterion(codes, anchor_idx, positive_idx, negative_idx)
        else:
            distances = batch_info
            loss = self.criterion(codes, distances)

        # Add bit independence regularization
        if self.independence_weight > 0:
            loss = loss + self.ind_loss(codes)

        loss.backward()
        optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader, device):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc="Training", total=len(dataloader))
        for batch_vectors in pbar:
            batch_vectors = batch_vectors.to(device)
            loss = self.train_step(batch_vectors, self.optimizer)
            pbar.set_postfix(loss=loss)
            total_loss += loss

        return total_loss / len(dataloader)

    def evaluate(self, dataloader, device):
        """
        Evaluate model on test data
        """
        self.model.eval()
        original_distances = []
        hash_distances = []
        pbar = tqdm(dataloader, desc="Evaluating", total=len(dataloader))
        with torch.no_grad():
            for batch in pbar:
                vectors = batch.to(device)

                hash_codes = self.model.get_binary_hash(vectors)

                # Compute average precision for distance preservation
                batch_original_distances = pdist_upper(vectors)
                batch_hash_distances = pdist_upper(hash_codes)
                original_distances.append(batch_original_distances)
                hash_distances.append(batch_hash_distances)

            original_distances = torch.cat(original_distances)
            hash_distances = torch.cat(hash_distances)

            # Normalize distances
            original_distances = original_distances / original_distances.max()
            hash_distances = hash_distances / hash_codes.size(1)

            # Compute correlation between distances
            correlation = torch.corrcoef(
                torch.stack([original_distances, hash_distances])
            )[0, 1].item()

            return correlation, hash_codes.cpu().numpy()


def train_hash_function(vectors, hash_dim, batch_size=128, epochs=100, device='mps'):
    """
    Train a deep hash function on the given vectors.

    Args:
        vectors: Input vectors of shape (N, input_dim)
        hash_dim: Number of bits in output hash
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')

    Returns:
        model: Trained DeepHash model
        history: Training history
    """
    # Convert to torch tensors
    dataset_loader, num_dataset, dim = prepare_data(vectors, batch_size)

    # Initialize model
    model = DeepHash(dim, hash_dim).to(device)
    trainer = DeepHashTrainer(model)

    history = []
    best_correlation = 0
    best_encoding = None
    best_state = None
    for epoch in range(epochs):
        # Train for one epoch
        loss = trainer.train_epoch(dataset_loader, device)

        # Evaluate
        correlation, encoding = trainer.evaluate(dataset_loader, device)
        history.append({
            'epoch': epoch,
            'loss': loss,
            'correlation': correlation
        })
        print(f"\n\nEpoch {epoch}, Loss: {loss}, Correlation: {correlation}\n")

        # Save best model
        if correlation > best_correlation:
            best_correlation = correlation
            best_encoding = encoding
            best_state = model.state_dict()

    # Restore best model
    model.load_state_dict(best_state)
    return model, history, best_encoding


if __name__ == '__main__':
    dataset = "sift"
    base_vectors = get_base_data(dataset)
    hash_dim = 14
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model, history, encoding = train_hash_function(base_vectors, hash_dim, device=device)
    print("Training complete!")
    print("Final correlation:", history[-1]['correlation'])
    print("Final loss:", history[-1]['loss'])
