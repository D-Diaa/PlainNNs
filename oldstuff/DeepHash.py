from typing import List, Dict

import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from activation_functions import BinaryActivation
from loss_functions import build_loss, BitIndependenceLoss
from utils import *


class HashingEvaluator:
    def __init__(self, original_vectors: np.ndarray, hash_codes: np.ndarray):
        """
        Initialize evaluator with original vectors and their hash codes

        Args:
            original_vectors: Original feature vectors (N, dim)
            hash_codes: Binary hash codes (N, bits)
        """
        self.original = original_vectors
        self.hashed = hash_codes
        self.num_samples = len(original_vectors)

        # Precompute nearest neighbors in original space
        self.nn_original = NearestNeighbors(n_neighbors=100)
        self.nn_original.fit(original_vectors)

        # Precompute nearest neighbors in hash space
        self.nn_hash = NearestNeighbors(n_neighbors=100, metric='hamming')
        self.nn_hash.fit(hash_codes)

    def evaluate_all(self, num_queries: int = 1000) -> Dict[str, float]:
        """
        Compute all evaluation metrics

        Args:
            num_queries: Number of random queries to use for evaluation

        Returns:
            Dictionary containing all metrics
        """
        # Randomly sample query points
        query_indices = np.random.choice(self.num_samples, num_queries, replace=False)

        metrics = {}

        # Retrieval quality metrics
        maps = self.mean_average_precision(query_indices)
        metrics['mAP@10'] = maps[0]
        metrics['mAP@100'] = maps[1]

        recalls = self.recall_at_k(query_indices)
        for k, recall in zip([1, 10, 100], recalls):
            metrics[f'R@{k}'] = recall

        # Hash quality metrics
        metrics['bit_balance'] = self.bit_balance()
        metrics['bit_entropy'] = self.bit_entropy()
        metrics['bit_variance'] = self.bit_variance()
        metrics['independence_score'] = self.bit_independence()

        # Distance preservation metrics
        metrics['kendall_tau'] = self.kendall_tau(query_indices)
        metrics['distance_ratio_score'] = self.distance_ratio_score(query_indices)

        return metrics

    def mean_average_precision(self, query_indices: np.ndarray, k_values: List[int] = [10, 100]) -> List[float]:
        """
        Compute Mean Average Precision at different k values
        """
        maps = []

        # Get ground truth neighbors
        original_distances, original_indices = self.nn_original.kneighbors(
            self.original[query_indices], n_neighbors=max(k_values)
        )

        # Get hash-based neighbors
        hash_distances, hash_indices = self.nn_hash.kneighbors(
            self.hashed[query_indices], n_neighbors=max(k_values)
        )

        for k in k_values:
            aps = []
            for i in range(len(query_indices)):
                ground_truth = set(original_indices[i, :k])
                retrieved = hash_indices[i, :k]

                # Compute precision at each point
                precisions = []
                num_relevant = 0
                for j, retrieved_idx in enumerate(retrieved, 1):
                    if retrieved_idx in ground_truth:
                        num_relevant += 1
                        precisions.append(num_relevant / j)

                ap = sum(precisions) / k if precisions else 0
                aps.append(ap)

            maps.append(np.mean(aps))

        return maps

    def recall_at_k(self, query_indices: np.ndarray, k_values: List[int] = [1, 10, 100]) -> List[float]:
        """
        Compute Recall@k for different k values
        """
        recalls = []

        # Get ground truth neighbors
        original_distances, original_indices = self.nn_original.kneighbors(
            self.original[query_indices], n_neighbors=max(k_values)
        )

        # Get hash-based neighbors
        hash_distances, hash_indices = self.nn_hash.kneighbors(
            self.hashed[query_indices], n_neighbors=max(k_values)
        )

        for k in k_values:
            recall_scores = []
            for i in range(len(query_indices)):
                ground_truth = set(original_indices[i, :k])
                retrieved = set(hash_indices[i, :k])
                recall = len(ground_truth.intersection(retrieved)) / k
                recall_scores.append(recall)

            recalls.append(np.mean(recall_scores))

        return recalls

    def bit_balance(self) -> float:
        """
        Compute bit balance - how close each bit is to 50/50 split
        Perfect balance = 1.0, worst = 0.0
        """
        bit_means = np.mean(self.hashed, axis=0)
        balance_scores = 1 - 2 * np.abs(bit_means - 0.5)
        return float(np.mean(balance_scores))

    def bit_entropy(self) -> float:
        """
        Compute entropy of each bit
        Higher is better (more information content)
        """
        bit_means = np.mean(self.hashed, axis=0)
        entropies = []
        for p in bit_means:
            if 0 < p < 1:
                entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
                entropies.append(entropy)
        return float(np.mean(entropies))

    def bit_variance(self) -> float:
        """
        Compute variance of each bit
        Higher variance indicates better utilization of the bit
        """
        return float(np.mean(np.var(self.hashed, axis=0)))

    def bit_independence(self) -> float:
        """
        Compute independence between bits using correlation
        Lower absolute correlation indicates better independence
        """
        correlations = np.corrcoef(self.hashed.T)
        # Get upper triangular part without diagonal
        upper_triangle = correlations[np.triu_indices_from(correlations, k=1)]
        return 1.0 - float(np.mean(np.abs(upper_triangle)))

    def kendall_tau(self, query_indices: np.ndarray, sample_size: int = 1000) -> float:
        """
        Compute Kendall's Tau rank correlation between original and hash distances
        Measures how well relative distances are preserved
        """
        from scipy.stats import kendalltau

        tau_scores = []
        for idx in query_indices:
            # Sample points to compare distances with
            sample_indices = np.random.choice(
                [i for i in range(self.num_samples) if i != idx],
                min(sample_size, self.num_samples - 1),
                replace=False
            )

            # Compute distances in original space
            original_dists = np.linalg.norm(
                self.original[sample_indices] - self.original[idx:idx + 1],
                axis=1
            )

            # Compute Hamming distances in hash space
            hash_dists = np.sum(
                self.hashed[sample_indices] != self.hashed[idx:idx + 1],
                axis=1
            )

            # Compute rank correlation
            tau, _ = kendalltau(original_dists, hash_dists)
            tau_scores.append(tau)

        return float(np.mean(tau_scores))

    def distance_ratio_score(self, query_indices: np.ndarray, sample_size: int = 1000) -> float:
        """
        Compute how well the hash preserves distance ratios
        Score of 1.0 means perfect preservation
        """
        scores = []
        for idx in query_indices:
            # Sample pairs of points
            sample_indices = np.random.choice(
                [i for i in range(self.num_samples) if i != idx],
                min(sample_size, self.num_samples - 1),
                replace=False
            )

            # Compute distances in original space
            original_dists = np.linalg.norm(
                self.original[sample_indices] - self.original[idx:idx + 1],
                axis=1
            )

            # Normalize distances
            original_dists = original_dists / np.max(original_dists)

            # Compute Hamming distances in hash space
            hash_dists = np.sum(
                self.hashed[sample_indices] != self.hashed[idx:idx + 1],
                axis=1
            )
            hash_dists = hash_dists / self.hashed.shape[1]  # Normalize by hash length

            # Compare distance ratios
            ratio_diff = np.abs(original_dists - hash_dists)
            scores.append(1.0 - np.mean(ratio_diff))

        return float(np.mean(scores))


def evaluate_hash_quality(original_vectors: np.ndarray,
                          hash_codes: np.ndarray,
                          num_queries: int = 1000) -> Dict[str, float]:
    """
    Convenience function to compute all hash quality metrics

    Args:
        original_vectors: Original feature vectors (N, dim)
        hash_codes: Binary hash codes (N, bits)
        num_queries: Number of random queries to use for evaluation

    Returns:
        Dictionary containing all metrics
    """
    evaluator = HashingEvaluator(original_vectors, hash_codes)
    return evaluator.evaluate_all(num_queries)

class DeepHash(nn.Module):
    def __init__(self, input_dim, hash_dim, config):
        """
        Neural network that learns to generate binary hash codes.

        Args:
            input_dim (int): Dimension of input vectors
            hash_dim (int): Number of bits in output hash
            config (dict): Configuration dictionary
        """
        super(DeepHash, self).__init__()

        # Build encoder network
        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for dim in config["hidden_dims"]:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim

        self.net = nn.Sequential(*layers)
        self.encoding_layer = nn.Linear(prev_dim, hash_dim)
        self.binary_activation = BinaryActivation(config["activation_type"])
        self.hash_dim = hash_dim

    def forward(self, inputs, return_preencoding=False):
        """
        Forward pass using smooth approximation of sign function
        """
        x = self.net(inputs)
        binary_codes = self.binary_activation(self.encoding_layer(x))
        if return_preencoding:
            return binary_codes, x
        return binary_codes

    def get_binary_hash(self, x):
        """
        Get discrete binary hash codes
        """
        with torch.no_grad():
            h = self.forward(x)
            return (h >= 0).float()


class DeepHashTrainer:
    def __init__(self, model, config):
        """
        Trainer for DeepHash model.

        Args:
            model: DeepHash model instance
            config: Dictionary containing training configuration
        """
        self.model = model
        self.optimizer = config["optimizer"]["type"](model.parameters(), **config["optimizer"]["optim_params"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.loss_type = config["loss_type"]
        self.ind_loss = BitIndependenceLoss(weight=config["independence_weight"])
        self.criterion = build_loss(config)
        self.config = config

    def train_step(self, batch_vectors, optimizer):
        optimizer.zero_grad()
        batch_vectors, batch_info = self.criterion.prepare_batch(batch_vectors)
        call_kwargs = {
            "return_preencoding": self.loss_type == "bihalf",
            "inputs": batch_vectors
        }
        output = self.model(**call_kwargs)
        if self.loss_type == "bihalf":
            codes, batch_info = output
        else:
            codes = output
        loss = self.criterion(codes, batch_info)
        # Add bit independence regularization
        loss = loss + self.ind_loss(codes, None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        self.model.eval()
        with torch.no_grad():
            # Get all vectors and their hash codes
            all_vectors = []
            all_hashes = []
            for batch in dataloader:
                vectors = batch.to(device)
                hash_codes = self.model.get_binary_hash(vectors)
                all_vectors.append(vectors.cpu().numpy())
                all_hashes.append(hash_codes.cpu().numpy())

            all_vectors = np.concatenate(all_vectors)
            all_hashes = np.concatenate(all_hashes)

            # Compute all metrics
            metrics = evaluate_hash_quality(all_vectors, all_hashes)

            return metrics, all_hashes


def train_hash_function(vectors, hash_dim, config, device='mps'):
    """
    Train a deep hash function on the given vectors.
    """
    # Prepare data
    dataset_loader, num_dataset, dim = prepare_data(vectors, config["batch_size"])

    if "hidden_dims" not in config:
        config["hidden_dims"] = [512, 256]

    # Initialize model
    model = DeepHash(dim, hash_dim, config).to(device)
    trainer = DeepHashTrainer(model, config)

    # Training history
    history = []
    best_map = 0
    best_state = None
    best_encoding = None
    best_metrics = None

    eval_frequency = 10  # Evaluate every 10 epochs

    try:
        for epoch in range(config["epochs"]):
            # Train for one epoch
            loss = trainer.train_epoch(dataset_loader, device)

            # Evaluate periodically
            if epoch % eval_frequency == 0:
                metrics, encoding = trainer.evaluate(dataset_loader, device)

                # Log all metrics
                print(f"\nEpoch {epoch}")
                print(f"Loss: {loss:.4f}")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")

                # Update learning rate based on mAP
                trainer.scheduler.step(metrics['mAP@100'])

                # Save history
                history.append({
                    'epoch': epoch,
                    'loss': loss,
                    **metrics
                })

                # Save best model based on mAP@100
                if metrics['mAP@100'] > best_map:
                    print(f"New best model! mAP@100: {metrics['mAP@100']:.4f}")
                    best_map = metrics['mAP@100']
                    best_encoding = encoding
                    best_state = model.state_dict()
                    best_metrics = metrics

            # Early stopping check
            if trainer.scheduler.num_bad_epochs > 10:  # No improvement for 10 evaluations
                print(f"Early stopping at epoch {epoch}")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Print final performance
    print("\nTraining complete!")
    print("\nBest model performance:")
    for metric_name, value in best_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Restore best model
    model.load_state_dict(best_state)
    return model, history, best_encoding

def get_config():
    return {
        "independence_weight": 1e-3,
        "batch_size": 256,
        "epochs": 200,
        "margin": 0.5,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "loss_type": "combined",
        "activation_type": "tanh",
        "hidden_dims": [512, 256],
        "optimizer": {
            "type": optim.SGD,
            "optim_params": {
                "lr": 1e-3
            }
        }
    }


if __name__ == '__main__':
    dataset = "siftsmall"
    hash_dim = 14
    config = get_config()
    base_vectors = get_base_data(dataset)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model, history, encoding = train_hash_function(base_vectors, hash_dim, config, device=device)
    print("Training complete!")
    print("Final correlation:", history[-1]['correlation'])
    print("Final loss:", history[-1]['loss'])
