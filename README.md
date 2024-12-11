# HNSW Vector Search with Clustering

This project implements an enhanced version of the Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search, incorporating clustering to improve memory and dynamicity. The implementation includes both a standard HNSW index and a clustered variant that organizes vectors into clusters for more efficient searching.

## Overview

The HNSW algorithm creates a hierarchical graph structure for efficient similarity search in high-dimensional spaces. This implementation extends the basic HNSW approach by:

1. Adding clustering capabilities to group similar vectors together
2. Supporting different insertion methods for managing clusters
3. Providing comprehensive evaluation metrics and visualization tools
4. Enabling batch operations for large-scale data processing

### Key Features

- Standard HNSW implementation with configurable parameters
- Clustered HNSW variant with multiple insertion strategies:
  - ClusterThenInsert: Creates clusters before inserting into HNSW
  - InsertBlindly: Direct insertion without clustering
  - InsertWithMitosis: Dynamic cluster splitting based on size
- Comprehensive evaluation framework
- Support for standard benchmark datasets (SIFT, GIST)
- Visualization tools for performance analysis
- Parallel processing support for evaluations

## Installation

### Prerequisites

The project requires Python 3.7+ and the following dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:
- numba: For optimized distance computations
- scikit-learn: For clustering algorithms
- tqdm: For progress tracking
- plotly: For visualization
- kaleido: For plot export support

## Project Structure

- `classes.py`: Core data structures and configuration classes
- `hnsw.py`: Base HNSW implementation
- `clustered_hnsw.py`: Clustered variant of HNSW
- `evaluator.py`: Evaluation framework
- `utils.py`: Utility functions for data handling
- `plot.py`: Visualization tools
- `main.py`: Main execution script

## Usage

### Basic Usage

Run evaluations on default datasets:

```bash
python main.py
```

### Advanced Usage

Specify custom parameters:

```bash
python main.py --datasets siftsmall sift --k 10 --ef_values 10 24 32 48 64 128
```

Parameters:
- `--datasets`: List of datasets to evaluate (choices: siftsmall, sift, gist)
- `--k`: Number of nearest neighbors to consider
- `--ef_values`: List of ef search values to evaluate

### Visualization

Generate performance plots:

```bash
python plot.py --results_dir results
```

## Configuration

### HNSW Configuration

The `HNSWConfig` class allows customization of HNSW parameters:

```python
config = HNSWConfig(
    M=16,          # Maximum number of connections per layer
    M0=32,         # Maximum number of connections for layer 0
    mL=1/np.log(16), # Level multiplier
    ef_construction=256  # Size of dynamic candidate list during construction
)
```

### Cluster Configuration

The `ClusterConfig` class controls clustering behavior:

```python
cluster_config = ClusterConfig(
    average_cluster_size=128,     # Target average cluster size
    maximum_cluster_size=256,     # Maximum allowed cluster size
    insert_method="ClusterThenInsert",  # Insertion strategy
    algorithm_name="MiniBatchKMeans"    # Clustering algorithm
)
```

## Evaluation Metrics

The evaluator measures:

- Recall and precision
- Query time performance
- Memory usage
- Distance computations
- Construction time
- Level distribution statistics

## Implementation Details

### Distance Computation

The project uses Numba-optimized distance calculations for performance:

```python
@njit(fastmath=True)
def batch_distances(query_vec, indices, all_vectors):
    # Optimized Euclidean distance computation
```

### Clustering Strategies

1. **ClusterThenInsert**:
   - Creates clusters first
   - Inserts cluster centroids into HNSW
   - Good for static datasets

2. **InsertBlindly**:
   - Direct insertion without initial clustering
   - Suitable for dynamic updates

3. **InsertWithMitosis**:
   - Monitors cluster sizes
   - Splits oversized clusters
   - Balances between performance and maintenance

## Visualization Output

The plotting tools generate:

- 2D performance comparisons
- 3D visualizations of trade-offs
- Interactive HTML plots
- Static PNG exports

Results are saved in the `results/<dataset>/` directory.


## References

The HNSW implementation is based on:
- "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Yu. A. Malkov and D. A. Yashunin