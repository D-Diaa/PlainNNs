import heapq
import logging
import random
import time
from collections import defaultdict, deque
from typing import List, Dict, Tuple
import numpy as np
from math import ceil, log2, floor, log
from tqdm import tqdm
from utils import batch_distances
from classes import HNSWConfig, Node, SearchResult


class HNSW:
    """
    Hierarchical Navigable Small World (HNSW) is an approximate nearest neighbor search index.

    This implementation organizes nodes in hierarchical layers where each node represents a data vector.
    It is designed for efficient similarity search in high-dimensional spaces.

    Attributes:
        dim (int): Dimensionality of the vectors being indexed.
        nodes_in_memory (int): Count of the nodes currently stored in the index.
        config (HNSWConfig): Configuration parameters that control the behavior of the HNSW index.
        current_max_level (int): Highest level in the hierarchical graph.
        entry_point (int): ID of the node used as the starting point for searches.
        nodes (Dict[int, Node]): Dictionary mapping node IDs to Node objects.
        nodes_per_level (defaultdict): Groups nodes by their respective levels.
        vectors (np.ndarray): Array storing the vectors being indexed.
        next_index (int): Counter for assigning unique IDs to new nodes.
        distance_computations (int): Tracks the number of distance calculations performed.
        construction_time (float): Total time spent building the index.
    """
    def __init__(self, dim, config: HNSWConfig):
        """
        Initializes the HNSW index with the specified dimensionality and configuration.

        Args:
            dim (int): Dimensionality of the vectors being indexed.
            config (HNSWConfig): Configuration parameters for the HNSW index.
        """
        self.dim = dim  # Dimensionality of the vectors
        self.nodes_in_memory = 0  # Total number of nodes currently in memory
        self.config = config  # Configuration parameters for the HNSW index
        self.current_max_level = 0  # Tracks the highest level in the graph
        self.entry_point = None  # Entry point for search operations
        self.nodes: Dict[int, Node] = {}  # Maps node IDs to Node objects
        self.nodes_per_level = defaultdict(list)  # Nodes grouped by levels
        self.vectors = None  # Stores the vectors being indexed
        self.next_index = 0  # Counter for assigning unique node IDs
        self.distance_computations = 0  # Tracks the number of distance computations
        self.construction_time = 0  # Tracks the total time spent constructing the graph

    def __str__(self):
        """Returns a string representation of the HNSW configuration."""
        return f"(M={self.config.M})"

    def batch_insert(self, vectors: np.ndarray):
        """
        Inserts a batch of vectors into the HNSW index.

        Args:
            vectors (np.ndarray): Array of vectors to be inserted.
        """
        if self.vectors is None:
            self.vectors = vectors  # Initialize the vector storage
        else:
            self.vectors = np.vstack((self.vectors, vectors))  # Append new vectors
        self.nodes_in_memory += len(vectors)  # Update the count of nodes in memory
        start_time = time.time()
        for vector in tqdm(vectors):
            self.insert(vector)  # Insert each vector into the graph
        self.construction_time += time.time() - start_time  # Update construction time

    def _reset_distance_counter(self):
        """Resets the distance computation counter to zero."""
        self.distance_computations = 0

    def compute_distances(self, query_vec: np.ndarray, ids: List[int]) -> List[float]:
        """
        Computes distances between a query vector and a list of node IDs.

        Args:
            query_vec (np.ndarray): Query vector.
            ids (List[int]): List of node IDs to compute distances to.

        Returns:
            List[float]: List of computed distances.
        """
        ids = np.array(ids, dtype=np.int32)  # Ensure IDs are in correct format

        distances = batch_distances(query_vec, ids, self.vectors)  # Compute distances
        self.distance_computations += len(ids)  # Update the computation counter
        return distances.tolist()

    def delete(self, node_id: int):
        """
        Deletes a node from the HNSW index.

        Args:
            node_id (int): ID of the node to delete.
        """
        if node_id not in self.nodes:
            return  # Do nothing if the node does not exist
        node = self.nodes[node_id]
        self.nodes_per_level[node.level].remove(node_id)  # Remove from level list
        if self.entry_point == node_id:
            # Adjust the entry point if the deleted node was the entry point
            while self.current_max_level > 0 and len(self.nodes_per_level[self.current_max_level]) == 0:
                self.current_max_level -= 1
            self.entry_point = random.choice(self.nodes_per_level[self.current_max_level])
        for level, neighbors in node.neighbors.items():
            for neighbor in neighbors:
                self.nodes[neighbor].neighbors[level].remove(node_id)  # Remove connections
        self.nodes.pop(node_id)  # Remove node from the dictionary
        self.nodes_in_memory -= 1  # Decrement the count of nodes in memory

    def rand_level(self) -> int:
        """
        Generates a random level for a new node based on the configuration.

        Returns:
            int: Generated level for the node.
        """
        return min(floor(-log(random.uniform(0, 1)) * self.config.mL), self.config.max_level)

    def search_layer(self,
                     query: np.ndarray,
                     entry_points: List[int],
                     ef: int,
                     level: int,
                     distances: List[float] = None) -> List[Tuple[float, id]]:
        """
        Searches within a specific layer of the graph.

        Args:
            query (np.ndarray): Query vector.
            entry_points (List[int]): List of entry point node IDs.
            ef (int): Size of the dynamic list for candidate selection.
            level (int): Layer level to search in.
            distances (List[float], optional): Precomputed distances to entry points. Defaults to None.

        Returns:
            List[Tuple[float, id]]: Sorted list of nodes with distances.
        """
        visited = set(entry_points)  # Track visited nodes
        if distances is None:
            distances = self.compute_distances(query, entry_points)  # Compute initial distances
        else:
            assert isinstance(distances, list), "distances must be a list"
            assert len(distances) == len(entry_points), "distances length must match entry_points length"

        # Initialize the candidates heap (min-heap based on distance)
        candidates = list(zip(distances, entry_points))
        heapq.heapify(candidates)

        # Initialize the dynamic_list heap (max-heap based on distance using negative distances)
        dynamic_list = [(-dist, node) for dist, node in candidates]
        heapq.heapify(dynamic_list)
        while candidates:
            current_dist, current_node = heapq.heappop(candidates)
            if current_dist > -dynamic_list[0][0]:
                break  # Stop if the current distance is greater than the largest in dynamic_list

            current_neighbors = self.nodes[current_node].neighbors[level]
            unvisited_neighbors = [n for n in current_neighbors if n not in visited]

            # Skip if no new neighbors
            if not unvisited_neighbors:
                continue

            # Update visited set
            visited.update(unvisited_neighbors)

            neighbor_distances = self.compute_distances(query, unvisited_neighbors)
            for neighbor, dist in zip(unvisited_neighbors, neighbor_distances):
                if len(dynamic_list) < ef or dist < -dynamic_list[0][0]:
                    heapq.heappush(candidates, (dist, neighbor))
                    heapq.heappush(dynamic_list, (-dist, neighbor))

                    if len(dynamic_list) > ef:
                        heapq.heappop(dynamic_list)

        return sorted([(-dist, node) for dist, node in dynamic_list], key=lambda x: x[0])

    def insert(self, vector: np.ndarray) -> int:
        """
        Inserts a single vector into the HNSW index.

        Args:
            vector (np.ndarray): Vector to insert.

        Returns:
            int: ID assigned to the inserted vector.
        """
        level = self.rand_level()  # Determine the level for the new node
        node = Node(self.next_index, level, defaultdict(list))  # Create the new node
        self.nodes_per_level[level].append(node.id)  # Add the node to the level list
        self.nodes[node.id] = node  # Store the node in the dictionary
        self.next_index += 1  # Increment the index for future nodes

        if self.entry_point is None:
            self.entry_point = node.id  # Set the first node as the entry point
            self.current_max_level = level  # Update the maximum level
            return node.id

        curr = [self.entry_point]  # Start traversal from the entry point
        distances = self.compute_distances(vector, curr)  # Compute initial distances

        for lc in range(self.current_max_level, level, -1):
            nearest = self.search_layer(vector, curr, ef=1, level=lc, distances=distances)
            if nearest:
                distances, curr = zip(*nearest)
                curr = list(curr)
                distances = list(distances)

        for lc in range(min(level, self.current_max_level), -1, -1):
            nearest = self.search_layer(vector, curr, ef=self.config.ef_construction, level=lc, distances=distances)
            if nearest:
                distances, curr = zip(*nearest)
                curr = list(curr)
                distances = list(distances)

            m = self.config.M0 if lc == 0 else self.config.M
            neighbors = self.select_neighbors(nearest, m, keep_pruned=True)

            for neighbor in neighbors:
                node.neighbors[lc].append(neighbor)
                self.nodes[neighbor].neighbors[lc].append(node.id)
            for neighbor in neighbors:
                neighbor_neighbors = self.nodes[neighbor].neighbors[lc]
                if len(neighbor_neighbors) > m:
                    # Get all current neighbors with distances
                    neighbor_vec = self.vectors[neighbor]
                    neighbor_distances = self.compute_distances(neighbor_vec, neighbor_neighbors)
                    neighbor_candidates = [(dist, n) for dist, n in zip(neighbor_distances, neighbor_neighbors)]

                    # Select best neighbors
                    selected_neighbors = self.select_neighbors(neighbor_candidates, m, keep_pruned=False)

                    # Remove this node from neighbors that weren't selected
                    removed_neighbors = set(neighbor_neighbors) - set(selected_neighbors)
                    for removed_neighbor in removed_neighbors:
                        if neighbor in self.nodes[removed_neighbor].neighbors[lc]:
                            self.nodes[removed_neighbor].neighbors[lc].remove(neighbor)

                    # Update neighbor's connections
                    self.nodes[neighbor].neighbors[lc] = selected_neighbors

        if level > self.current_max_level:
            self.current_max_level = level  # Update the maximum level
            self.entry_point = node.id  # Update the entry point

        return node.id

    def select_neighbors(
            self,
            candidates: List[Tuple[float, int]],
            M: int,
            keep_pruned: bool = False
    ) -> List[int]:
        """
        Selects the best neighbors for a node based on distance and configuration.

        Args:
            candidates (List[Tuple[float, int]]): List of candidates with distances.
            M (int): Maximum number of neighbors to select.
            keep_pruned (bool, optional): Whether to retain pruned neighbors. Defaults to False.

        Returns:
            List[int]: List of selected neighbor IDs.
        """
        kept = set()  # Set of selected neighbors
        kept_list = []  # List of selected neighbors
        discarded_neighbors = []  # List of discarded neighbors

        working_queue = deque(sorted(candidates, key=lambda x: x[0]))  # Sort candidates by distance
        while len(kept) < M and working_queue:
            dist, node = working_queue.popleft()
            if node in kept:
                continue
            if kept:
                distances = self.compute_distances(self.vectors[node], kept_list)
                min_distance = min(distances)
                if min_distance < dist:
                    heapq.heappush(discarded_neighbors, (dist, node))
                    continue
            kept.add(node)
            kept_list.append(node)

        if keep_pruned:
            while len(kept) < M and discarded_neighbors:
                _, node = heapq.heappop(discarded_neighbors)
                kept.add(node)
                kept_list.append(node)

        return kept_list

    def search(self, query: np.ndarray, k: int, ef: int) -> SearchResult:
        """
        Searches the HNSW index for the k-nearest neighbors to the query vector.

        Args:
            query (np.ndarray): Query vector.
            k (int): Number of nearest neighbors to return.
            ef (int): Size of the dynamic list for candidate selection.

        Returns:
            SearchResult: Object containing the search results.
        """
        if ef < k:
            logging.warning(f"ef={ef} is less than k={k}, setting ef to k")
            ef = k  # Ensure ef is at least as large as k
        start_time = time.time()
        self._reset_distance_counter()

        curr = [self.entry_point]  # Start from the entry point
        distances = self.compute_distances(query, curr)  # Compute initial distances
        for level in range(self.current_max_level, 0, -1):
            nearest = self.search_layer(query, curr, 1, level, distances)
            curr = [n for _, n in nearest]  # Update current nodes
            distances = [d for d, _ in nearest]  # Update distances
        nearest = self.search_layer(query, curr, ef, 0, distances)  # Perform final search in level 0

        query_time = time.time() - start_time  # Measure query time

        # Convert to SearchResult format
        top_k = nearest[:k]
        return SearchResult(
            indices=[n for _, n in top_k],
            distances=[d for d, _ in top_k],
            query_time=query_time
        )

    def get_construction_time(self) -> float:
        """
        Returns the total time spent constructing the index.

        Returns:
            float: Construction time in seconds.
        """
        return self.construction_time

    def compute_stats(self) -> dict:
        """
        Computes various statistics about the index structure.

        Returns:
            dict: Dictionary containing statistics like number of nodes, level distribution, and memory usage.
        """
        level_dist = defaultdict(int)  # Distribution of nodes across levels
        total_edges = 0  # Total number of edges in the graph
        for node in self.nodes.values():
            level_dist[node.level] += 1  # Increment count for the node's level
            for level_edges in node.neighbors.values():
                total_edges += len(level_edges)  # Count edges

        avg_out_degree = total_edges / len(self.nodes) if self.nodes else 0
        memory_costs = [
            self.dim * 4,  # vector (float32)
            4,  # id (int32)
            4,  # level (int32)
        ]
        cost_per_node = sum(memory_costs)  # Memory cost per node
        memory_usage = cost_per_node * self.nodes_in_memory + total_edges * 4  # Total memory usage
        return {
            "num_nodes": len(self.nodes),
            "current_max_level": self.current_max_level,
            "level_distribution": dict(level_dist),
            "average_out_degree": avg_out_degree,
            "memory_usage_bytes": memory_usage,
        }


if __name__ == "__main__":
    """
    Main script for testing the HNSW implementation.

    This script generates random vectors, inserts them into an HNSW index, and performs
    nearest neighbor searches. The results are printed to the console.
    """
    np.random.seed(42)
    dim = 2
    vectors = np.random.randn(5000, dim).astype(np.float32)  # Generate random 2D vectors
    k = 10  # Number of nearest neighbors to search for
    m = 32  # Configuration parameter for graph connectivity
    config = HNSWConfig(M=m, M0=2 * m, mL=1 / np.log(m), ef_construction=4 * m, max_level=ceil(log2(m)))
    index = HNSW(dim=dim, config=config)  # Initialize HNSW index
    index.batch_insert(vectors)  # Insert vectors into the index
    for vector in vectors:
        result = index.search(vector, k, 4 * m)  # Search for k nearest neighbors
        print(result)

    index.batch_insert(vectors)  # Insert vectors again
    for vector in vectors:
        result = index.search(vector, k, 4 * m)
        print(result)
