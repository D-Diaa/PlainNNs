import heapq
import logging

from math import floor, ceil, log, log2
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List
from typing import Tuple
import numpy as np
from tqdm import tqdm
from utils import batch_distances


@dataclass
class HNSWConfig:
    """Configuration for HNSW index"""
    M: int
    M0: int
    mL: float
    max_level: int
    ef_construction: int = 256

    @staticmethod
    def create(m):
        return HNSWConfig(M=m, M0=2 * m, mL=1 / np.log(m), max_level=ceil(log2(m)))

    def __str__(self):
        return f"HNSWConfig(M={self.M}, M0={self.M0}, mL={self.mL}, ef_construction={self.ef_construction}, max_level={self.max_level})"


@dataclass
class Node:
    id: int
    level: int
    neighbors: Dict[int, List[int]]

    def degree(self, level: int) -> int:
        return len(self.neighbors[level])

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id


@dataclass
class SearchResult:
    indices: List[int]
    distances: List[float]
    query_time: float


class HNSW:
    def __init__(self, dim, config: HNSWConfig):
        self.dim = dim
        self.nodes_in_memory = 0
        self.config = config
        self.current_max_level = 0
        self.entry_point = None
        self.nodes: Dict[int, Node] = {}
        self.nodes_per_level = defaultdict(list)
        self.vectors = None
        self.next_index = 0
        self.distance_computations = 0
        self.construction_time = 0

    def __str__(self):
        return f"(M={self.config.M})"

    def batch_insert(self, vectors: np.ndarray):
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack((self.vectors, vectors))
        self.nodes_in_memory += len(vectors)
        start_time = time.time()
        for vector in tqdm(vectors):
            self.insert(vector)
        self.construction_time += time.time() - start_time

    def _reset_distance_counter(self):
        self.distance_computations = 0

    def compute_distances(self, query_vec: np.ndarray, ids: List[int]) -> List[float]:
        ids = np.array(ids, dtype=np.int32)

        distances = batch_distances(query_vec, ids, self.vectors)
        self.distance_computations += len(ids)
        return distances.tolist()

    def delete(self, node_id: int):
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        self.nodes_per_level[node.level].remove(node_id)
        if self.entry_point == node_id:
            while self.current_max_level > 0 and len(self.nodes_per_level[self.current_max_level]) == 0:
                self.current_max_level -= 1
            self.entry_point = random.choice(self.nodes_per_level[self.current_max_level])
        for level, neighbors in node.neighbors.items():
            for neighbor in neighbors:
                self.nodes[neighbor].neighbors[level].remove(node_id)
        self.nodes.pop(node_id)
        self.nodes_in_memory -= 1

    def rand_level(self) -> int:
        return min(floor(-log(random.uniform(0, 1)) * self.config.mL), self.config.max_level)

    def search_layer(self,
                     query: np.ndarray,
                     entry_points: List[int],
                     ef: int,
                     level: int,
                     distances: List[float] = None) -> List[Tuple[float, id]]:
        visited = set(entry_points)
        if distances is None:
            distances = self.compute_distances(query, entry_points)
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
                break

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
        level = self.rand_level()
        node = Node(self.next_index, level, defaultdict(list))
        self.nodes_per_level[level].append(node.id)
        self.nodes[node.id] = node
        self.next_index += 1

        if self.entry_point is None:
            self.entry_point = node.id
            self.current_max_level = level
            return node.id

        curr = [self.entry_point]
        distances = self.compute_distances(vector, curr)

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
            self.current_max_level = level
            self.entry_point = node.id

        return node.id

    def select_neighbors(
            self,
            candidates: List[Tuple[float, int]],
            M: int,
            keep_pruned: bool = False
    ) -> List[int]:
        kept = set()
        kept_list = []
        discarded_neighbors = []

        working_queue = deque(sorted(candidates, key=lambda x: x[0]))
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
        if ef < k:
            logging.warning(f"ef={ef} is less than k={k}, setting ef to k")
            ef = k
        start_time = time.time()
        self._reset_distance_counter()

        curr = [self.entry_point]
        distances = self.compute_distances(query, curr)
        for level in range(self.current_max_level, 0, -1):
            nearest = self.search_layer(query, curr, 1, level, distances)
            curr = [n for _, n in nearest]
            distances = [d for d, _ in nearest]
        nearest = self.search_layer(query, curr, ef, 0, distances)

        query_time = time.time() - start_time

        # Convert to SearchResult format
        top_k = nearest[:k]
        return SearchResult(
            indices=[n for _, n in top_k],
            distances=[d for d, _ in top_k],
            query_time=query_time
        )

    def get_construction_time(self) -> float:
        return self.construction_time

    def compute_stats(self) -> dict:
        """Compute various statistics about the index structure"""
        level_dist = defaultdict(int)
        total_edges = 0
        for node in self.nodes.values():
            level_dist[node.level] += 1
            for level_edges in node.neighbors.values():
                total_edges += len(level_edges)

        avg_out_degree = total_edges / len(self.nodes) if self.nodes else 0
        memory_costs = [
            self.dim * 4,  # vector (float32)
            4,  # id (int32)
            4,  # level (int32)
        ]
        cost_per_node = sum(memory_costs)
        memory_usage = cost_per_node * self.nodes_in_memory + total_edges * 4
        return {
            "num_nodes": len(self.nodes),
            "current_max_level": self.current_max_level,
            "level_distribution": dict(level_dist),
            "average_out_degree": avg_out_degree,
            "memory_usage_bytes": memory_usage,
        }


if __name__ == "__main__":
    np.random.seed(42)
    dim = 2
    vectors = np.random.randn(5000, dim).astype(np.float32)
    k = 10
    m = 32
    config = HNSWConfig(M=m, M0=2 * m, mL=1 / np.log(m), ef_construction=4 * m, max_level=ceil(log2(m)))
    index = HNSW(dim=dim, config=config)
    index.batch_insert(vectors)
    for vector in vectors:
        result = index.search(vector, k, 4 * m)
        print(result)

    index.batch_insert(vectors)
    for vector in vectors:
        result = index.search(vector, k, 4 * m)
        print(result)
