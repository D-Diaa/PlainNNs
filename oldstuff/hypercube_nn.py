import logging
import random

import numpy as np
from tqdm import tqdm

from mappers import create_mapper
from nsw_nn import NSWNN


class HypercubeNN(NSWNN):
    def __init__(self, m=5, init_strategy='random', permutation_strategy='pca', bucket_size=10):
        """
        Initialize the hypercube graph structure using networkx.

        Parameters:
        - m: The number of neighbors each node should be connected to (also the bitwidth of the hypercube).
        - init_strategy: The strategy to use for initializing the initial traversal nodes.
        - permutation_strategy: The strategy to use for ordering the points on the hypercube.
        - bucket_size: The maximum number of vectors in each node.
        """
        super().__init__(m)
        self.init_strategy = init_strategy
        self.permutation_strategy = permutation_strategy
        self.bit_width = m
        self.bucket_size = bucket_size
        self.mapper = None
        self.current_label = 0

    def build(self, data, bfs_width=None):
        """
        Build the hypercube graph by adding data points one by one.

        Parameters:
        - data: list of points or feature vectors to build the graph
        """
        self.mapper = create_mapper(self.permutation_strategy, data, self.bit_width)
        indices = self.mapper.fit_transform(data)
        labels = np.arange(len(data))
        self.current_label = len(data)
        for point, index, label in tqdm(zip(data, indices, labels)):
            self._add_node(index, point, label)
        self.ensure_connectedness()

    def ensure_connectedness(self):
        """
        Ensure that the hypercube graph is connected by adding edges between disconnected components.
        """
        nodes = list(self.graph.nodes)
        valid_nodes = set([node for node in nodes if len(self._get_neighbors(node)) < self.bit_width])
        pbar = tqdm(total=len(valid_nodes))
        while valid_nodes:
            node = valid_nodes.pop()
            pbar.update(1)
            current_degree = len(self._get_neighbors(node))
            if current_degree >= self.bit_width:
                continue  # Node already satisfies the degree requirement
            for bit in range(self.bit_width):
                if current_degree >= self.bit_width:
                    break  # Desired degree reached
                neighbor = node ^ (1 << bit)
                if neighbor not in nodes:
                    # missing node
                    candidate = self._bfs(neighbor, valid_nodes)
                    if candidate is not None:
                        self.graph.add_edge(node, candidate)
                        current_degree += 1
                        # Update degrees and valid_nodes
                        if len(self._get_neighbors(candidate)) >= self.bit_width:
                            valid_nodes.discard(candidate)
                            pbar.update(1)
        valid_nodes = set([node for node in nodes if len(self._get_neighbors(node)) < self.bit_width])
        while valid_nodes:
            node = valid_nodes.pop()
            pbar.update(1)
            current_degree = len(self._get_neighbors(node))
            needed = self.bit_width - current_degree
            if needed > len(valid_nodes):
                logging.warning(f"Node {node} has only {current_degree} neighbors, but {needed} more are not available.")
                needed = len(valid_nodes)
            for _ in range(needed):
                neighbor = random.choice(list(valid_nodes))
                while neighbor == node or len(self._get_neighbors(neighbor)) >= self.bit_width:
                    neighbor = random.choice(list(valid_nodes))
                self.graph.add_edge(node, neighbor)
                if len(self._get_neighbors(neighbor)) >= self.bit_width:
                    valid_nodes.discard(neighbor)
                if len(self._get_neighbors(node)) >= self.bit_width:
                    valid_nodes.discard(node)
                    break

    def _bfs(self, start, target_nodes):
        """
        Perform a breadth-first search from the start node to find the nearest target node.

        Parameters:
        - start: The starting node for the BFS
        - target_nodes: The set of target nodes to search for

        Returns:
        - The nearest target node found during the BFS
        """
        visited = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in target_nodes and len(self._get_neighbors(node)) < self.bit_width:
                return node
            visited.add(node)
            for bit in range(self.bit_width):
                neighbor = node ^ (1 << bit)
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)
        return None

    def _add_node(self, index, point, label):
        if index in self.graph.nodes:
            self.graph.nodes[index]['value'].append(point)
            self.graph.nodes[index]['label'].append(label)
            if len(self.graph.nodes[index]['value']) > self.bucket_size:
                current_size = len(self.graph.nodes[index]['value'])
                victim = random.choice(range(len(self.graph.nodes[index]['value'])))
                value = self.graph.nodes[index]['value'][victim]
                label = self.graph.nodes[index]['label'][victim]
                self.graph.nodes[index]['value'] = [self.graph.nodes[index]['value'][i] for i in range(current_size) if
                                                    i != victim]
                self.graph.nodes[index]['label'] = [self.graph.nodes[index]['label'][i] for i in range(current_size) if
                                                    i != victim]
                # TODO: handle deleted nodes: where should they go?
                # Maybe a second mapper? a near neighbor with the same mapper?
                pushed = False
                neighbors = []
                for bit in range(self.bit_width):
                    neighbor = index ^ (1 << bit)
                    if neighbor in self.graph.nodes:
                        neighbors.append(neighbor)
                        occupied = len(self.graph.nodes[neighbor]['label'])
                        available_space = self.bucket_size - occupied
                        if available_space > 0:
                            self.graph.nodes[neighbor]['value'].append(value)
                            self.graph.nodes[neighbor]['label'].append(label)
                            pushed = True
                            break
                if not pushed:
                    neighbor = random.choice(neighbors)
                    self._add_node(neighbor, value, label)

        else:
            self.graph.add_node(index, value=[point], label=[label])
            for bit in range(self.bit_width):
                neighbor = index ^ (1 << bit)
                if neighbor in self.graph.nodes:
                    self.graph.add_edge(index, neighbor)

    def add(self, point, bfs_width=None):
        index = self.mapper.transform(point)
        label = self.current_label
        self.current_label += 1
        self._add_node(index, point, label)


if __name__ == "__main__":
    # Example usage
    data_points = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [5, 5, 5]
    ])

    # Initialize the hypercube graph with 2 neighbors per node
    hypercube = HypercubeNN(m=3, permutation_strategy='default')
    hypercube.build(data_points)  # Build the graph using the provided data points
    print("Nearest neighbors (Hypercube):", hypercube.find([4, 4, 4], k=2))
    hypercube.add([4, 4, 4])
    print("Nearest neighbors after adding a new point (Hypercube):", hypercube.find([4, 4, 4], k=2))
    hypercube.display()
    hypercube.summary()
