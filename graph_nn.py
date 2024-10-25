import networkx as nx
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import *
import random


class GraphNN(ABC):
    def __init__(self, m=5):
        """
        Initialize the base graph structure using networkx.

        Parameters:
        - m: The number of neighbors each node should be connected to.
        """
        self.graph = nx.Graph()
        self.m = m

    def build(self, data, bfs_width=None):
        """
        Build the nearest neighbor graph by adding data points one by one.

        Parameters:
        - data: list of points or feature vectors to build the graph
        - bfs_width: Number of initial random attempts for searching

        Returns:
        - list of indices of the data points in the graph
        """
        indices = []
        for point in tqdm(data):
            idx = self.add(point)
            indices.append(idx)
        return indices

    @abstractmethod
    def find(self, query_point, k=1, bfs_width=3):
        """
        Abstract method to find the k-nearest neighbors of a query point.

        Parameters:
        - query_point: The point to find neighbors for
        - k: Number of nearest neighbors to find
        - bfs_width: Number of initial random attempts for searching

        Returns:
        - list of k-nearest neighbors
        """
        pass

    def add(self, point, bfs_width=None):
        """
        Abstract method to add a new point to the existing graph.

        Parameters:
        - point: The new point to add to the graph
        """
        new_node_id = len(self.graph.nodes)
        self.graph.add_node(new_node_id, value=[point], label=[new_node_id])
        return new_node_id

    def _add_edge(self, point_a, point_b, weight=1.0):
        """
        Helper method to add an edge between two nodes with a given weight.

        Parameters:
        - point_a: The first point (node)
        - point_b: The second point (node)
        - weight: Weight of the edge between point_a and point_b (default is 1.0)
        """
        self.graph.add_edge(point_a, point_b, weight=weight)

    def _get_neighbors(self, node):
        """
        Helper method to get the neighbors of a given node.

        Parameters:
        - node: The node for which neighbors are to be fetched

        Returns:
        - list of neighbors of the given node
        """
        return set(self.graph.neighbors(node))

    def _get_order(self):
        """
        Helper method to get the order of the graph.

        Returns:
        - order of the graph
        """
        return self.graph.order()

    def display(self):
        """
        Display the graph using networkx.
        """
        nx.draw(self.graph, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title(f"Nearest Neighbor Graph (m={self.m})")
        plt.show()

    def summary(self):
        """
        Display the graph summary.
        """
        order = self._get_order()
        diameter = -1
        if order > 100_000:
            print("Graph is too large to calculate diameter")
        else:
            try:
                diameter = nx.diameter(self.graph)
            except nx.NetworkXError:
                print("Graph is not connected")
                diameter = -2
        results_dict = {
            "average_degree": sum(dict(self.graph.degree()).values()) / self._get_order(),
            "max_degree": max(dict(self.graph.degree()).values()),
            "num_vertices": order,
            "num_edges": self.graph.number_of_edges(),
            "diameter": diameter
        }
        for key, value in results_dict.items():
            print(f"{key}: {value}")

        return results_dict

    def prune(self):
        """
        Prune the graph to ensure that each node has at most m neighbors.
        """
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            node_vectors = np.array(self.graph.nodes[node]["value"])
            if len(neighbors) > self.m:
                neighbors_vectors = [np.array(self.graph.nodes[neighbor]["value"]) for neighbor in neighbors]
                distances = np.sum((node_vectors[:, np.newaxis] - neighbors_vectors)**2, axis=2).min(axis=1)
                degrees = [self.graph.degree(neighbor) for neighbor in neighbors]
                distances = list(zip(neighbors, degrees, distances))
                # degree first to keep the graph connected, then distance
                distances.sort(key=lambda x: (x[1], x[2]))
                for neighbor, _, _ in distances[self.m:]:
                    self.graph.remove_edge(node, neighbor)


# Example of extending this base class for a specific nearest neighbor algorithm
class LinearNN(GraphNN):
    def find(self, query_point, k=1, bfs_width=None):
        """
        Find the k-nearest neighbors using a simple traversal
        """
        distances = []
        num_queries = -1
        for node in self.graph.nodes(data=True):
            node_id, attributes = node
            vector = attributes["value"][0]
            distances.append((node_id, euclidean_distance(query_point, vector)))
        distances.sort(key=lambda x: x[1])
        return [node_id for node_id, _ in distances[:k]], num_queries

    def add(self, point, bfs_width=None):
        """
        Add a new point to the graph.
        """
        new_node_id = super().add(point)
        return new_node_id


class RandomNN(GraphNN):
    def find(self, query_point, k=1, bfs_width=3):
        """
        Find k-nearest neighbors by selecting bfs_width random starting points and traversing once from each.
        """
        nodes = list(self.graph.nodes())
        if not nodes:
            return []
        num_queries = 1
        visited = set()
        for _ in range(bfs_width):
            # Select a random starting point
            start_node = random.choice(nodes)
            visited.add(start_node)
            # Traverse from this starting point to its neighbors
            neighbors = self._get_neighbors(start_node)
            visited.update(neighbors)

        distances = []
        for node_id in visited:
            vector = self.graph.nodes[node_id]["value"][0]
            distances.append((node_id, euclidean_distance(query_point, vector)))
        distances.sort(key=lambda x: x[1])
        return [node_id for node_id, _ in distances[:k]], num_queries

    def add(self, point, bfs_width=3):
        """
        Add a new point to the graph and connect it to its m-nearest neighbors.
        """
        new_node_id = super().add(point)
        # Connect the new point to its k nearest neighbors
        nearest_neighbors, _ = self.find(point, k=self.m, bfs_width=bfs_width)
        for neighbor in nearest_neighbors:
            if neighbor == new_node_id:
                continue
            distance = euclidean_distance(point, self.graph.nodes[neighbor]["value"][0])
            self._add_edge(new_node_id, neighbor, weight=distance)
        return new_node_id


if __name__ == "__main__":
    # Example usage
    data_points = [
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 5]
    ]

    simple_nn = LinearNN(m=0)
    simple_nn.build(data_points)
    print("Nearest neighbors (LinearNN):", simple_nn.find([4, 4], k=2, bfs_width=1))
    simple_nn.add([4, 4])
    print("Nearest neighbors after adding a new point (LinearNN):", simple_nn.find([4, 4], k=2))
    simple_nn.display()
    # simple_nn.summary()
    random_nn = RandomNN(m=2)
    random_nn.build(data_points)
    print("Nearest neighbors (RandomNN):", random_nn.find([4, 4], k=2, bfs_width=1))
    random_nn.add([4, 4])
    print("Nearest neighbors after adding a new point (RandomNN):", random_nn.find([4, 4], k=2))
    random_nn.display()
    random_nn.summary()
    random_nn.prune()
