import heapq

from graph_nn import *


class NSWNN(GraphNN):
    def find(self, query_point, k=1, bfs_width=3, initial_nodes=None):
        """
        Find the k-nearest neighbors using multiple search attempts.

        Parameters:
        - query_point: The point to find neighbors for
        - k: Number of nearest neighbors to find
        - bfs_width: Number of random starting points for searching (traversal width)

        Returns:
        - list of k-nearest neighbors, and the number of search rounds
        """
        graph = self.graph
        order = graph.order()
        nodes = list(graph.nodes)
        if order == 0:
            return [], 0

        # Cap bfs_width and k by the graph size
        bfs_width = min(bfs_width, order)
        k = min(k, order)

        visited = set()
        best_nodes = []  # Max-heap to keep track of the best nodes
        current_front = []
        rounds = 0

        query_np = np.array(query_point)

        # Step 1: Initialize search with random nodes
        if initial_nodes is None:
            initial_nodes = random.sample(nodes, bfs_width)
        initial_vectors = [graph.nodes[node]['value'] for node in initial_nodes]
        distances = [np.sum((vector - query_np) ** 2, axis=1).min() for vector in initial_vectors]

        for node, dist in zip(initial_nodes, distances):
            visited.add(node)
            heapq.heappush(best_nodes, (-dist, node))
            current_front.append(node)

        # Step 2: Expand the search by exploring neighbors
        while current_front:
            rounds += 1
            neighbors = set()

            for node in current_front:
                neighbors.update(self._get_neighbors(node))

            new_neighbors = list(neighbors - visited)
            if not new_neighbors:
                break
            visited.update(new_neighbors)

            neighbor_vectors = [graph.nodes[n]['value'] for n in new_neighbors]
            distances = [np.sum((vector - query_np) ** 2, axis=1).min() for vector in neighbor_vectors]
            candidates = list(zip(new_neighbors, distances))
            candidates.sort(key=lambda x: x[1])
            candidates = candidates[:bfs_width]  # Limit the number of candidates to bfs_width (traversal width)
            next_front = []
            for node, dist in candidates:
                if len(best_nodes) < k:
                    heapq.heappush(best_nodes, (-dist, node))
                    next_front.append(node)
                elif dist < -best_nodes[0][0]:
                    heapq.heappushpop(best_nodes, (-dist, node))
                    next_front.append(node)
                else:
                    # Since candidates are sorted, no need to check further
                    break

            current_front = next_front

        # Extract nodes from best_nodes and sort them by distance
        best_nodes_sorted = sorted(best_nodes, key=lambda x: -x[0])
        nearest_neighbors = [node for _, node in best_nodes_sorted[:k]]
        return nearest_neighbors, rounds

    def add(self, point, bfs_width=3):
        """
        Add a new point to the graph and connect it to the k nearest neighbors found using the NSW strategy.

        Parameters:
        - point: The new point to add to the graph
        - bfs_width: Number of initial random attempts for searching
        """
        graph = self.graph
        new_node_id = super().add(point)
        order = graph.order()
        if order == 1:
            return new_node_id

        # Step 1: Find the nearest neighbors for the new point
        local_mins, _ = self.find(point, k=self.m, bfs_width=bfs_width)

        # Step 2: Create a candidate set from local minimums and their neighbors
        u = set(local_mins)
        for local_min in local_mins:
            u.update(self._get_neighbors(local_min))

        u -= {new_node_id}

        if not u:
            return new_node_id

        u = list(u)
        candidate_vectors = np.array([graph.nodes[node]['value'][0] for node in u])
        point_np = np.array(point)
        distances = np.sum((candidate_vectors - point_np)**2, axis=1)

        # Step 3: Select the top m nearest neighbors
        if len(distances) > self.m:
            nearest_indices = np.argpartition(distances, self.m)[:self.m]
        else:
            nearest_indices = np.arange(len(distances))

        # Step 4: Connect the new node to the selected neighbors
        for idx in nearest_indices:
            neighbor = u[idx]
            self._add_edge(new_node_id, neighbor)

        return new_node_id


if __name__ == "__main__":
    data_points = [
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 5]
    ]

    nsw = NSWNN(m=2)
    nsw.build(data_points)  # Build the graph using the provided data points
    print("Nearest neighbors (NSW):", nsw.find([4, 4], k=2))
    nsw.add([4, 4], bfs_width=3)
    print("Nearest neighbors after adding a new point (NSW):", nsw.find([4, 4], k=2))
    nsw.display()  # Visualize the graph using networkx and matplotlib
    nsw.summary()  # Print a summary of the graph
