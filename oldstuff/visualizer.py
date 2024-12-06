import math
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hnsw.hnsw import HNSW


class HNSWVisualizer:
    def __init__(self, hnsw: HNSW):
        self.hnsw = hnsw
        self.base_positions = self._compute_base_positions()
        self.max_degree = self._compute_max_degree()

    def _compute_base_positions(self) -> Dict[int, np.ndarray]:
        """Compute consistent base positions for all nodes using ground layer"""
        graph = nx.Graph()

        # Add all nodes
        for node_id in self.hnsw.nodes:
            graph.add_node(node_id)

        # Add ground layer (level 0) connections for layout computation
        for node_id, node in self.hnsw.nodes.items():
            for neighbor in node.neighbors[0]:  # Use level 0 connections
                graph.add_edge(node_id, neighbor.id)

        # Compute layout using spring_layout
        pos = nx.spring_layout(graph, k=2 / np.sqrt(graph.number_of_nodes()), iterations=100)

        return {node_id: np.array(pos[node_id]) for node_id in pos}

    def _compute_max_degree(self) -> int:
        """Compute the maximum degree across all levels for normalization"""
        max_deg = 0
        for node in self.hnsw.nodes.values():
            for neighbors in node.neighbors.values():
                max_deg = max(max_deg, len(neighbors))
        return max_deg if max_deg > 0 else 1

    def _create_graph_for_level(self, level: int) -> nx.Graph:
        """Create a networkx graph for a specific level"""
        graph = nx.Graph()

        # Add nodes and edges for this level
        for node_id, node in self.hnsw.nodes.items():
            if node.level >= level:
                graph.add_node(node_id, level=node.level, degree=len(node.neighbors[level]))
                for neighbor in node.neighbors[level]:
                    graph.add_edge(node_id, neighbor.id)

        return graph

    def _get_node_colors(self, G: nx.Graph) -> list[np.ndarray]:
        """Get colors for nodes based on their level using a distinct color palette"""
        colors = []
        max_level = self.hnsw.current_max_level
        cmap = plt.get_cmap('viridis')

        for node in G.nodes():
            node_level = self.hnsw.nodes[node].level
            # Normalize level for color mapping
            normalized_level = node_level / max_level if max_level > 0 else 0
            colors.append(cmap(normalized_level))

        return colors

    def _get_node_sizes(self, G: nx.Graph, scale: float = 20.0, level=0) -> List[float]:
        """Get sizes for nodes based on their degree"""
        sizes = []
        for node in G.nodes():
            degree = self.hnsw.nodes[node].degree(level)
            # Scale sizes between 10 and 50
            size = 10 + (degree / self.max_degree) * scale
            sizes.append(size)
        return sizes

    def visualize_level_interactive(
            self,
            level: int,
            show_labels: bool = False,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize a specific level of the HNSW graph interactively using Plotly.

        Args:
            level (int): The level to visualize.
            show_labels (bool): Whether to display node labels.
            save_path (Optional[str]): If provided, save the plot to the given path.
        """
        graph = self._create_graph_for_level(level)
        pos = {node_id: self.base_positions[node_id] for node_id in graph.nodes()}

        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = []
        node_y = []
        for node_id in graph.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)

        node_sizes = self._get_node_sizes(graph, level=level)

        hover_text = []
        for node in graph.nodes():
            text = f'Node ID: {node}<br>Level: {self.hnsw.nodes[node].level}<br>Degree: {len(self.hnsw.nodes[node].neighbors[level])}'
            hover_text.append(text)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=hover_text if show_labels else None,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=[self.hnsw.nodes[node].level for node in graph.nodes()],
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    title='Node Level',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'HNSW Graph - Level {level}',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="",
                                showarrow=False,
                                xref="paper", yref="paper"
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.update_xaxes(range=[-1, 1])
        fig.update_yaxes(range=[-1, 1])

        if save_path:
            fig.write_html(save_path)

        fig.show()

    def visualize_all_levels_interactive(self, save_path: Optional[str] = None) -> None:
        """
        Visualize all levels of the HNSW graph interactively using Plotly in a single figure with subplots.

        Args:
            save_path (Optional[str]): If provided, save the plot to the given path.
        """
        num_levels = self.hnsw.current_max_level + 1
        cols = 2
        rows = math.ceil(num_levels / cols)

        fig = make_subplots(rows=rows, cols=cols,
                            subplot_titles=[f'Level {lvl}' for lvl in range(num_levels)],
                            horizontal_spacing=0.05, vertical_spacing=0.05)

        for lvl in range(num_levels):
            graph = self._create_graph_for_level(lvl)
            pos = {node_id: self.base_positions[node_id] for node_id in graph.nodes()}

            edge_x = []
            edge_y = []
            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )

            node_x = []
            node_y = []
            for node_id in graph.nodes():
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)

            node_sizes = self._get_node_sizes(graph, level=lvl)

            hover_text = []
            for node in graph.nodes():
                text = f'Node ID: {node}<br>Level: {self.hnsw.nodes[node].level}<br>Degree: {len(self.hnsw.nodes[node].neighbors[lvl])}'
                hover_text.append(text)

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                hoverinfo='text',
                text=hover_text,
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    color=[self.hnsw.nodes[node].level for node in graph.nodes()],
                    size=node_sizes,
                    colorbar=dict(
                        thickness=15,
                        title='Node Level',
                        xanchor='left',
                        titleside='right'
                    ),
                    line_width=2
                )
            )

            row = lvl // cols + 1
            col = lvl % cols + 1
            fig.add_trace(edge_trace, row=row, col=col)
            fig.add_trace(node_trace, row=row, col=col)
            fig.update_xaxes(range=[-1, 1], row=row, col=col)
            fig.update_yaxes(range=[-1, 1], row=row, col=col)

        fig.update_layout(
            height=500 * rows,
            width=750 * cols,
            title_text="HNSW Graph - All Levels",
            showlegend=False,
            hovermode='closest',
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()

    def plot_level_statistics_interactive(self, save_path: Optional[str] = None) -> None:
        """
        Plot statistics about level distribution and connectivity interactively using Plotly.

        Args:
            save_path (Optional[str]): If provided, save the plot to the given path.
        """
        level_dist = defaultdict(int)
        degrees = defaultdict(list)

        for node in self.hnsw.nodes.values():
            level_dist[node.level] += 1
            for lvl, neighbors in node.neighbors.items():
                degrees[lvl].append(len(neighbors))

        levels = sorted(level_dist.keys())
        counts = [level_dist[lvl] for lvl in levels]

        # Bar chart for node distribution across levels
        bar_trace = go.Bar(
            x=levels,
            y=counts,
            name='Node Count',
            marker=dict(color='indianred')
        )

        # Line chart for average degree by level
        avg_degrees = [np.mean(degrees[lvl]) if degrees[lvl] else 0 for lvl in levels]
        std_degrees = [np.std(degrees[lvl]) if degrees[lvl] else 0 for lvl in levels]

        line_trace = go.Scatter(
            x=levels,
            y=avg_degrees,
            mode='lines+markers',
            name='Average Degree',
            line=dict(color='royalblue')
        )

        # Error bars for standard deviation
        error_trace = go.Scatter(
            x=levels,
            y=avg_degrees,
            mode='markers',
            name='Std Dev',
            error_y=dict(
                type='data',
                array=std_degrees,
                visible=True
            ),
            marker=dict(color='royalblue'),
            showlegend=False
        )

        fig = go.Figure(data=[bar_trace, line_trace, error_trace],
                        layout=go.Layout(
                            title='HNSW Graph - Level Statistics',
                            xaxis=dict(title='Level'),
                            yaxis=dict(title='Count / Average Degree'),
                            barmode='group'
                        ))

        if save_path:
            fig.write_html(save_path)

        fig.show()


# Example usage in the demo function:
def visualization_demo():
    import time
    from index import HNSW

    # Create a small dataset for visualization
    np.random.seed(42)
    dim = 2  # Use 2D for easier visualization
    num_vectors = 200
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)

    # Initialize index with smaller parameters for visualization
    index = HNSW(
        dim=dim,
        M=5,
        M0=10,
        mL=1 / np.log(5),
        ef_construction=20,
        max_level=4
    )

    # Build index
    print("Building index for visualization...")
    start_time = time.time()
    for vector in vectors:
        index.insert(vector)
    build_time = time.time() - start_time
    print(f"Index built in {build_time:.2f} seconds.")

    # Create visualizer
    visualizer = HNSWVisualizer(index)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Single level interactive visualization
    # visualizer.visualize_level_interactive(level=0, show_labels=False)

    # 2. All levels interactive visualization
    visualizer.visualize_all_levels_interactive()

    # 3. Statistics interactive visualization
    visualizer.plot_level_statistics_interactive()


if __name__ == "__main__":
    visualization_demo()
