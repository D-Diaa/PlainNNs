import json
import os
import argparse

import numpy as np
import plotly.graph_objects as go


def sym(name):
    """
    Determines the symbol type based on the configuration name.

    Args:
        name (str): The name of the configuration.

    Returns:
        str: The corresponding symbol type.
    """
    if "C=20" in name:
        return "x"
    elif "C=1000" in name:
        return "square"
    elif "C=500" in name:
        return "diamond"
    elif "C=100" in name:
        return "circle-open"
    else:
        return "circle"


def get_method_info(name):
    """
    Determines the method information (color, legend group, display name) based on the algorithm name.

    Args:
        name (str): The name of the algorithm.

    Returns:
        tuple: A tuple containing the color, legend group, and display name.
    """
    if 'HNSW' in name and "ClusteredHNSW" not in name:
        return 'blue', 'Base HNSW', 'HNSW'
    elif 'ClusteredHNSW' in name and 'ClusterThenInsert' in name:
        if "after" in name:
            return 'green', 'ClusterThenInsert', 'Batched: Clustered HNSW (CTI)'
        else:
            return 'orange', 'ClusterThenInsert-All', 'Clustered HNSW (CTI)'
    elif 'ClusteredHNSW' in name and 'InsertWithMitosis' in name:
        return 'red', 'InsertWithMitosis', 'Batched: Clustered HNSW (IWM)'
    elif 'ClusteredHNSW' in name and 'InsertBlindly' in name:
        return 'purple', 'InsertBlindly', 'Batched: Clustered HNSW (IB)'
    return 'orange', 'Other', 'Other'


labels = {
    "memory_usage_bytes": "Memory Usage (Bytes)",
    "distance_computations_per_query": "Distance Computations",
    "average_query_time": "Average Query Time",
    "queries_per_second": "Queries per Second",
    "recall": "Recall"
}

threed_dicts = [
    {
        "x": "memory_usage_bytes",
        "y": "queries_per_second",
        "z": "recall"
    },
    {
        "x": "memory_usage_bytes",
        "y": "average_query_time",
        "z": "recall"
    },
    {
        "x": "memory_usage_bytes",
        "y": "distance_computations_per_query",
        "z": "recall"
    }
]

twod_dicts = [
    {
        "x": "memory_usage_bytes",
        "y": "recall"
    },
    {
        "x": "distance_computations_per_query",
        "y": "recall"
    },
    {
        "x": "average_query_time",
        "y": "recall"
    },
    {
        "x": "queries_per_second",
        "y": "recall"
    },
    {
        "x": "memory_usage_bytes",
        "y": "queries_per_second"
    }
]


def create_3d_plot(data, dataset, results_dir="results"):
    """
    Creates 3D scatter plots for the given data and saves them as HTML files.

    Args:
        data (dict): The dataset results.
        dataset (str): The dataset name.
        results_dir (str): The directory to save the results.
    """
    for metrics in threed_dicts:
        fig = go.Figure()
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for algorithm, settings in data.items():
            if "before" in algorithm:
                continue

            for conf, results in settings.items():
                results = [result for result in results if result["distance_computations_per_query"] < 100000]
                x_vals = [result[metrics["x"]] for result in results]
                y_vals = [result[metrics["y"]] for result in results]
                z_vals = [result[metrics["z"]] for result in results]

                min_x = min(min_x, min(x_vals))
                max_x = max(max_x, max(x_vals))
                min_y = min(min_y, min(y_vals))
                max_y = max(max_y, max(y_vals))
                name = f"{algorithm}-{conf}"
                color, legend_group, display_name = get_method_info(name)

                marker = sym(name)

                fig.add_trace(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='markers+lines',
                    marker=dict(size=4, color=color, symbol=marker),
                    line=dict(color=color, width=2),
                    legendgroup=legend_group,
                    legendgrouptitle_text=display_name,
                    name=conf
                ))

        plane_x = np.linspace(min_x - 0.1, max_x + 0.1, 10)
        plane_y = np.linspace(min_y - 0.1, max_y + 0.1, 10)
        X, Y = np.meshgrid(plane_x, plane_y)
        Z = np.full_like(X, 0.95)

        # Add the plane as a surface
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            showscale=False,
            opacity=0.5,
            colorscale=[[0, 'rgb(200,200,100)'], [1, 'rgb(200,200,100)']],
            name='Recall=0.95'
        ))

        fig.update_layout(
            title=f"{labels[metrics['z']]} vs {labels[metrics['x']]} vs {labels[metrics['y']]} for  {dataset}",
            scene=dict(
                xaxis_title=labels[metrics["x"]],
                yaxis_title=labels[metrics["y"]],
                zaxis_title=labels[metrics["z"]],
            )
        )

        fig.write_html(f"{results_dir}/{dataset}/{metrics['x']}_vs_{metrics['y']}_vs_{metrics['z']}.html")


def create_2d_plot(data, dataset, results_dir="results"):
    """
    Creates 2D scatter plots for the given data and saves them as PNG and HTML files.

    Args:
        data (dict): The dataset results.
        dataset (str): The dataset name.
        results_dir (str): The directory to save the results.
    """
    for metrics in twod_dicts:
        plot_data = {}
        filter_for_recall = metrics["y"] != "recall"

        # Extract the relevant data
        for algorithm, settings in data.items():
            if "before" in algorithm:
                continue
            for conf, results in settings.items():
                if filter_for_recall:
                    results = [result for result in results if result["recall"] >= 0.95]
                else:
                    results = [result for result in results if result["distance_computations_per_query"] < 100000]
                x_values = [result[metrics["x"]] for result in results]
                y_values = [result[metrics["y"]] for result in results]
                plot_data[f"{algorithm}-{conf}"] = (x_values, y_values)

        fig = go.Figure()

        for name, (x_values, y_values) in plot_data.items():
            color, legend_group, display_name = get_method_info(name)
            marker = sym(name)
            conf = name.split("-")[1]
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                legendgroup=legend_group,
                legendgrouptitle_text=display_name,
                name=conf,
                mode='lines+markers',
                line=dict(color=color),
                marker=dict(
                    symbol=marker,
                    size=10,
                    color=color,
                    opacity=0.8
                )
            ))

        # Update layout
        fig.update_layout(
            title=f"{labels[metrics['y']]} vs {labels[metrics['x']]} for {dataset}",
            xaxis_title=labels[metrics["x"]],
            yaxis_title=labels[metrics["y"]],
            template="plotly_white",
            width=1200,
            height=800,
        )

        # Save the plot
        fig.write_image(f'{results_dir}/{dataset}/{metrics["x"]}_vs_{metrics["y"]}.png')
        fig.write_html(f'{results_dir}/{dataset}/{metrics["x"]}_vs_{metrics["y"]}.html')


def main():
    parser = argparse.ArgumentParser(description="Generate 2D and 3D plots for dataset results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing dataset results. Defaults to 'results'."
    )

    args = parser.parse_args()
    results_dir = args.results_dir

    for dataset in os.listdir(results_dir):
        file_path = f'{results_dir}/{dataset}/summary.json'
        with open(file_path, 'r') as file:
            data = json.load(file)
        os.makedirs(f"{results_dir}/{dataset}", exist_ok=True)
        create_3d_plot(data, dataset, results_dir)
        create_2d_plot(data, dataset, results_dir)


if __name__ == "__main__":
    main()
