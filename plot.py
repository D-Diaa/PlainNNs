import json
import os

import numpy as np
import plotly.graph_objects as go


def sym(name):
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
    # Return (color, legend_group, display_name)
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


results_dir = "results_all"
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


def create_3d_plot(data, dataset):
    # Select the metrics to plot
    for metrics in threed_dicts:
        fig = go.Figure()
        # Track min/max values for x and y to set plane dimensions
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

                # Update min/max values
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

        # Add recall=0.95 plane
        # Create a grid of points
        plane_x = np.linspace(min_x - 0.5, max_x + 0.5, 10)
        plane_y = np.linspace(min_y - 0.5, max_y + 0.5, 10)
        X, Y = np.meshgrid(plane_x, plane_y)
        Z = np.full_like(X, 0.95)  # Create plane at z=0.95

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


def create_2d_plot(data, dataset):
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

        # Create the plot
        fig = go.Figure()

        # Add traces for each algorithm
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
            xaxis_title=labels[metrics['x']],
            yaxis_title=labels[metrics['y']],
            template="plotly_white",  # Includes grid by default
            width=1200,
            height=800,
        )

        # Save the plot
        fig.write_image(f'{results_dir}/{dataset}/{metrics["x"]}_vs_{metrics["y"]}.png')
        fig.write_html(f'{results_dir}/{dataset}/{metrics["x"]}_vs_{metrics["y"]}.html')


def main():
    for dataset in ['siftsmall', 'sift']:
        # Load the data
        file_path = f'{results_dir}/{dataset}/summary.json'
        with open(file_path, 'r') as file:
            data = json.load(file)
        os.makedirs(f"{results_dir}/{dataset}", exist_ok=True)
        create_3d_plot(data, dataset)
        create_2d_plot(data, dataset)


if __name__ == "__main__":
    main()
