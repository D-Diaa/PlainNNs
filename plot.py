import argparse
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plot_utils import *


# Function to generate a color legend

def generate_color_legend(color_map: Dict[str, str], output_path: str):
    fig = go.Figure()

    for i, (label, color) in enumerate(color_map.items()):
        fig.add_trace(go.Scatter(
            x=[0], y=[i],
            mode='markers',
            marker=dict(size=15, color=color),
            name=label,
            hoverinfo='none',
        ))

    fig.update_layout(
        showlegend=True,
        legend=dict(title="", orientation="h", yanchor="middle", xanchor="center", x=0.5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
        width=900,
        height=90,
    )

    fig.write_image(output_path, format="pdf")
    print(f"Color legend saved to {output_path}")


# Function to generate a symbol legend

def generate_symbol_legend(symbol_map: Dict[str, str], output_path: str):
    fig = go.Figure()

    for i, (label, symbol) in enumerate(symbol_map.items()):
        fig.add_trace(go.Scatter(
            x=[0], y=[i],
            mode='markers',
            marker=dict(size=15, symbol=symbol, color='black'),
            name=label,
            hoverinfo='none',
        ))

    fig.update_layout(
        showlegend=True,
        legend=dict(title="", orientation="h", yanchor="middle", xanchor="center", x=0.5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
        width=900,
        height=90,
    )

    fig.write_image(output_path, format="pdf")
    print(f"Symbol legend saved to {output_path}")


# Function to create 3D plots using the dataset metrics
def create_3d_plot(data: Dict, dataset: str, results_dir: str, color_map: Dict[str, str], ignored_keylists=None):
    for metrics in THREED_METRICS:
        fig = go.Figure()
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        # Loop through each algorithm and configuration to extract metrics
        for algorithm, settings in data.items():
            if "before" in algorithm:
                continue

            for conf, results in settings.items():
                name = f"{algorithm}-{conf}"
                if any(all(key.lower() in name.lower() for key in key_list) for key_list in ignored_keylists):
                    continue

                x_vals = [result[metrics["x"]] for result in results]  # Extract x-axis values
                y_vals = [result[metrics["y"]] for result in results]  # Extract y-axis values
                z_vals = [result[metrics["z"]] for result in results]  # Extract z-axis values

                # Update axis ranges
                min_x = min(min_x, min(x_vals))
                max_x = max(max_x, max(x_vals))
                min_y = min(min_y, min(y_vals))
                max_y = max(max_y, max(y_vals))

                color, legend_name = get_method_info(name, color_map)  # Get color and legend info
                conf = format_config(conf)  # Format configuration for readability
                marker = SYMBOL_MAP.get(conf, 'circle')  # Determine marker style

                # Add a 3D scatter plot for the current configuration
                fig.add_trace(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='markers+lines',
                    marker=dict(size=4, color=color, symbol=marker),
                    line=dict(color=color, width=2),
                    legendgroup=legend_name,
                    legendgrouptitle_text=legend_name,
                    name=conf
                ))

        # Add a surface plane for reference recall value
        plane_x = np.linspace(min_x * 0.99, max_x * 1.01, 10)
        plane_y = np.linspace(min_y * 0.99, max_y * 1.01, 10)
        X, Y = np.meshgrid(plane_x, plane_y)
        Z = np.full_like(X, 0.95)

        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            showscale=False,
            opacity=0.5,
            colorscale=[[0, 'rgb(200,200,100)'], [1, 'rgb(200,200,100)']],
            name='Recall=0.95'
        ))

        # Update the layout with axis labels and titles
        fig.update_layout(
            scene=dict(
                xaxis_title=LABELS[metrics["x"]],
                yaxis_title=LABELS[metrics["y"]],
                zaxis_title=LABELS[metrics["z"]],
            )
        )

        # Save the plot as an HTML file
        fig.write_html(f"{results_dir}/{dataset}/{metrics['x']}_vs_{metrics['y']}_vs_{metrics['z']}.html")


# Function to create 2D plots using the dataset metrics
def create_2d_plot(data: Dict, dataset: str, results_dir: str, color_map: Dict[str, str], ignored_keylists=None):
    for metrics in TWOD_METRICS:
        plot_data = {}
        filter_for_recall = metrics["y"] != "recall"  # Check if we need to filter based on recall

        # Loop through each algorithm and configuration to extract metrics
        for algorithm, settings in data.items():
            if "before" in algorithm:
                continue
            for conf, results in settings.items():
                name = f"{algorithm}-{conf}"
                if any(all(key.lower() in name.lower() for key in key_list) for key_list in ignored_keylists):
                    continue
                if filter_for_recall:
                    results = [result for result in results if result["recall"] >= 0.95]
                x_values = [result[metrics["x"]] for result in results]  # Extract x-axis values
                y_values = [result[metrics["y"]] for result in results]  # Extract y-axis values
                plot_data[name] = (x_values, y_values)

        fig = go.Figure()

        # Add scatter plots for each configuration
        for name in sorted(plot_data.keys()):
            (x_values, y_values) = plot_data[name]
            color, legend_name = get_method_info(name, color_map)  # Get color and legend info
            conf = name.split("-")[1]
            conf = format_config(conf)  # Format configuration for readability
            marker = SYMBOL_MAP.get(conf, 'circle')  # Determine marker style
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                legendgroup=legend_name,
                legendgrouptitle_text=legend_name,
                name=conf,
                mode='lines+markers',
                line=dict(color=color),
                marker=dict(
                    symbol=marker,
                    size=16,
                    color=color,
                    opacity=0.8
                )
            ))

        # Update the layout with axis labels and titles
        fig.update_layout(
            xaxis=dict(
                title=dict(text=LABELS[metrics["x"]], font=dict(size=35)),  # Font size for axis title
                tickfont=dict(size=26),  # Font size for ticks
                type=AXES_TYPES[metrics["x"]],
            ),
            yaxis=dict(
                title=dict(text=LABELS[metrics["y"]], font=dict(size=35)),  # Font size for axis title
                tickfont=dict(size=26),  # Font size for ticks
                type=AXES_TYPES[metrics["y"]],
            ),
            template="plotly_white",
            width=1200,
            height=800,
        )

        # Save the plot as an HTML file
        html_path = f'{results_dir}/{dataset}/{metrics["x"]}_vs_{metrics["y"]}.html'
        fig.write_html(html_path)

        # Remove legend and save the plot as a PDF
        fig_no_legend = fig.to_dict()
        for trace in fig_no_legend['data']:
            if 'legendgroup' in trace:
                del trace['legendgroup']

        fig_no_legend = go.Figure(fig_no_legend)
        fig_no_legend.update_layout(showlegend=False)

        pdf_path = f'{results_dir}/{dataset}/{metrics["x"]}_vs_{metrics["y"]}_no_legend.pdf'
        fig_no_legend.write_image(pdf_path, format='pdf')


# Function to generate summary plots for memory usage and construction time
def generate_memory_and_construction_plots(json_path: str, color_map: Dict[str, str], ignored_keylists=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    rows = []
    # Extract relevant data from the JSON file
    for method, configs in data.items():
        for config, metrics_list in configs.items():
            name = f"{method}-{config}"
            if any(all(key.lower() in name.lower() for key in key_list) for key_list in ignored_keylists):
                continue
            for metrics in metrics_list:
                rows.append({
                    "method_config": name,
                    "construction_time": metrics.get("construction_time"),
                    "memory_usage_bytes": metrics.get("memory_usage_bytes")
                })

    df = pd.DataFrame(rows)

    # Filter out irrelevant configurations and prepare summary data
    filtered_df = df[~df['method_config'].str.contains("before")]
    filtered_df['name'], filtered_df['config'] = zip(*filtered_df['method_config'].apply(rename_method_config))
    filtered_df['config'] = filtered_df['config'].apply(format_config)
    # sort by config
    categories = sorted(filtered_df['config'].unique(), key=lambda x: int(x.split('=')[1]) if 'C=' in x else 0)
    filtered_df['config'] = pd.Categorical(filtered_df['config'], categories=categories, ordered=True)

    summary_table = (
        filtered_df.groupby(['name', 'config'])
        .agg(
            max_construction_time=('construction_time', 'max'),
            max_memory=('memory_usage_bytes', 'max')
        )
        .reset_index()
    )

    save_dir = os.path.dirname(json_path)

    # Create a bar plot for construction time
    fig_time = go.Figure()
    for name in summary_table['name'].unique():
        subset = summary_table[summary_table['name'] == name]
        fig_time.add_trace(go.Bar(
            x=subset['config'],
            y=subset['max_construction_time'],
            name=name,
            marker=dict(color=color_map[name])
        ))
    fig_time.update_layout(
        xaxis=dict(
            title=dict(text="Average Cluster Size", font=dict(size=18)),  # Font size for axis title
            tickfont=dict(size=16),  # Font size for ticks
        ),
        yaxis=dict(
            title=dict(text="Construction Time (s)", font=dict(size=18)),  # Font size for axis title
            tickfont=dict(size=16),  # Font size for ticks
        ),
        barmode='group'
    )
    time_save_path = os.path.join(save_dir, "max_construction_time.html")
    fig_time.write_html(file=time_save_path)

    # Save the construction time plot without legend as a PDF
    fig_time_no_legend = fig_time.to_dict()
    fig_time_no_legend['layout']['showlegend'] = False
    fig_time_no_legend = go.Figure(fig_time_no_legend)
    time_pdf_path = os.path.join(save_dir, "max_construction_time_no_legend.pdf")
    fig_time_no_legend.write_image(time_pdf_path, format='pdf')

    print(f"Saved construction time plot to: {time_save_path}")
    print(f"Saved construction time plot without legend to: {time_pdf_path}")

    # Create a bar plot for memory usage
    fig_memory = go.Figure()
    for name in summary_table['name'].unique():
        subset = summary_table[summary_table['name'] == name]
        fig_memory.add_trace(go.Bar(
            x=subset['config'],
            y=subset['max_memory'],
            name=name,
            marker=dict(color=color_map[name])
        ))
    fig_memory.update_layout(
        xaxis=dict(
            title=dict(text="Average Cluster Size", font=dict(size=18)),  # Font size for axis title
            tickfont=dict(size=16),  # Font size for ticks
        ),
        yaxis=dict(
            title=dict(text="Max Memory Usage (bytes)", font=dict(size=18)),  # Font size for axis title
            tickfont=dict(size=16),  # Font size for ticks
            type="log",
        ),
        barmode='group'
    )
    memory_save_path = os.path.join(save_dir, "max_memory_usage.html")
    fig_memory.write_html(file=memory_save_path)

    # Save the memory usage plot without legend as a PDF
    fig_memory_no_legend = fig_memory.to_dict()
    fig_memory_no_legend['layout']['showlegend'] = False
    fig_memory_no_legend = go.Figure(fig_memory_no_legend)
    memory_pdf_path = os.path.join(save_dir, "max_memory_usage_no_legend.pdf")
    fig_memory_no_legend.write_image(memory_pdf_path, format='pdf')

    print(f"Saved memory usage plot to: {memory_save_path}")
    print(f"Saved memory usage plot without legend to: {memory_pdf_path}")


# Main function to parse arguments and generate plots
def main():
    parser = argparse.ArgumentParser(description="Generate 2D and 3D plots for dataset results.")
    parser.add_argument(
        "--results_dirs",
        nargs='+',
        help="Paths to the directories containing the results",
        default=["results", "results_all", "results"]
    )
    args = parser.parse_args()
    color_map = {}

    for results_dir in args.results_dirs:
        ignored_keylists = [] if results_dir == "results" else [
            ['c=20'],
            ['c=500'],
            ['c=1000'],
        ]
        # Iterate over each dataset in the results directory
        for dataset in os.listdir(results_dir):
            file_path = os.path.join(results_dir, dataset, "summary.json")
            if not os.path.isfile(file_path):
                continue

            with open(file_path, 'r') as file:
                data = json.load(file)

            os.makedirs(os.path.join(results_dir, dataset), exist_ok=True)
            create_3d_plot(data, dataset, results_dir, color_map, ignored_keylists)  # Generate 3D plots
            create_2d_plot(data, dataset, results_dir, color_map, ignored_keylists)  # Generate 2D plots
            generate_memory_and_construction_plots(file_path, color_map, ignored_keylists)  # Generate summary plots

            # Generate color and symbol legends
            generate_color_legend(color_map, f"{results_dir}/{dataset}/color_legend.pdf")
            generate_symbol_legend(SYMBOL_MAP, f"{results_dir}/{dataset}/symbol_legend.pdf")
    print(color_map)


if __name__ == "__main__":
    main()
