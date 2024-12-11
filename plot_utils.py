import itertools
from typing import Dict, Tuple

# Constants
SYMBOL_MAP = {
    "C=20": "x",
    "C=50": "circle",
    "C=100": "circle-open",
    "C=500": "diamond",
    "C=1000": "square",
}

COLOR_CYCLE = itertools.cycle([
    'blue', 'green', 'red', 'purple', 'orange', 'yellow', 'pink', 'brown', 'black'
])

LABELS = {
    "memory_usage_bytes": "Memory Usage (Bytes)",
    "distance_computations_per_query": "Distance Computations per Query",
    "queries_per_second": "Queries per Second",
    "average_query_time": "Average Query Time",
    "median_query_time": "Median Query Time (s)",
    "recall": "Recall",
    "construction_time": "Construction Time (s)",
}

AXES_TYPES = {
    "memory_usage_bytes": "log",
    "distance_computations_per_query": "log",
    "queries_per_second": "linear",
    "average_query_time": "linear",
    "median_query_time": "log",
    "recall": "linear",
    "construction_time": "linear",
}

THREED_METRICS = [
    {"x": "memory_usage_bytes", "y": "queries_per_second", "z": "recall"},
    {"x": "memory_usage_bytes", "y": "distance_computations_per_query", "z": "recall"},
    {"x": "construction_time", "y": "queries_per_second", "z": "recall"},
]

TWOD_METRICS = [
    {"x": "memory_usage_bytes", "y": "recall"},
    {"x": "distance_computations_per_query", "y": "recall"},
    {"x": "queries_per_second", "y": "recall"},
    {"x": "memory_usage_bytes", "y": "queries_per_second"},
    {"x": "median_query_time", "y": "recall"},
]


# Helper Functions
def format_config(config: str) -> str:
    if 'C=' in config:
        c_value = config.split('C=')[1].split(',')[0]
        return f"C={c_value}"
    return ""


def rename_method_config(name: str) -> Tuple[str, str]:
    if 'HNSW' in name and "ClusteredHNSW" not in name:
        return 'Full: HNSW', name.split('-')[1]
    elif 'ClusteredHNSW' in name and 'ClusterThenInsert' in name:
        if "after" in name:
            return 'Batched: Clustered HNSW (CTI)', name.split('-')[1]
        else:
            return 'Full: Clustered HNSW (CTI)', name.split('-')[1]
    elif 'ClusteredHNSW' in name and 'InsertWithMitosis' in name:
        return 'Batched: Clustered HNSW (IWM)', name.split('-')[1]
    elif 'ClusteredHNSW' in name and 'InsertBlindly' in name:
        return 'Batched: Clustered HNSW (IB)', name.split('-')[1]
    return None, None


def get_method_info(name: str, color_map: Dict[str, str]) -> Tuple[str, str]:
    renamed_name, config = rename_method_config(name)
    if renamed_name is not None:
        name = renamed_name
    if name not in color_map:
        color_map[name] = next(COLOR_CYCLE)
    return color_map[name], name