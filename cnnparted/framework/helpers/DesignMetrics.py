from typing import Dict

SUPPORTED_METRICS = ["edp", "eap", "edap", "eda2p", "latency", "energy", "area", "power_density"]

def calc_metric(layer: Dict, metric: str) -> float:
    assert metric in SUPPORTED_METRICS

    if metric == "edap":
        metric = layer["energy"]*layer["latency"]*layer["area"]
    if metric == "eda2p":
        metric = layer["energy"]*layer["latency"]*layer["area"]**2
    elif metric == "edp":
        metric = layer["energy"]*layer["latency"]
    elif metric == "eap":
        metric = layer["energy"]*layer["area"]
    elif metric == "area":
        metric = layer["area"]
    elif metric == "energy":
        metric = layer["energy"]
    elif metric == "latency":
        metric = layer["latency"]
    elif metric == "power_density":
        metric = layer["energy"]*layer["latency"] / layer["area"]

    return metric

# Helper Function to get name and unit of the metric for plotting
def get_metric_info(metric: str):
    assert metric in SUPPORTED_METRICS

    if metric == "edap":
        metric_str = metric.upper()
        unit = r"$mJ \cdot ms \cdot mm^2$"
    if metric == "eda2p":
        metric_str = "EDA$^2$P"
        unit = r"$mJ \cdot ms \cdot mm^4$"
    elif metric == "edp":
        metric_str = metric.upper()
        unit = r"$mJ \cdot ms$"
    elif metric == "eap":
        metric_str = metric.upper()
        unit = r"$mJ \cdot mm^2$"
    elif metric == "area":
        metric_str = metric.capitalize()
        unit = "$mm^2$"
    elif metric == "energy":
        metric_str = metric.capitalize()
        unit = "$mJ$"
    elif metric == "latency":
        metric_str = metric.capitalize()
        unit = "$ms$"
    elif metric == "power_density":
        metric_str = "Power Density"
        unit = r'$(\frac{mJ \cdot ms}{mm^2})$'

    return metric_str, unit