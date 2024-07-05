import numpy as np

SUPPORTED_METRICS = ["edp", "eap", "edap", "eda2p", "latency", "energy", "area"]

def calc_metric(energy_per_design: np.array, latency_per_design: np.array, area_per_design: np.array, metric: str, reduction = True):
    # function assumes that x_per_design arrays have this layout:
    #  x | l0 | l1 | l2 | l3 |
    # ------------------------
    # d0 | ...| ...| ...| ...|
    # d1 | ...| ...| ...| ...|

    assert metric in SUPPORTED_METRICS

    if metric == "edap":
        result = np.multiply(energy_per_design, latency_per_design)
        if reduction:
            result = np.sum(result, axis=1)
        result = np.multiply(result, area_per_design)

    elif metric == "eda2p":
        result = np.multiply(energy_per_design, latency_per_design)
        if reduction:
            result = np.sum(result, axis=1)
        result = np.multiply(result, np.power(area_per_design,2))

    elif metric == "edp":
        result = np.multiply(energy_per_design, latency_per_design)
        if reduction:
            result = np.sum(result, axis=1)

    elif metric == "eap":
        result = np.multiply(energy_per_design, area_per_design)
        if reduction:
            result = np.sum(result, axis=1)

    elif metric == "area":
        result = area_per_design

    elif metric == "energy":
        if reduction:
            result = np.sum(energy_per_design, axis=1)
        else:
            result = energy_per_design

    elif metric == "latency":
        if reduction:
            result = np.sum(latency_per_design, axis=1)
        else:
            result = latency_per_design

    return result

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