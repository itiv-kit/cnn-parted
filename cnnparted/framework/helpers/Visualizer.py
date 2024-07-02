import os
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np

def plotMetricPerConfigPerLayer(stats: Dict, dir: str, metric: str, type: str = "line"):
    metric = metric.lower()

    assert type in ["line", "bar"], "Currently only 'line' and 'bar' are supported as plot types"
    assert metric in ["edp", "eap", "edap", "eda2p","latency", "energy", "area", "power_density"]

    metric_str, metric_unit = _get_metric_info(metric)

    # The metric_per_design array has this structure, with
    # every cell holding EAP, EDP or some other metric:
    #  x | l0 | l1 | l2 | l3 |
    # ------------------------
    # d0 | ...| ...| ...| ...|
    # d1 | ...| ...| ...| ...|
    metric_per_design = []
    labels = []

    for tag, design in stats.items():
        metric_per_layer = []
        layers = design["layers"]
        for key, layer in layers.items():
            metric_per_layer.append(_get_metric(layer, metric))

        labels.append(tag)
        metric_per_design.append(metric_per_layer)

    plt.figure(dpi=1200)
    plt.xlabel("Layer Number")
    plt.ylabel(metric_str)
    plt.title(f"{metric_str} for all Designs by layer")
    plt.gca().set_prop_cycle(marker=["o", "+", "*", "s", "x", "d"], 
                            color=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02'])
    layer_idx = np.arange(1, len(metric_per_design[0])+1)

    if type== "line":
        for i in range(len(metric_per_design)): 
            plt.plot(layer_idx, metric_per_design[i], label = labels[i], linestyle="dotted")
    elif type == "bar":
        width = 1/len(metric_per_design)
        multiplier = 0
        for i in range(len(metric_per_design)):
            offset = width * multiplier
            plt.bar(layer_idx+offset, metric_per_design[i], width=width, label=labels[i])
            multiplier += 1

    plt.legend()
    plt.savefig(os.path.join(dir, metric+"_"+type+".png"))
    

def _get_metric(layer: Dict, metric: str) -> float:
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

def _get_metric_info(metric: str):
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
