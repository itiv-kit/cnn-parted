import os
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np

from framework.helpers.DesignMetrics import calc_metric, get_metric_info

def plotMetricPerConfigPerLayer(stats: Dict, dir: str, metric: str, type: str = "line"):
    assert type in ["line", "bar"], "Currently only 'line' and 'bar' are supported as plot types"

    metric = metric.lower()
    metric_str, metric_unit = get_metric_info(metric)

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
            metric_per_layer.append(calc_metric(layer, metric))

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
        width = (0.5)/len(metric_per_design)
        multiplier = 0
        for i in range(len(metric_per_design)):
            offset = width * multiplier
            plt.bar(layer_idx+offset, metric_per_design[i], width=width, label=labels[i])
            multiplier += 1

    plt.legend()
    plt.savefig(os.path.join(dir, metric+"_"+type+".png"))
    
