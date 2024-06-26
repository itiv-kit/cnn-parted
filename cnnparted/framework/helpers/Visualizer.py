import os
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np

def plotMetricPerConfigPerLayer(stats: Dict, dir: str, metric: str):
    metric = metric.lower()

    designs: List[Dict] = [v for v in stats.values()]
    metric_per_design = []
    for design in designs:
        metric_per_layer = []
        for layer in design.values():
            if metric == "edap":
                metric_per_layer.append(layer["energy"]*layer["latency"]*layer["area"])
                metric_str = metric.upper()
            elif metric == "edp":
                metric_per_layer.append(layer["energy"]*layer["latency"])
                metric_str = metric.upper()
            elif metric == "eap":
                metric_per_layer.append(layer["energy"]*layer["area"])
                metric_str = metric.upper()
            elif metric == "area":
                metric_per_layer.append(layer["area"])
                metric_str = metric.capitalize()
            elif metric == "energy":
                metric_per_layer.append(layer["energy"])
                metric_str = metric.capitalize()
            elif metric == "latency":
                metric_per_layer.append(layer["latency"])
                metric_str = metric.capitalize()
            elif metric == "power_density":
                metric_per_layer.append(layer["energy"]*layer["latency"] / layer["area"])
                metric_str = "Power Density"

        metric_per_design.append(metric_per_layer)

    np.save("metric_per_design.npy", np.array(metric_per_design))
    plt.figure(dpi=1200)
    plt.xlabel("Layer Number")
    plt.ylabel(metric_str)
    plt.title(f"{metric_str} for all Designs by layer")
    layer_idx = [x+1 for x in range(len(metric_per_design[0]))] #x indices with 1 based indexing
    for i in range(len(metric_per_design)): 
        plt.plot(layer_idx, metric_per_design[i], label = f"design{i}")
    plt.legend()
    plt.savefig(os.path.join(dir, metric+".png"))
    
