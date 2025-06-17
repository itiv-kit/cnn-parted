import os
from turtle import title
from matplotlib import legend
from matplotlib.pylab import xlabel, ylabel, yscale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from framework.helpers.design_metrics import calc_metric, get_metric_info

COLOR_SEQUENCE = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
MARKER_SEQUENCE = ["x",          "+",     "1",       "2",       "3",         "4",      "."]

def plotMetricPerConfigPerLayer(stats: dict, dir: str, metric: str, type: str = "line", scale: str = "linear", prefix: str = ""):
    assert type in ["line", "bar"], "Currently only 'line' and 'bar' are supported as plot types"

    metric = metric.lower()
    metric_str, metric_unit = get_metric_info(metric)

    # The metric_per_design array has this structure, with
    # every cell holding EAP, EDP or some other metric:
    #  x | l0 | l1 | l2 | l3 |
    # ------------------------
    # d0 | ...| ...| ...| ...|
    # d1 | ...| ...| ...| ...|
    energy_per_design = []
    latency_per_design = []
    area_per_design = []
    labels = []

    for tag, design in stats.items():
        assert len(design["networks"]) == 1
        network = list(design["networks"].keys())[0]
        layers = design["networks"][network]
        energy_per_layer = []
        latency_per_layer = []
        for key, layer in layers.items():
            energy_per_layer.append(layer["energy"])    
            latency_per_layer.append(layer["latency"])    

        labels.append(tag)
        energy_per_design.append(energy_per_layer)
        latency_per_design.append(latency_per_layer)
        area_per_design.append([layer["area"]])

    metric_per_design = calc_metric(np.array(energy_per_design), np.array(latency_per_design), np.array(area_per_design), metric, reduction=False)

    fig = plt.figure(dpi=500)
    plt.xlabel("Layer Number")
    #plt.ylabel(f"{metric_str} [{metric_unit}]")
    plt.ylabel(f"{metric_str}")
    plt.yscale(scale)
    plt.title(f"{metric_str} for all Designs by layer")
    plt.gca().set_prop_cycle(marker=["o", "+", "*", "s", "x", "d", "o"], 
                            color=COLOR_SEQUENCE)
    layer_idx = np.arange(1, len(layers)+1)

    if type== "line":
        for i in range(len(metric_per_design)): 
            plt.plot(layer_idx, metric_per_design[i], label = labels[i], linestyle="dotted")
    elif type == "bar":
        data = np.array(metric_per_design).T
        df = pd.DataFrame(data, columns=labels)
        df.plot.bar(color=COLOR_SEQUENCE, 
                    title=f"{metric_str} for all Designs by layer", 
                    ylabel=f"{metric_str} [{metric_unit}]", 
                    xlabel="Layer Number",
                    logy=(scale=="log"),
                    )

    plt.legend()
    
    fname = prefix + metric + "_" + type + ".png"
    fig.tight_layout()
    fig.savefig(os.path.join(dir, fname))
    plt.close()
    
