import matplotlib.pyplot as plt
from typing import Dict, List

def plotEdpPerConfigPerLayer(stats: Dict):
    designs: List[Dict] = [v for v in stats.values()]
    edp_per_design = []
    for design in designs:
        edp_per_layer = []
        for stats in design.values():
            edp_per_layer.append(stats["energy"]*stats["latency"])
        edp_per_design.append(edp_per_layer)

    plt.xlabel("Layer Number")
    plt.ylabel("EDP")
    plt.title("EDP for all Designs by layer")
    x = [x+1 for x in range(len(edp_per_design[0]))] #x indices with 1 based indexing
    for i in range(len(edp_per_design[0])): 
        plt.plot(x, edp_per_design[i], label = f"design{i}")
    plt.legend()
    plt.show()
    plt.savefig("visualizer2.png")
    
