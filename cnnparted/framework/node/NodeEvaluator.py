from abc import ABC
from unittest import result

## Information about a single layer
class LayerResult:
    def __init__(self):
        self.name: str
        self.area: float
        self.latency: float #must be in ms
        self.energy: float #must be in mJ
        self.cycles: int #must be in mm2

    def to_dict(self):
        return {self.name: 
                {"latency": self.latency,
                 "area": self.area,
                 "energy": self.energy}}

## Information about a design
#   layers: layerwise information about energy, area, ...
#   architecture: architecture parameters, such as number of PEs, memory sizes, ...
class DesignResult:
    def __init__(self):
        self.layers: list[LayerResult]
        self.architecture: dict = None

    def add_layer(self, results: LayerResult):
        self.layers.append(results)

    def to_dict(self):
        res = {}
        res["arch_config"] = self.architecture
        res["layers"] = {} 
        for layer in self.layers:
            res["layers"].setdefault(layer.to_dict().keys(), layer.to_dict().values())

## Information about a compute node/platform of the overall system
class NodeResult:
    def __init__(self):
        self.designs: list[DesignResult] = []

    def add_design(self, result: DesignResult):
        self.designs.append(result)

    def to_dict(self):
        stats = {}
        for i, design in enumerate(self.designs):
            stats[f"design_{i}"] = design.to_dict()
    

class NodeEvaluator(ABC):
    def eval_network(self, config: dict):
        ...