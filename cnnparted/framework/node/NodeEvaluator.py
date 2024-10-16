from abc import ABC

## Information about a single layer
class LayerResult:
    def __init__(self):
        self.name: str
        self.area: float
        self.latency: float #must be in ms
        self.energy: float #must be in mJ
        self.cycles: int #must be in mm2

    def to_dict(self) -> dict :
        return {self.name: 
                {"latency": self.latency,
                 "area": self.area,
                 "energy": self.energy}}

## Information about a design
#   layers: layerwise information about energy, area, ...
#   architecture: architecture parameters, such as number of PEs, memory sizes, ...
class DesignResult:
    def __init__(self, architecture: dict = {}):
        self.layers: list[LayerResult] = []
        self.architecture: dict = architecture

    def add_layer(self, results: LayerResult):
        self.layers.append(results)

    def to_dict(self) -> dict:
        res = {}
        res["arch_config"] = self.architecture
        res["layers"] = {} 
        for layer in self.layers:
            ld: dict = layer.to_dict()
            name = list(ld.keys())[0]
            data = list(ld.values())[0]
            res["layers"][name] = data
        return res

## Information about a compute node/platform of the overall system
class NodeResult:
    def __init__(self):
        self.designs: list[DesignResult] = []

    def add_design(self, result: DesignResult):
        self.designs.append(result)

    def to_dict(self) -> dict:
        stats = {}
        for i, design in enumerate(self.designs):
            stats[f"design_{i}"] = design.to_dict()
        return stats
    

class NodeEvaluator(ABC):
    def eval_network(self, config: dict):
        ...