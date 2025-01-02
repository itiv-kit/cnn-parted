import os
from abc import ABC

class LayerResult:
    """ 
    Contains basic information about a single layer of a nerual networ

    Attributes:
        name:           A name that must be unique in the network
        area:           Area of the accelerator this layer runs on
        latency:        Latency when this layer is executed
        energy:         Energy required to run this layer on an accelerator
        cycles          Latency in clock cycles
    """
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

class DesignResult:
    """
    Information about the execution of a full network and the specific architecture it is run on.
    The architecture can contain information such as number of PEs, size of memory, Crossbar specification etc. and is accelerator specific.
    
    Attributes:
        layers:         List with the results of the individual layers
        architecture:   Information about the architecture
    """
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

class NodeResult:
    """
    Complete information about a node on which the simulation is run. For the design space exploration multiple designs can be
    evaluated. For this reason we include a list of DesignResult here

    Attributes:
        designs:         Results for each design point
    """
    def __init__(self):
        self.designs: list[DesignResult] = []

    def add_design(self, result: DesignResult):
        self.designs.append(result)

    def to_dict(self) -> dict:
        stats = {}
        for i, design in enumerate(self.designs):
            stats[f"design_{i}"] = design.to_dict()
        return stats
    
class SystemResult:
    def __init__(self):
        self.platforms = {}

    def add_platform(self, id: int, result: NodeResult):
        self.platforms[id] = result

    def get_num_platforms(self):
        return len(self.platforms)

    def get_num_designs(self):
        ...

    def get_platform_ids(self):
        return list(self.platforms.keys())

    def get_design_tags(self, platform_id: int):
        return list(self.platforms[platform_id]["eval"].keys())


class NodeEvaluator(ABC):
    """
    Interface which all node evluators should inherit from
    """
    fname_result: str
    config: dict

    def set_workdir(self, work_dir: str, runname: str, id: int):
        self.workdir = work_dir
        self.runroot = os.path.join(work_dir, "system_evaluation", str(id)+"_"+self.config["accelerator"])
        fname_csv = os.path.join(work_dir, runname + "_" + str(id) + "_" + self.config["accelerator"] + "_" + self.fname_result)
        return fname_csv

    def run(self, layers: list):
        ...