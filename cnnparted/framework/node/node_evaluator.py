import os
import csv
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
        self.area: float  #must be in mm2
        self.latency: float #must be in ms
        self.energy: float #must be in mJ
        self.cycles: int 

    def to_dict(self) -> dict :
        return {self.name: 
                {"latency": self.latency,
                 "area": self.area,
                 "energy": self.energy}}
    
    def __repr__(self):
        return str(self.to_dict())
    
    def __str__(self):
        return str(self.to_dict())
    
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

    @classmethod
    def from_dict(cls, design_dict: dict):
        data = design_dict.values()
        arch = design_dict["arch_config"]
        layers = design_dict["layers"]
        design_res = cls()
        design_res.architecture = arch
        for name, layer in layers.items():
            l = LayerResult()
            l.name = name
            l.area = layer["area"]
            l.energy  = layer["energy"]
            l.latency = layer["latency"]
            design_res.add_layer(l)
        return design_res 

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

    @property
    def area(self):
        return self.layers[0].area
    
    @property
    def total_latency(self):
        return sum([layer.latency for layer in self.layers])
    
    @property
    def total_energy(self):
        return sum([layer.energy for layer in self.layers])

class NodeResult:
    """
    Complete information about a node on which the simulation is run. For the design space exploration multiple designs can be
    evaluated. For this reason we include a list of DesignResult here

    Attributes:
        designs:         Results for each design point
    """
    def __init__(self):
        self.designs: list[DesignResult] = []
        self.bits: int = 8
        self.fault_rates = [0.0, 0.0]
        self.faulty_bits = 0
        self.type = "tl"
        self.accelerator_name = ""

    @classmethod
    def from_dict(cls, node_dict: dict):
        node_res = cls()
        node_res.bits = node_dict.get("bits")
        node_res.fault_rates = node_dict.get("fault_rates")
        node_res.faulty_bits = node_dict.get("faulty_bits")
        node_res.type = node_dict.get("type")
        for design, eval in node_dict["eval"].items():
            des = DesignResult()
            des.architecture = eval["arch_config"]
            for name, layer in eval["layers"].items():
                l = LayerResult()
                l.name = name
                l.area = layer["area"]
                l.energy = layer["energy"]
                l.latency = layer["latency"]
                des.add_layer(l)
            node_res.add_design(des)
        return node_res

    def add_design(self, result: DesignResult):
        self.designs.append(result)

    def to_dict(self) -> dict:
        stats = {"bits": self.bits,
                 "eval": {},
                 "fault_rates": self.fault_rates,
                 "faulty_bits": self.faulty_bits,
                 "type": self.type,
                 }

        for i, design in enumerate(self.designs):
            stats["eval"][f"design_{i}"] = design.to_dict()
        return stats

    def write_csv(self, out_path: str):
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            header = [
                "No.",
                "Design Tag",
                "Layer",
                "Latency [ms]",
                "Energy [mJ]",
                'Area [mm2]'
            ]
            writer.writerow(header)
            row_num = 1
            for (tag, design_result) in enumerate(self.designs):
                for layer in design_result.layers:
                    row = [
                        row_num,
                        "design_" + str(tag),
                        layer.name,
                        str(layer.latency),
                        str(layer.energy),
                        str(layer.area),
                    ]
                    writer.writerow(row)
                    row_num += 1
    
    @classmethod
    def from_csv(cls, in_path: str):
        node_res = cls()

        design_result = DesignResult()
        with open(in_path, 'r', newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            current_design_tag = "design_0"
            for row in reader:
                if row["Design Tag"] != current_design_tag:
                    node_res.add_design(design_result)
                    design_result = DesignResult()
                    current_design_tag = row["Design Tag"]
                
                layer_res = LayerResult()
                layer_res.name = row["Layer"]
                layer_res.latency = float(row['Latency [ms]'])
                layer_res.energy = float(row['Energy [mJ]'])
                layer_res.area = float(row['Area [mm2]'])
                design_result.add_layer(layer_res)

        # Add results for the final design
        node_res.add_design(design_result)
        return node_res

    
class SystemResult:
    def __init__(self):
        self.platforms: dict[int, NodeResult] = {}

    def __getitem__(self, item: int) -> NodeResult:
        return self.platforms[item]

    def register_platform(self, id: int):
        self.platforms[id] = NodeResult()

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
    
    def to_dict(self) -> dict:
        stats = {}
        for id, node_res in self.platforms.items():
            stats[id] = node_res.to_dict()
        return stats
    
    def to_csv(self, out_path, runname: str = ""):
        for id, node_res in self.platforms.items():
            accelerator_name= ""
            file_str = str(id) + "_" + accelerator_name + "tl_layers.csv"
            file_path = os.path.join(out_path,  file_str)
            node_res.write_csv(file_path)
    

class NodeEvaluator(ABC):
    """
    Interface which all node evluators should inherit from
    """
    fname_result: str
    config: dict

    def set_workdir(self, work_dir: str, runname: str, id: int):
        self.workdir = work_dir
        self.runroot = os.path.join(work_dir, "system_evaluation", str(id)+"_"+self.config["accelerator"])
        fname_csv = os.path.join(work_dir, str(id) + "_" + self.config["accelerator"] + "_" + self.fname_result)
        return fname_csv

    def run(self, layers: list):
        ...