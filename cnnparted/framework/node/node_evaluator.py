import os
import csv
from abc import ABC
from copy import deepcopy
from functools import reduce
import operator

from framework.helpers.design_metrics import calc_metric

import numpy as np

class LayerResult:
    """ 
    Contains basic information about a single layer of a neural network

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
    

class NetworkResult:
    """
    Information about the execution of the networks for a given design of a specific accelerator
    """
    def __init__(self):
        self._networks: dict[str, list[LayerResult]] = {}

    def get_networks(self):
        return list(self._networks.keys())
    
    def get_network_dict(self, network: str):
        layers_list_of_dict = [l.to_dict() for l in self._networks[network]]
        layers = reduce(operator.ior, layers_list_of_dict, {})
        return layers

    def add_layer(self, results: LayerResult, network: str):
        if network not in self._networks:
            self._networks[network] = []

        self._networks[network].append(results)
    
    def get_networks_dict(self):
        networks = {network: self.get_network_dict(network) for network  in self._networks.keys()}
        return networks
    
    def __getitem__(self, item: str):
        return self._networks[item]

    
class DesignResult:
    """
    Information about the execution of a full network and the specific architecture it is run on.
    The architecture can contain information such as number of PEs, size of memory, Crossbar specification etc. and is accelerator specific.
    
    Attributes:
        layers:         List with the results of the individual layers
        architecture:   Information about the architecture
    """
    def __init__(self, architecture: dict = {}):
        self.tag: str = ""
        self.networks: dict[str, list[LayerResult]] = {}

        #TODO Remove after networks member is implemented
        self.layers: list[LayerResult] = []
        self.layers_dict: dict[str, LayerResult] = {}

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

    def add_layer(self, results: LayerResult, network: str = "test"):
        self.layers.append(results)
        self.layers_dict[results.name] = results


    def to_dict(self) -> dict:
        res = {}
        res["arch_config"] = self.architecture
        res["networks"] = {}

        for nw in self.networks.keys():
            res["networks"][nw] = self.get_network_dict(nw)

        #res["layers"] = {} 
        #for layer in self.layers:
        #    ld: dict = layer.to_dict()
        #    name = list(ld.keys())[0]
        #    data = list(ld.values())[0]
        #    res["layers"][name] = data
        return res
    
    def get_layer_dict(self) -> dict:
        layers = {}
        for layer in self.layers:
            name = layer.name
            layers[name] = {
                "area": layer.area,
                "energy": layer.energy,
                "latency": layer.latency,
                }
        return layers

    def get_networks(self):
        return list(self.networks.keys())
    
    def get_network_dict(self, network: str):
        layers_list_of_dict = [l.to_dict() for l in self.networks[network]]
        layers = reduce(operator.ior, layers_list_of_dict, {})
        return layers

    def add_layer_to_network(self, results: LayerResult, network: str):
        if network not in self.networks:
            self.networks[network] = []

        self.networks[network].append(results)
    
    def get_networks_dict(self):
        networks = {network: self.get_network_dict(network) for network  in self.networks.keys()}
        return networks

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
    def __init__(self, config: dict):
        self.designs: list[DesignResult] = []
        self._all_designs: list[DesignResult] = [] #only set by the prune_designs function for bookkeeping
        self.design_tag = 0
        self.bits = config.get("bits", 8)
        self.fault_rates = config.get("fault_rates", [0.0, 0.0])
        self.faulty_bits = config.get("faulty_bits", 0)
        self.max_memory_size = config["max_memory_size"]
        self.type = config["evaluation"]["simulator"]
        self.accelerator_name = config["evaluation"]["accelerator"]
        self.sim_time = None


    @classmethod
    def from_dict(cls, node_dict: dict, node_config: dict):
        node_res = cls(node_config)
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
        result.tag = "design_" + str(self.design_tag)
        self.designs.append(result)
        self.design_tag += 1

    def to_dict(self) -> dict:
        stats = {"bits": self.bits,
                 "eval": {},
                 "fault_rates": self.fault_rates,
                 "faulty_bits": self.faulty_bits,
                 "type": self.type,
                 }

        for i, design in enumerate(self.designs):
            stats["eval"][design.tag] = design.to_dict()
        return stats

    def to_csv(self, out_path: str):
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

    def prune_designs(self, top_k: int, metric: str, strict: bool=False):
        design_stats = self.to_dict()
        self._all_designs = deepcopy(self.designs)

        if len(design_stats["eval"]) <= top_k or top_k == -1 or any( [len(d.networks)>1 for d in self.designs]) :
            return

        network = self.designs[0].get_networks()[0]

        # The metric_per_design array has this structure, with
        # every cell holding EAP, EDP or some other metric:
        #  x | l0 | l1 | l2 | l3 |
        # ------------------------
        # d0 | ...| ...| ...| ...|
        # d1 | ...| ...| ...| ...|
        metric_per_design = []
        energy_per_design = []
        latency_per_design = []
        area_per_design = []

        for tag, design in design_stats["eval"].items():
            #tag = design["tag"]
            layers = design["networks"][network]
            energy_per_layer = []
            latency_per_layer = []
            for name, layer in layers.items():
                energy_per_layer.append(layer["energy"])
                latency_per_layer.append(layer["latency"])

            energy_per_design.append(energy_per_layer)
            latency_per_design.append(latency_per_layer)
            area_per_design.append([layer["area"]])

        metric_per_design = calc_metric(np.array(energy_per_design), np.array(latency_per_design), np.array(area_per_design), metric, reduction=strict)

        # Now, we need to find the top_k designs per layer
        design_candidates = []
        if not strict:
            for col in metric_per_design.T:
                metric_for_layer = col.copy()
                metric_for_layer = np.argsort(metric_for_layer)

                for i in metric_for_layer[0:top_k]:
                    design_candidates.append(f"design_{i}")
        else:
            metric_per_design_sort = np.argsort(metric_per_design.flatten())
            for i in metric_per_design_sort[0:top_k]:
                design_candidates.append(f"design_{i}")

        design_candidates = np.unique(design_candidates)

        pruned_stats = {tag: results for tag, results in design_stats["eval"].items() if tag in design_candidates}

        pruned_designs = [design for design in self.designs if design.tag in design_candidates]
        self.designs = pruned_designs

    
    @classmethod
    def from_csv(cls, in_path: str, network: str, node_config: dict):
        node_res = cls(node_config)

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
                design_result.add_layer_to_network(layer_res, network)

        # Add results for the final design
        node_res.add_design(design_result)
        return node_res

    
class SystemResult:
    def __init__(self):
        self.platforms: dict[int, NodeResult] = {}

    def __getitem__(self, item: int) -> NodeResult:
        return self.platforms[item]

    def register_platform(self, id: int, node_config):
        self.platforms[id] = NodeResult(node_config)

    def add_platform(self, id: int, result: NodeResult):
        self.platforms[id] = result

    def get_num_platforms(self):
        return len(self.platforms)

    def get_num_designs(self):
        ...

    def get_platform_ids(self):
        return list(self.platforms.keys())

    def get_design_tags(self, platform_id: int):
        return [design.tag for design in self.platforms[platform_id].designs]
    
    def to_dict(self) -> dict:
        stats = {}
        for id, node_res in self.platforms.items():
            stats[id] = node_res.to_dict()
        return stats
    
    def to_csv(self, out_path, runname: str = ""):
        for id, node_res in self.platforms.items():
            file_str = str(id) + "_" + node_res.accelerator_name + "_tl_layers.csv"
            file_path = os.path.join(out_path,  file_str)
            node_res.to_csv(file_path)
    

class NodeEvaluator(ABC):
    """
    Interface which all node evaluators should inherit from
    """
    fname_result: str
    config: dict

    def set_workdir(self, work_dir: str, network: str, id: int):
        self.workdir = work_dir
        self.runroot = os.path.join(work_dir, "system_evaluation", str(id)+"_"+self.config["accelerator"])
        fname_csv = os.path.join(work_dir, str(id) + "_" + network + "_" + self.config["accelerator"] + "_" + self.fname_result)
        return fname_csv

    def run(self, network: str, layers: list):
        ...