from dataclasses import dataclass
from typing import Literal
import itertools as it
import math

import numpy as np

from framework.helpers.dtypes import NodeId

@dataclass
class MnsimConfig:
    conf_path: str

    def __init__(self, cfg: dict):
        self.conf_path = cfg["conf_path"]

@dataclass
class ZigzagConfig:
    mapping: str
    optimization: Literal["EDP", "latency", "energy"]

    def __init__(self, cfg):
        self.mapping = cfg["mapping"]
        self.optimization = cfg["optimization"]

@dataclass
class TimeloopConfig:
    layer: str
    mapper: dict

    def __init__(self, cfg):
        self.layer = cfg["layer"]
        self.mapper = cfg.get("mapper", {})

@dataclass
class EvaluationConfig:
    accelerator: str
    frequency: int | None
    simulator: Literal["timeloop", "zigzag", "mnsim"]
    simulator_cfg : MnsimConfig | ZigzagConfig | TimeloopConfig

    def __init__(self, cfg: dict):
        self.accelerator = cfg["accelerator"]
        self.simulator = cfg["simulator"]

        match cfg["simulator"]:
            case "timeloop":
                self.simulator_cfg = TimeloopConfig(cfg)
                self.frequency = cfg["frequency"]
            case "mnsim":
                self.simulator_cfg = MnsimConfig(cfg)
            case "zigzag":
                self.simulator_cfg = ZigzagConfig(cfg)
                self.frequency = cfg["frequency"]

@dataclass
class DseNodeConfig:
    min_pe: list[int]
    max_pe: list[int]
    step_mem_shared: list[int]
    step_mem_local: list[int]
    min_size_mem_shared: list[int] 
    max_size_mem_shared: list[int]
    min_size_mem_local: list[int]
    max_size_mem_local: list[int]
    mutator: str | None

    def __init__(self, cfg: dict):
        self.min_pe = cfg["min_pe"]
        self.max_pe = cfg["max_pe"]
        self.step_mem_shared = cfg.get("step_mem_shared", 8)
        self.min_size_mem_shared = cfg["min_size_mem_shared"]
        self.max_size_mem_shared = cfg["max_size_mem_shared"]
        self.min_size_mem_local = cfg.get("min_size_mem_local", [1,1,1])
        self.max_size_mem_local = cfg.get("max_size_mem_local", [1,1,1])

        default_local_step = [(mx+1) - mn for mn, mx in zip(self.min_size_mem_shared, self.max_size_mem_shared)]
        self.step_mem_local = cfg.get("step_mem_local", default_local_step) #default: one step
        self.mutator = cfg.get("mutator", None)

@dataclass
class NodeConfig:
    id: NodeId
    bits: int
    fault_rates: list[float]
    faulty_bits: int
    max_memory_size: int
    accelerator: str
    instances: int
    noc_coords: list[tuple[int, ...]] | None | Literal["auto"]
    virtual_ids: dict[str, int] | None
    evaluation: EvaluationConfig
    dse: DseNodeConfig

    def __init__(self, cfg: dict):
        self.id = NodeId(cfg["id"])
        self.bits = cfg.get("bits", 8)
        self.fault_rates = cfg.get("fault_rates", [0.0, 0.0])
        self.faulty_bits = cfg.get("faulty_bits", 0)
        self.max_memory_size = cfg["max_memory_size"]
        self.evaluation = EvaluationConfig(cfg["evaluation"])
        self.accelerator = self.evaluation.accelerator
        self.instances = cfg.get("instances", 1)
        self.noc_coords = cfg.get("noc_coords") if "noc_coords" in cfg else None
        self.dse = DseNodeConfig(cfg["dse"]) if "dse" in cfg else None
        self.virtual_ids: dict[str, NodeId] | None = {} if self.instances > 1 else None

        # Check if we need to assign virtual IDs, assign them to None. System config assigns actual int ID
        if self.instances > 1:
            for i in range(1, self.instances):
                v_id = "v" + str(i)
                self.virtual_ids[v_id] = None


@dataclass
class LinkConfig:
    id: int
    max_out_size: int
    data_bit_width: int
    fps: int
    parameters: dict
    kind: Literal["ethernet", "noi", "ble"]

    def __init__(self, cfg):
        self.id = cfg["id"]
        self.max_out_size = cfg["max_out_size"]
        self.data_bit_width = cfg["data_bit_width"]
        self.fps = cfg["fps"]

        if "ethernet" in cfg: 
            self.parameters = cfg["ethernet"]
            self.kind = "ethernet"
        elif "noi" in cfg:
            self.parameters = cfg["noi"]
            self.kind = "noi"
        elif "ble" in cfg:
            self.parameters = cfg["ble"]
            self.kind = "ble"

@dataclass
class DramConfig:
    location: Literal["below", "above", "left", "right"] | None
    neighbors: list[list[int]] | None
        
@dataclass
class NocConfig:
    topology: Literal["fully_connected", "mesh"]
    shape: list[int]
    dram: DramConfig

    def __init__(self, cfg):
        self.topology = cfg["topology"]
        self.shape = cfg["shape"]
        #self.dram = DramConfig(cfg["dram"])

@dataclass
class PlatformIdInfo:
    accelerator_name: str
    node_config: NodeConfig
    noc_coord: tuple[int, ...] = None


@dataclass
class SystemConfig:
    compute: list[NodeConfig]
    link: list[LinkConfig]
    num_platforms: int
    platform_ids: list[int]
    platform_names: list[str]
    noc: NocConfig | None

    def __init__(self, cfg):
        self.compute = []
        self.link = []
        compute_nodes = cfg["compute"]
        links = cfg["link"]

        for node in compute_nodes:
            self.compute.append(NodeConfig(node))
        for link in links:
            self.link.append(LinkConfig(link))
        
        self.num_platforms = sum([c.instances for c in self.compute])
        self.max_id = max([c.id for c in self.compute] + [lnk.id for lnk in self.link])
        self.platform_ids = []
        self.platform_names = []
        self.platform_id_mapping: dict[int, PlatformIdInfo] = {}

        # Assign int IDs for all nodes
        for c in self.compute:
            self.platform_ids.append(c.id)
            accelerator_name = str(c.id)+"_"+c.accelerator
            self.platform_names.append(accelerator_name)
            self.platform_id_mapping[c.id] = PlatformIdInfo(accelerator_name, c)
            if c.virtual_ids is not None:
                for v_ids in c.virtual_ids.keys():
                    self.max_id += 1
                    current_id = self.max_id
                    c.virtual_ids[v_ids] = NodeId(current_id)
                    self.platform_ids.append(current_id)
                    accelerator_name = str(current_id)+"_"+c.accelerator
                    self.platform_names.append(accelerator_name)
                    self.platform_id_mapping[current_id] = PlatformIdInfo(accelerator_name, c)
            #else:

        # Configuration for the NoC
        self.noc = NocConfig(cfg["noc"]) if "noc" in cfg else None
        all_available_slots = it.product(*(range(s) for s in self.noc.shape))

        if self.noc is not None:
            assert self.num_platforms <= math.prod(self.noc.shape), "Number of platforms exceeds available slots of the configured NoC"
            for compute_node in self.compute:
                if compute_node.noc_coords is None or compute_node.noc_coords == "auto":
                    # Generate default values for indices
                    compute_node.noc_coords = []
                    
                    virtual_ids = list(compute_node.virtual_ids.values()) if c.virtual_ids is not None else []
                    for current_id in [compute_node.id] + virtual_ids:
                        next_available_slot = next(all_available_slots)
                        self.platform_id_mapping[current_id].noc_coord = next_available_slot
                        compute_node.noc_coords.append(next_available_slot)

                elif isinstance(compute_node.noc_coords, list):
                    assert compute_node.noc_coords is not None, f"NoC coordinates are not specified for platform ID {compute_node.id}" 
                    assert len(compute_node.noc_coords) == compute_node.instances, f"Must specify all NoC coordinates for platform ID {compute_node.id}"
