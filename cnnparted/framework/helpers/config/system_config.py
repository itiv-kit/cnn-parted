from dataclasses import dataclass
from typing import Literal

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
class SystemConfig:
    compute: list[NodeConfig]
    link: list[LinkConfig]
    num_platforms: int
    platform_ids: list[int]
    platform_names: list[str]

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

        # Assign int IDs for all nodes
        for c in self.compute:
            self.platform_ids.append(c.id)
            self.platform_names.append(str(c.id)+"_"+c.accelerator)
            if c.virtual_ids is not None:
                for v_ids in c.virtual_ids.keys():
                    self.max_id += 1
                    c.virtual_ids[v_ids] = NodeId(self.max_id)
                    self.platform_ids.append(c.virtual_ids[v_ids])
                    self.platform_names.append(str(c.virtual_ids[v_ids])+"_"+c.accelerator)
