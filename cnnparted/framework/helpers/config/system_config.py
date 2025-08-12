from ast import Lambda
import yaml
from dataclasses import dataclass
from typing import Literal, TypedDict

@dataclass
class MnsimConfig:
    conf_path: str

    def __init__(self, cfg: dict):
        self.conf_path = cfg["conf_path"]

@dataclass
class ZigzagConfig:
    mapping: str
    optimization: str

    def __init__(self, cfg):
        self.mapping = cfg["mapping"]
        self.optimization = cfg["optimization"]

@dataclass
class TimeloopConfig:
    layer: str
    mapper: dict

    def __init__(self, cfg):
        self.layer = cfg["layer"]
        self.mapper = cfg["mapper"]

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
    min_size_mem_shared: list[int] 
    max_size_mem_shared: list[int]
    min_size_mem_local: list[int]
    max_size_mem_local: list[int]

    def __init__(self, cfg: dict):
        self.min_pe = cfg["min_pe"]
        self.max_pe = cfg["max_pe"]
        self.step_mem_shared = cfg["step_mem_shared"]
        self.min_size_mem_shared = cfg["min_size_mem_shared"]
        self.max_size_mem_shared = cfg["max_size_mem_shared"]
        self.min_size_mem_local = cfg["min_size_mem_local"]
        self.max_size_mem_local = cfg["max_size_mem_local"]

@dataclass
class NodeConfig:
    id: int
    bits: int
    fault_rates: list[float]
    faulty_bits: int
    max_memory_size: int
    #type: str
    accelerator: str
    evaluation: EvaluationConfig
    dse: DseNodeConfig

    def __init__(self, cfg: dict):
        self.id = cfg["id"]
        self.bits = cfg.get("bits", 8)
        self.fault_rates = cfg.get("fault_rates", [0.0, 0.0])
        self.faulty_bits = cfg.get("faulty_bits", 0)
        self.max_memory_size = cfg["max_memory_size"]
        self.evaluation = EvaluationConfig(cfg["evaluation"])
        self.accelerator = self.evaluation.accelerator
        self.dse = DseNodeConfig(cfg["dse"]) if "dse" in cfg else None


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

    def __init__(self, cfg):
        self.compute = []
        self.link = []
        compute_nodes = cfg["compute"]
        links = cfg["link"]

        for node in compute_nodes:
            self.compute.append(NodeConfig(node))
        for link in links:
            self.link.append(LinkConfig(link))