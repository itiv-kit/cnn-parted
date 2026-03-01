from dataclasses import dataclass
from typing import Literal
from enum import Enum, auto

@dataclass
class GeneticAlgConfig:
    algorithm: str
    num_gen: int
    pop_size: int

    def __init__(self, cfg):
        self.algorithm = cfg["algorithm"]
        self.num_gen = cfg["num_gen"]
        self.pop_size = cfg["pop_size"]

@dataclass
class SimulatedAnnealingConfig:
    latency_weight: float
    edp_weight: float
    util_weight: float

    def __init__(self, cfg: dict):
        self.latency_weight = cfg["latency_weight"]
        self.edp_weight = cfg["edp_weight"]
        self.util_weight = cfg["util_weight"]

@dataclass
class OptunaConfig:
    n_trials: int
    sampler: Literal["random", "grid", "tpe", "cmaes", "nsga2", "qmc", "gp"]

    def __init__(self, cfg: dict):
        self.n_trials = int(cfg["n_trials"])
        self.sampler = cfg["sampler"]

@dataclass
class PartitioningStageConfig:
    fixed_sys: bool
    acc_once: bool
    num_pp: int
    num_jobs: int
    num_topos: int
    optimization: Literal["all", "edp", "ppa"]
    max_num_platforms: int | None

    def __init__(self, cfg: dict):
        self.fixed_sys = cfg.get("fixed_sys", False)
        self.acc_once = cfg.get("acc_once", False)
        self.num_pp = cfg.get("num_pp", -1)
        self.num_jobs = cfg.get("num_jobs", 1)
        self.num_topos = cfg.get("num_topos", 1)
        self.optimization = cfg.get("optimization", "all")
        self.max_num_platforms = cfg.get("max_num_platforms", -1)

@dataclass
class DseStageConfig:
    optimization: str
    top_k: int
    prune_strict: bool
    problem: str
    algorithm: str
    optimizer: GeneticAlgConfig

    def __init__(self, cfg: dict):
        self.optimization = cfg.get("optimization", "edp")
        self.top_k = cfg.get("top_k", -1)
        self.prune_strict = cfg.get("prune_strict", False)
        self.problem = cfg["problem"]
        self.optimizer = GeneticAlgConfig(cfg["optimizer"])
        self.algorithm = self.optimizer.algorithm

class SchedulingPriority(Enum):
    EARLIEST_START = auto()
    EARLIEST_FINISH = auto()
    IDLE_TIME = auto()
    LONGEST_JOB = auto()
    SHORTEST_JOB = auto()
    DEADLINE = auto()

@dataclass
class MultiTenancyStageConfig:
    optimizer: GeneticAlgConfig | SimulatedAnnealingConfig | OptunaConfig
    algorithm: Literal["sa", "genetic", "optuna"]
    scheduling: Literal["list", "multi_queue"]
    scheduling_priority: SchedulingPriority
    binding_granularity: Literal["tile", "layer", "network"]
    num_jobs: int

    def __init__(self, cfg: dict):
        self.algorithm = cfg["algorithm"]
        self.scheduling = cfg["scheduling"]
        self.scheduling_priority = SchedulingPriority[cfg["scheduling_priority"].upper()]
        self.binding_granularity = cfg["binding_granularity"]
        self.num_jobs = cfg["num_jobs"]

        match self.algorithm:
            case "sa":
                self.optimizer = SimulatedAnnealingConfig(cfg["optimizer"])
            case "genetic":
                self.optimizer = GeneticAlgConfig(cfg["optimizer"])
            case "optuna":
                self.optimizer = OptunaConfig(cfg["optimizer"])