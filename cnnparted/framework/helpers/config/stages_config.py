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

    def __init__(self, cfg: dict):
        self.algorithm = cfg["algorithm"]
        self.scheduling = cfg["scheduling"]
        self.scheduling_priority = SchedulingPriority[cfg["scheduling_priority"].upper()]

        match self.algorithm:
            case "sa":
                self.optimizer = SimulatedAnnealingConfig(cfg["optimizer"])
            case "genetic":
                self.optimizer = GeneticAlgConfig(cfg["optimizer"])
            case "optuna":
                self.optimizer = OptunaConfig(cfg["optimizer"])