import yaml
from dataclasses import dataclass
from typing import Literal, TypedDict

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
