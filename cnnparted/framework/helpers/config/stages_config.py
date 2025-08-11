import yaml
from dataclasses import dataclass
from typing import Literal, TypedDict

@dataclass
class GeneticAlgConfig:
    algorithm: str
    num_gen: int
    pop_size: int

    def __init__(self, cfg):
        self.num_gen = cfg["num_gen"]
        self.pop_size = cfg["pop_size"]


@dataclass
class DseStageConfig:
    optimization: str
    top_k: int
    problem: str
    algorithm: str
    optimizer: GeneticAlgConfig

    def __init__(self, cfg: dict):
        self.optimization = cfg["optimization"]
        self.top_k = cfg["top_k"]
        self.problem = cfg["problem"]
        self.optimizer = GeneticAlgConfig(cfg["optimizer"])
