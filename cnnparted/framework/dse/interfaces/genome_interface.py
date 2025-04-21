from abc import ABC, abstractmethod

import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation


class GenomeInterface(ABC):
    def __init__(self):
        self.crossover: Crossover = None
        self.mutation: Mutation = None

    @property
    @abstractmethod
    def pe_array_dim(self):
        ...

    @property
    @abstractmethod
    def pe_array_mem(self):
        ...
    
    @property
    @abstractmethod
    def local_mems(self):
        ...

    @classmethod
    @abstractmethod
    def from_genome(cls, genome: list):
        ...

    @abstractmethod
    def to_genome(self) -> list:
        ...
