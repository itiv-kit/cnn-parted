from abc import ABC, abstractmethod

import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation


class Genome(ABC):
    N_VAR_OPT: int

    def __init__(self):
        super().__init__()
        self.crossover: Crossover = None
        self.mutation: Mutation = None

    @classmethod
    def n_var_opt(cls):
        """
        Returns the number of variables needed for the optimizer. For example,
        if the accelerator design space is [xdim, ydim] this should return two.
        Generally:  n_var_opt = len(pre_array_dim)+len(pre_array_mem)+len(local_mems)
        """
        return cls.N_VAR_OPT

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
