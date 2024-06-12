from abc import ABC, abstractmethod

import yaml

## Base class for design space exploration. For a given Timeloop architecture description this
#  class is used to mutate the architecture configuration to evaluate different configurations.
class ArchitectureMutator(ABC):

    def __init__(self, cfg):
        self.cfg = cfg
        self.design_space = self.generate_design_space()


    @abstractmethod
    def mutate_arch(self):
        ...

    @abstractmethod
    def mutate_arch_constraints(self):
        ...
    
    @abstractmethod
    def mutate_map_constraints(self):
        ...

    @abstractmethod
    def generate_design_space(self):
        ...

    def generate_mapper(self):
        # Provide a default config that can be overwritten by child classes
        mapper = {}
        mapper["mapper"]["optimization-metrics"] = ["delay", "energy"]
        mapper["live-status"] = False
        mapper["num-threads"] = 8
        mapper["timeout"] = 0
        mapper["victory-condition"] = 100
        mapper["algorithm"] = "linear_pruned"


    def run(self):
        ...
