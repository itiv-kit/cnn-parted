from abc import ABC, abstractmethod

## Base class for design space exploration. For a given Timeloop architecture description this
#  class is used to mutate the architecture configuration to evaluate different configurations.
class ArchitectureMutator(ABC):

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

    @abstractmethod
    def run(self):
        ...


class ArchitectureConfig(ABC):
        ...