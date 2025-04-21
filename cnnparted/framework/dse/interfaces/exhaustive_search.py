from abc import ABC, abstractmethod

class ExhaustiveSearch(ABC):
    def __init__(self):
        self.design_space_exhausted = False

    @abstractmethod
    def generate_design_space(self):
        ...

    @abstractmethod
    def read_space_cfg(self, cfg):
        ...