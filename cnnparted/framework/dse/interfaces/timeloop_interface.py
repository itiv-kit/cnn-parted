from abc import ABC, abstractmethod
import os
import yaml

from framework.dse.interfaces.architecture_config import ArchitectureConfig

## Base class for design space exploration. For a given Timeloop architecture description this
#  class is used to mutate the architecture configuration to evaluate different configurations.
class TimeloopInterface(ABC):
    def __init__(self, cfg):
        self.design_space = []
        self.config: ArchitectureConfig = None
        self.cfg: dict = cfg
        self.tl_in_configs_dir: str = cfg["tl_in_configs_dir"]
        self.tl_out_configs_dir: str = ""

    @abstractmethod
    def write_tl_arch(self, config=None, outdir=None):
        ...

    @abstractmethod
    def write_tl_arch_constraints(self, config=None, outdir=None):
        ...
    
    @abstractmethod
    def write_tl_map_constraints(self, config=None, outdir=None):
        ...


    @abstractmethod
    def run(self):
        self.write_tl_arch()
        self.write_tl_arch_constraints()
        self.write_tl_map_constraints()


    @abstractmethod
    def run_from_config(self, config, outdir=None):
        if outdir is None:
            outdir = self.tl_out_configs_dir

        self.write_tl_arch(config=config, outdir=outdir)
        self.write_tl_arch_constraints(config=config, outdir=outdir)
        self.write_tl_map_constraints(config=config, outdir=outdir)
        

