from abc import ABC, abstractmethod
import os
import yaml

class ArchitectureConfig(ABC):
    def to_yaml(self, dir: str):
        stats = vars(self)
        fname = os.path.join(dir, "architecture_params.yaml")
        with open(fname, "w") as f:
            yaml.dump(stats, f, sort_keys=False)

    @abstractmethod
    def get_config(self) -> dict:
        ...

## Base class for design space exploration. For a given Timeloop architecture description this
#  class is used to mutate the architecture configuration to evaluate different configurations.
class ArchitectureMutator(ABC):
    def __init__(self, cfg):
        self.design_space = []
        self.config: ArchitectureConfig = None
        self.design_space_exhausted = False
        self.cfg: dict = cfg
        self.tl_in_configs_dir: str = cfg["tl_in_configs_dir"]
        self.tl_out_configs_dir: str = ""

    @abstractmethod
    def mutate_arch(self, config=None, outdir=None):
        ...

    @abstractmethod
    def mutate_arch_constraints(self, config=None, outdir=None):
        ...
    
    @abstractmethod
    def mutate_map_constraints(self, config=None, outdir=None):
        ...

    @abstractmethod
    def generate_design_space(self):
        ...

    @abstractmethod
    def run(self):
        try:
            self.config = self.design_space.pop()
            self.mutate_arch()
            self.mutate_arch_constraints()
            self.mutate_map_constraints()
            # Check if there are any configs left to simulate
            if not self.design_space:
                self.design_space_exhausted=True
        except IndexError:
            self.design_space_exhausted = True
            self.config = None

        return self.config

    @abstractmethod
    def run_from_config(self, config, outdir=None):
        if outdir is None:
            outdir = self.tl_out_configs_dir

        self.mutate_arch(config=config, outdir=outdir)
        self.mutate_arch_constraints(config=config, outdir=outdir)
        self.mutate_map_constraints(config=config, outdir=outdir)
        

