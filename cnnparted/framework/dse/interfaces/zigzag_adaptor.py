from abc import ABC, abstractmethod
from pathlib import Path
import os

from framework.constants import ROOT_DIR

## Base class for design space exploration. For a given Timeloop architecture description this
#  class is used to mutate the architecture configuration to evaluate different configurations.
class ZigzagAdaptor(ABC):
    def __init__(self):
        super().__init__()
        self.zigzag_in_configs_dir_local = os.path.join(ROOT_DIR, "configs", "zigzag_configs")
        self.zigzag_in_configs_dir_repo = os.path.join(ROOT_DIR, "tools", "zigzag", "zigzag", "inputs")
        self.zigzag_out_configs_dir: str = ""
    
    def _find_config(self, in_config_file: str | Path):
        if (local_path := Path(self.zigzag_in_configs_dir_local, in_config_file)).exists():
            return str(local_path)
        elif(repo_path := Path(self.zigzag_in_configs_dir_repo, in_config_file)).exists(): 
            return str(repo_path)

    @abstractmethod
    def write_zigzag_arch(self, config=None, outdir=None):
        ...

    @abstractmethod
    def write_zigzag_mapping(self, config=None, outdir=None):
        ...

    @abstractmethod
    def run_zigzag(self):
        self.write_zigzag_arch()
        self.write_zigzag_mapping()


    @abstractmethod
    def run_zigzag_from_config(self, config, outdir=None):
        if outdir is None:
            outdir = self.tl_out_configs_dir

        self.write_zigzag_arch(config=config, outdir=outdir)
        self.write_zigzag_mapping(config=config, outdir=outdir)
        


