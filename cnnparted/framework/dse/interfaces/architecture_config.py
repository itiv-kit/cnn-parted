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

class ArchitectureAdaptor(ABC):
    def __init__(self):
        super().__init__()
        self.config: ArchitectureConfig = None