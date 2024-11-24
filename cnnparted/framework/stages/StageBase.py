from abc import ABC, abstractmethod

STAGE_DEPENDENCIES = {}

class Stage(ABC):

    def __init__(self):
        self.name: str = self.__class__.__name__

    @abstractmethod
    def _update_artifacts(self, artifacts: dict):
        artifacts["stages"][self.name] = {}
        artifacts["stages"][self.name]["out"] = {}
        

    @abstractmethod
    def _take_artifacts(self, artifacts: dict):
        ...
    
    @abstractmethod
    def run(self, artifacts: dict):
        ...

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, value: object) -> bool:
        return self.name == value.name and isinstance(value, Stage)


# Decorator to mark required stages
def register_required_stage(*required_stages):
    def wrapper(cls):
        STAGE_DEPENDENCIES[cls] = []
        for s in required_stages:
            STAGE_DEPENDENCIES[cls].append(s)
        return cls
    
    return wrapper