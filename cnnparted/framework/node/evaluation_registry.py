from framework.dse.interfaces.architecture_config import ArchitectureConfig

class AcceleratorRegistry:
    _REGISTRY: dict[str, ArchitectureConfig] = {}
    def __init_subclass__(cls, accelerator_name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._REGISTRY[accelerator_name] = cls
    
    def __getattribute__(self, name: str):
        return self._REGISTRY[name]