from framework.dse.interfaces.architecture_config import ArchitectureConfig, ArchitectureAdaptor
    

class AcceleratorRegistry:
    _arch_registry: dict[str, ArchitectureConfig] = {}
    _adaptor_registry: dict[str, ArchitectureAdaptor] = {}

    @classmethod
    def register_architecture(cls, accelerator_name: str):
        def inner_wrapper(wrapped_class: ArchitectureConfig):
            if accelerator_name not in cls._arch_registry:
                cls._arch_registry[accelerator_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def register_adaptor(cls, accelerator_name: str):
        def inner_wrapper(wrapped_class: ArchitectureAdaptor):
            if accelerator_name not in cls._adaptor_registry:
                cls._adaptor_registry[accelerator_name] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
    @classmethod
    def get_architecture(cls, accelerator_name: str):
        return cls._arch_registry[accelerator_name]
    
    @classmethod
    def get_adaptor(cls, accelerator_name: str):
        return cls._adaptor_registry[accelerator_name]
    