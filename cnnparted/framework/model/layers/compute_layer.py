from .abstract_layer import LayerBase
from framework.model.layer_parameters import ComputeParameters

class ComputationLayer(LayerBase):
    layer_params: ComputeParameters

    def __repr__(self):
        return f"layers.{self.__class__.__name__}(layer_name={self.layer_name}), op_type={self.op_type}, params={self.layer_params}"

# Specialization Classes
class ConvLayer(ComputationLayer): ...
class GemmLayer(ComputationLayer): ...