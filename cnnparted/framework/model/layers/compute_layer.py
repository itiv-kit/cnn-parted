from dataclasses import dataclass

from .abstract_layer import LayerBase
from framework.model.layer_parameters import ComputeParameters
from framework.model.tiling.iterators import LoopDim, ConcreteLoopIterator
from framework.model.tensor_wrapper import TensorWrapper

@dataclass
class SchedulingInfo:
    binding: int = None
    earliest_start: float = 0.0
    start_time: float = None
    end_time: float = None


class ComputationLayer(LayerBase):
    layer_params: ComputeParameters
    scheduling: SchedulingInfo

    def __init__(self, 
                 layer_name: str, 
                 layer_params: ComputeParameters,
                 op_type: str,
                 inputs: list[TensorWrapper], 
                 output: TensorWrapper,
                 loop_dims: list[ConcreteLoopIterator]):
        self.layer_name = layer_name
        self.layer_params = layer_params
        self.op_type = op_type
        self.inputs = inputs
        self.output = output
        self.loop_dims = loop_dims
        self.scheduling = SchedulingInfo()

    def __repr__(self):
        return f"layers.{self.__class__.__name__}(layer_name={self.layer_name}), op_type={self.op_type}, params={self.layer_params}"

# Specialization Classes
class ConvLayer(ComputationLayer): ...
class GemmLayer(ComputationLayer): ...