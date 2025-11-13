from abc import ABC

from framework.model.onnx_tensor import OnnxTensor
from framework.model.layer_parameters import AnyLayerParameters

class LayerBase(ABC):
    def __init__(self, 
                 layer_name: str, 
                 layer_params: AnyLayerParameters,
                 op_type: str,
                 inputs: list[OnnxTensor], 
                 output: OnnxTensor):
        self.layer_name = layer_name
        self.layer_params = layer_params
        self.op_type = op_type
        self.inputs = inputs
        self.output = output