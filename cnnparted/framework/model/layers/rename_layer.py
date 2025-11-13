from .abstract_layer import LayerBase
from framework.model.onnx_tensor import OnnxTensor

class RenameLayer(LayerBase):
    ...
    def __repr__(self):
        return f"layers.{self.__class__.__name__}(layer_name={self.layer_name}), op_type={self.op_type}, params={self.layer_params}"

class InputLayer(LayerBase):
    def __init__(self, layer_name: str, output: OnnxTensor):
        super().__init__(layer_name=layer_name, op_type="Input", layer_params=None, inputs=[], output=output)

class OutputLayer(LayerBase):
    def __init__(self, layer_name: str, input: list[OnnxTensor]):
        super().__init__(layer_name=layer_name, op_type="Output", layer_params=None, inputs=input, output=[])