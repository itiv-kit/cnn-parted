from .onnx_layer import OnnxLayerParser
from ..layer_parameters import ElementwiseParameters 
from framework.model.onnx_tensor import OnnxTensor
from framework.model.layers.rename_layer import RenameLayer

class ElementwiseParser(OnnxLayerParser):
    def parse(self, inputs: list[OnnxTensor], output: OnnxTensor) -> RenameLayer:
        data = {
            "shape": self.in_shape,
            "kind": self.node.op_type,
        }
        params = ElementwiseParameters(data)
        eltwise_layer = RenameLayer(
            layer_name=self.node.name,
            layer_params=params,
            op_type=self.node.op_type,
            inputs=inputs,
            output=output
        )
        return eltwise_layer 
        