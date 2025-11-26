import numpy as np

from .onnx_layer import OnnxLayerParser
from ..layer_parameters import UnaryMiscOperationParameters
from framework.model.onnx_tensor import OnnxTensor
from framework.model.layers.rename_layer import RenameLayer

class ClipParser(OnnxLayerParser):
    def parse(self, inputs: list[OnnxTensor], output: OnnxTensor) -> RenameLayer:
        data = {
            "in_shape": self.in_shape,
            "out_shape": self.out_shape,
            "kind": self.node.op_type,
        }
        params = UnaryMiscOperationParameters(data)
        act_layer = RenameLayer(
            layer_name=self.node.name,
            layer_params=params,
            op_type=self.node.op_type,
            inputs=inputs,
            output=output
        )
        return act_layer
