from .onnx_layer import OnnxLayerParser
from ..layer_parameters import BinaryMiscOperationParameters 
from framework.model.onnx_tensor import OnnxTensor
from framework.model.layers.rename_layer import RenameLayer

class ConcatParser(OnnxLayerParser):
    def parse(self, inputs: list[OnnxTensor], output: OnnxTensor):
        cfg = {
            "a_shape": inputs[0].shape,
            "b_shape": inputs[1].shape,
            "out_shape": output.shape,
            "kind": self.node.op_type
        }
        params = BinaryMiscOperationParameters(cfg)
        layer = RenameLayer(
            layer_name=self.node.name,
            layer_params=params,
            op_type=self.node.op_type,
            inputs=inputs,
            output=output
        )
        return layer
