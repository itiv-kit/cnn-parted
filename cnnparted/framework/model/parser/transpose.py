from .onnx_layer import OnnxLayerParser
from ..layer_parameters import UnaryMiscOperationParameters 
from framework.model.onnx_tensor import OnnxTensor
from framework.model.layers.rename_layer import RenameLayer


class TransposeParser(OnnxLayerParser):
    def parse(self, inputs: list[OnnxTensor], output: OnnxTensor):
        assert len(inputs) == 1, f"Error when parsing layer of type {self.node.op_type}"
        cfg = {
            "in_shape": inputs[0].shape,
            "out_shape": output.shape,
            "kind": self.node.op_type
        }
        params = UnaryMiscOperationParameters(cfg)
        transpose_layer = RenameLayer(
            layer_name=self.node.name,
            layer_params=params,
            op_type=self.node.op_type,
            inputs=inputs,
            output=output,
        )
        return transpose_layer