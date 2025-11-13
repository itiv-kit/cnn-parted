import numpy as np

from .onnx_layer import OnnxLayerParser
from ..layers.compute_layer import ComputationLayer
from framework.model.onnx_tensor import OnnxTensor
from framework.model.layer_parameters import GemmParameters 

class GemmParser(OnnxLayerParser):
    def parse(self, inputs: list[OnnxTensor], output: OnnxTensor) -> ComputationLayer:
        # Shape layout: [1, b, x, y]
        maybe_batch_dims = len(self.in_shape) - 2
        in_batch_dims = self.in_shape[:maybe_batch_dims]
        in_matmul_dims = self.in_shape[maybe_batch_dims:]
        out_matmul_dims = self.out_shape[maybe_batch_dims:]

        # Input shape should have either 2 or 4 elements
        assert len(in_batch_dims) == 2 or len(in_batch_dims) == 0

        INPUT_LEN_WITH_BIAS = 3 #if there is a bias the inputs are [ifm, wgt, bias]

        cfg = {
            'b': in_batch_dims[1] if len(in_batch_dims) == 2 else 1,
            'n': in_matmul_dims[0],
            'm': out_matmul_dims[1],
            'c': in_matmul_dims[1],
            'weights': np.prod(self.in_shape + self.out_shape),
            'has_bias': len(inputs) == INPUT_LEN_WITH_BIAS,
        }
        params = GemmParameters(cfg)
        gemm_layer = ComputationLayer(
            layer_name=self.node.name,
            layer_params=params,
            op_type=self.node.op_type,
            inputs=inputs,
            output=output,
        )
        return gemm_layer
