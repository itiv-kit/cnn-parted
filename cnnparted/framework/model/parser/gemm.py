import numpy as np

from .onnx_layer import OnnxLayerParser
from ..layers.compute_layer import ComputationLayer
from framework.model.onnx_tensor import OnnxTensor
from framework.model.layer_parameters import GemmParameters
from framework.model.tiling.iterators import ConcreteLoopIterator, LoopDim
from framework.model.tensor_wrapper import TensorWrapper, OperandKind

class GemmParser(OnnxLayerParser):
    def parse(self, inputs: list[OnnxTensor], output: OnnxTensor) -> ComputationLayer:
        # Shape layout: [1, b, x, y]
        maybe_batch_dims = len(self.in_shape) - 2
        in_batch_dims = self.in_shape[:maybe_batch_dims]
        in_matmul_dims = self.in_shape[maybe_batch_dims:]
        out_matmul_dims = self.out_shape[maybe_batch_dims:]

        has_batch_dims = len(in_batch_dims) == 2

        # Input shape should have either 2 or 4 elements
        assert len(in_batch_dims) == 2 or len(in_batch_dims) == 0

        INPUT_LEN_WITH_BIAS = 3 #if there is a bias the inputs are [ifm, wgt, bias]
        has_bias = len(inputs) == INPUT_LEN_WITH_BIAS

        cfg = {
            'b': in_batch_dims[1] if has_batch_dims else 1,
            'n': in_matmul_dims[0],
            'm': out_matmul_dims[1],
            'c': in_matmul_dims[1],
            'weights': np.prod(self.in_shape + self.out_shape),
            'has_bias': has_bias,
        }

        params = GemmParameters(cfg)

        loop_dims = [
            ConcreteLoopIterator(LoopDim.N, params.n),
            ConcreteLoopIterator(LoopDim.M, params.m),
            ConcreteLoopIterator(LoopDim.C, params.c)
        ]
        if has_batch_dims:
            loop_dims = [ConcreteLoopIterator(LoopDim.B, params.b)] + loop_dims
        
        # Wrap all tensor to save loop iteration information
        inp, wgt, outp = self._wrap_tensors(inputs, output, params, has_batch_dims)
        inputs_wrapped = [inp, wgt]
        if has_bias:
            bias = self._wrap_bias(inputs[2], params, has_batch_dims)
            inputs_wrapped = [inp, wgt, bias]

        gemm_layer = ComputationLayer(
            layer_name=self.node.name,
            layer_params=params,
            op_type=self.node.op_type,
            inputs=inputs_wrapped,
            output=outp,
            loop_dims=loop_dims,
        )
        return gemm_layer

    def _wrap_tensors(self, inputs: list[OnnxTensor], output: OnnxTensor,  params: GemmParameters, has_batch_dims: bool):
        input_tensor = inputs[0]
        weight_tensor = inputs[1]

        if not has_batch_dims:
            input_wrapper = TensorWrapper(
                input_tensor,
                [
                    ConcreteLoopIterator(LoopDim.N, params.n),
                    ConcreteLoopIterator(LoopDim.C, params.c),
                ],
                OperandKind.INPUT
            )

            weight_wrapper = TensorWrapper(
                weight_tensor,
                [
                    ConcreteLoopIterator(LoopDim.C, params.c),
                    ConcreteLoopIterator(LoopDim.M, params.m),
                ],
                OperandKind.WEIGHT
            )

            output_wrapper = TensorWrapper(
                output,
                [
                    ConcreteLoopIterator(LoopDim.N, params.n),
                    ConcreteLoopIterator(LoopDim.M, params.m),
                ],
                OperandKind.OUTPUT
            )
        else:
            input_wrapper = TensorWrapper(
                input_tensor,
                [
                    None,
                    ConcreteLoopIterator(LoopDim.B, params.b),
                    ConcreteLoopIterator(LoopDim.N, params.n),
                    ConcreteLoopIterator(LoopDim.C, params.c),
                ],
                OperandKind.INPUT
            )

            weight_wrapper = TensorWrapper(
                weight_tensor,
                [
                    None,
                    ConcreteLoopIterator(LoopDim.B, params.b),
                    ConcreteLoopIterator(LoopDim.C, params.c),
                    ConcreteLoopIterator(LoopDim.M, params.m),
                ],
                OperandKind.WEIGHT
            )

            output_wrapper = TensorWrapper(
                output,
                [
                    None,
                    ConcreteLoopIterator(LoopDim.B, params.b),
                    ConcreteLoopIterator(LoopDim.N, params.n),
                    ConcreteLoopIterator(LoopDim.M, params.m),
                ],
                OperandKind.OUTPUT
            )
        
        return input_wrapper, weight_wrapper, output_wrapper
    
    def _wrap_bias(self, bias: OnnxTensor, params: GemmParameters, has_batch_dims: bool):
        if not has_batch_dims:
            bias_wrapper = TensorWrapper(
                bias,
                [
                    None,
                    ConcreteLoopIterator(LoopDim.B, params.b),
                    ConcreteLoopIterator(LoopDim.N, params.n),
                    ConcreteLoopIterator(LoopDim.M, params.m),
                ],
                OperandKind.BIAS
            )
        else:
            bias_wrapper = TensorWrapper(
                bias,
                [
                    ConcreteLoopIterator(LoopDim.N, params.n),
                    ConcreteLoopIterator(LoopDim.M, params.m),
                ],
                OperandKind.BIAS
            )
        return bias_wrapper

