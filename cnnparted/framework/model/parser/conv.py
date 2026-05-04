from math import ceil
import numpy as np

from .onnx_layer import OnnxLayerParser
from ..onnx_tensor import OnnxTensor
from ..layer_parameters import ConvParameters
from ..layers.compute_layer import ComputationLayer
from framework.model.tiling.iterators import ConcreteLoopIterator, LoopDim
from framework.model.tensor_wrapper import TensorWrapper, OperandKind

class ConvParser(OnnxLayerParser):
    def parse(self, inputs: list[OnnxTensor], output: OnnxTensor) -> ComputationLayer:
        attrs = self.node.attribute
        kernel_shape = self.get_int_attr_by_name("kernel_shape", attrs, default=None) #(kh, kw)
        strides = self.get_int_attr_by_name("strides", attrs, default=[1,1])
        dilations = self.get_int_attr_by_name("dilations", attrs, default=[1,1])
        groups = self.get_int_attr_by_name("groups", attrs, 1)
        pads = self.get_int_attr_by_name("pads", attrs, default=[0,0,0,0])

        ifms = np.prod(self.in_shape)
        ofms = np.prod(self.out_shape)
        weights = np.prod(kernel_shape)

        has_bias = len(inputs) == 3

        cfg = {
            'h': self.in_shape[2],
            'w': self.in_shape[3],
            'n': self.out_shape[0],
            'm': ceil(self.out_shape[1] / groups),
            'q': self.out_shape[2],
            'p': self.out_shape[3],
            'c': ceil(self.in_shape[1] / groups),
            's': kernel_shape[0],
            'r': kernel_shape[1],
            'wpad': pads[0],
            'hpad': pads[1],
            'wstride': strides[0],
            'hstride': strides[1],
            'ifms':ifms,
            'ofms':ofms,
            'weights':weights,
            'groups': groups,
            'wdilation': dilations[0],
            'hdilation': dilations[1],
        }
        params = ConvParameters(cfg)
        loop_dims = [
            ConcreteLoopIterator(LoopDim.N, params.n),
            ConcreteLoopIterator(LoopDim.M, params.m),
            ConcreteLoopIterator(LoopDim.P, params.p),
            ConcreteLoopIterator(LoopDim.Q, params.q),
            ConcreteLoopIterator(LoopDim.R, params.r),
            ConcreteLoopIterator(LoopDim.S, params.s),
            ConcreteLoopIterator(LoopDim.C, params.c),
        ]

        # Wrap tensors so that we can retain information about loop dimensions
        # Loop accesses for PyTorch Conv2d is specified as:
        #   in/out: (N, C, Q, P)
        #   weight: (M, C, S, R)
        inp, wgt, bias, outp = self._wrap_tensors(inputs, output, params)
        if has_bias:
            inputs_wrapped = [inp, wgt, bias]
        else:
            inputs_wrapped = [inp, wgt]

        conv_layer = ComputationLayer(
            layer_name=self.node.name,
            layer_params=params,
            op_type=self.node.op_type,
            inputs=inputs_wrapped,
            output=outp,
            loop_dims=loop_dims,
        )
        return conv_layer 
    
    def _wrap_tensors(self, inputs: list[OnnxTensor], output: OnnxTensor, params: ConvParameters) -> tuple[TensorWrapper, TensorWrapper, TensorWrapper|None, TensorWrapper]:
        input_tensor = inputs[0]
        kernel_tensor = inputs[1]
        input_wrapper = TensorWrapper(
            input_tensor,
            [
                ConcreteLoopIterator(LoopDim.N, params.n), 
                ConcreteLoopIterator(LoopDim.C, params.c),
                ConcreteLoopIterator(LoopDim.Q, params.h),
                ConcreteLoopIterator(LoopDim.P, params.w),
            ],
            OperandKind.INPUT
        )

        kernel_wrapper = TensorWrapper(
            kernel_tensor,
            [
                ConcreteLoopIterator(LoopDim.M, params.m),
                ConcreteLoopIterator(LoopDim.C, params.c),
                ConcreteLoopIterator(LoopDim.S, params.s),
                ConcreteLoopIterator(LoopDim.R, params.r),
            ],
            OperandKind.WEIGHT
        )
        
        # Check if there is a bias that needs to be wrapped
        bias_wrapper = None
        if len(inputs) == 3:
            bias_tensor = inputs[2]
            bias_wrapper = TensorWrapper(
                bias_tensor,
                [
                    ConcreteLoopIterator(LoopDim.M, params.m)
                ],
                OperandKind.BIAS
            )

        output_wrapper = TensorWrapper(
            output,
            [
                ConcreteLoopIterator(LoopDim.N, params.n), 
                ConcreteLoopIterator(LoopDim.C, params.c),
                ConcreteLoopIterator(LoopDim.Q, params.q),
                ConcreteLoopIterator(LoopDim.P, params.p),
            ],
            OperandKind.OUTPUT
        )

        return (input_wrapper, kernel_wrapper, bias_wrapper, output_wrapper)
