from math import ceil
import numpy as np

from .onnx_layer import OnnxLayerParser
from ..onnx_tensor import OnnxTensor
from ..layer_parameters import ConvParameters
from ..layers.compute_layer import ComputationLayer
from framework.model.tiling.iterators import ConcreteLoopIterator, LoopDim
from framework.model.tensor_wrapper import TensorWrapper, OperandKind

class ConvTransposeParser(OnnxLayerParser):
    def parse(self, inputs: list[OnnxTensor], output: OnnxTensor) -> ComputationLayer:
        attrs = self.node.attribute
        kernel_shape = self.get_int_attr_by_name("kernel_shape", attrs, default=None) #(kh, kw)
        strides = self.get_int_attr_by_name("strides", attrs, default=[1,1]) #(sh, sw)
        dilations = self.get_int_attr_by_name("dilations", attrs, default=[1,1]) #(dh, dw)
        groups = self.get_int_attr_by_name("groups", attrs, 1)
        pads = self.get_int_attr_by_name("pads", attrs, default=[0,0,0,0])

        if isinstance(pads, list):
            h_begin, w_begin, h_end, w_end = pads
        else:
            h_begin, w_begin, h_end, w_end = [pads]*4
        
        if isinstance(strides, list):
            sh, sw = strides
        else:
            sh, sw = [strides]*2

        # Model the execution of ConvTranspose as a Conv.
        # For this we need:
        #   - resolve strides by inserting zeros between elements in the input
        #   - pad the upsampled input with zeros 
        # This is pretty inefficient but works for every accelerator
        # see also: A guide to convolution arithmetic for deep learning (https://arxiv.org/pdf/1603.07285v1)

        old_in_height = self.in_shape[2]
        a_h = (old_in_height + (h_begin+h_end) - kernel_shape[0]) % sh
        new_in_height = old_in_height + ((sh-1)*(old_in_height-1)) #insert zeros to resolve stride and upsample input
        new_in_height = new_in_height + a_h
        pad_h = (kernel_shape[0]-1) - (h_begin+h_end)

        old_in_width = self.in_shape[3]
        a_w = (old_in_width + (w_begin+w_end) - kernel_shape[1]) % sw
        new_in_width = old_in_width + ((sw-1)*(old_in_width-1)) #insert zeros to resolve stride and upsample input
        new_in_width = new_in_width + a_w
        pad_w = (kernel_shape[1]-1) - (w_begin+w_end)

        strides = [1, 1]
        self.in_shape[2] = new_in_height
        self.in_shape[3] = new_in_width

        # Now we can compute the conv2d parameters as before
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
            'wpad': pad_w,
            'hpad': pad_h,
            'wstride': strides[0],
            'hstride': strides[1],
            'ifms':ifms,
            'ofms':ofms,
            'weights':weights,
            'groups': groups,
            'dilations': dilations
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

