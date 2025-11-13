from math import ceil
import numpy as np

from .onnx_layer import OnnxLayerParser
from ..onnx_tensor import OnnxTensor
from ..layer_parameters import ConvParameters
from ..layers.compute_layer import ComputationLayer

class ConvParser(OnnxLayerParser):
    def parse(self, inputs: list[OnnxTensor], output: OnnxTensor) -> ComputationLayer:
        attrs = self.node.attribute
        kernel_shape = self.get_int_attr_by_name("kernel_shape", attrs, default=None)
        strides = self.get_int_attr_by_name("strides", attrs, default=[1,1])
        dilations = self.get_int_attr_by_name("strides", attrs, default=[1,1])
        groups = self.get_int_attr_by_name("groups", attrs, 1)
        pads = self.get_int_attr_by_name("pads", attrs, default=[0,0,0,0])

        ifms = np.prod(self.in_shape)
        ofms = np.prod(self.out_shape)
        weights = np.prod(kernel_shape)

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
            'hstride': [1],
            'ifms':ifms,
            'ofms':ofms,
            'weights':weights,
            'groups': groups,
            'dilations': dilations
        }
        params = ConvParameters(cfg)
        conv_layer = ComputationLayer(
            layer_name=self.node.name,
            layer_params=params,
            op_type=self.node.op_type,
            inputs=inputs,
            output=output

        )
        return conv_layer 