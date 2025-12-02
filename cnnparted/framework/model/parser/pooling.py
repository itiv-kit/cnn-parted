from .onnx_layer import OnnxLayerParser
from ..layer_parameters import PoolingParameters 
from framework.model.onnx_tensor import OnnxTensor
from framework.model.layers.rename_layer import RenameLayer

class PoolingParser(OnnxLayerParser): 
    def parse(self, inputs: list[OnnxTensor], output: OnnxTensor) -> RenameLayer:
        attrs = self.node.attribute
        strides = self.get_int_attr_by_name("strides", attrs, default=[1,1])
        dilations = self.get_int_attr_by_name("strides", attrs, default=[1,1])
        pads = self.get_int_attr_by_name("pads", attrs, default=[0,0,0,0])

        if self.node.op_type in ["MaxPool", "AveragePool"]:
            kernel_shape = self.get_int_attr_by_name("kernel_shape", attrs, default=None)
        elif self.node.op_type in ["GlobalMaxPool", "GlobalAveragePool"]:
            # Perform pooling over the full input shape, i.e. each channel
            # is reduced to a 1x1 pixel -> kernel_shape = [in_heigh, in_width]
            assert len(self.in_shape) == 4, "Expected 4D input for global pooling ops"
            kernel_shape = [self.in_shape[2], self.in_shape[3]]
        else:
            raise RuntimeError(f"Unexpected pooling operation of type: {self.node.op_type}")
        
        cfg = {
            'kernel': kernel_shape,
            'padding': pads,
            'stride': strides,
            'dilations': dilations
        }
        params = PoolingParameters(cfg)
        pool_layer = RenameLayer(
            layer_name=self.node.name,
            layer_params=params,
            op_type=self.node.op_type,
            inputs=inputs,
            output=output
        )
        return pool_layer
