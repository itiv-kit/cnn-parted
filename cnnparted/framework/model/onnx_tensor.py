from dataclasses import dataclass
from enum import Enum, auto

from onnx import TypeProto, TensorProto
from onnx.mapping import TENSOR_TYPE_MAP, TensorDtypeMap

class OnnxTensorKind(Enum):
    INPUT = auto()
    OUTPUT = auto()
    HIDDEN = auto()
    STATIC = auto()


@dataclass
class OnnxTensor:
    name: str
    shape: list[int]
    elem_type_map: TensorDtypeMap
    kind: OnnxTensorKind

    @classmethod
    def from_tensor(cls, name: str, tensor: TypeProto.Tensor, kind: OnnxTensorKind):
        shape = [d.dim_value for d in tensor.shape.dim]
        elem_type = TENSOR_TYPE_MAP[tensor.elem_type]
        return cls(name, shape, elem_type, kind)
    
    @classmethod
    def from_initializer(cls, name: str, init: TensorProto):
        shape = list(init.dims)
        elem_type = init.data_type
        return cls(name, shape, elem_type, OnnxTensorKind.STATIC)
    
    @property
    def is_static(self):
        return self.kind == OnnxTensorKind.STATIC
    
    @property
    def is_input(self):
        return self.kind == OnnxTensorKind.INPUT

    @property
    def is_output(self):
        return self.kind == OnnxTensorKind.OUTPUT
    
    @property
    def is_hidden(self):
        return self.kind == OnnxTensorKind.HIDDEN
    
    @property
    def is_initially_available(self):
        return self.kind == OnnxTensorKind.INPUT \
                or self.kind == OnnxTensorKind.STATIC
    