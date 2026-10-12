from dataclasses import dataclass
from enum import Enum, auto

from framework.model.tensor_like import TensorLike, TensorKind

from onnx import TypeProto, TensorProto
from onnx.helper import tensor_dtype_to_np_dtype
import numpy as np

#class OnnxTensorKind(Enum):
#    INPUT = auto()
#    OUTPUT = auto()
#    HIDDEN = auto()
#    STATIC = auto()


@dataclass
class OnnxTensor(TensorLike):
    #name: str
    #shape: list[int]
    elem_type: np.dtype
    #kind: OnnxTensorKind

    @classmethod
    def from_tensor(cls, name: str, tensor: TypeProto.Tensor, kind: TensorKind):
        shape = [d.dim_value for d in tensor.shape.dim]
        elem_type = tensor_dtype_to_np_dtype(tensor.elem_type)
        return cls(name=name, _shape=shape, elem_type=elem_type, kind=kind)
    
    @classmethod
    def from_initializer(cls, name: str, init: TensorProto):
        shape = list(init.dims)
        elem_type = tensor_dtype_to_np_dtype(init.data_type)
        return cls(name=name, _shape=shape, elem_type=elem_type, kind=TensorKind.STATIC)
    
    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, new_shape: list[int]):
        self._shape = new_shape
    