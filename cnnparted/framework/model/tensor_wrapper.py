from enum import Enum, auto

import numpy as np
from framework.model.tensor_like import TensorLike
from framework.model.onnx_tensor import OnnxTensor
from framework.model.tiling.iterators import ConcreteLoopIterator, LoopDim, LoopIterator

class OperandKind(Enum):
    INPUT = auto()
    WEIGHT = auto()
    BIAS = auto()
    OUTPUT = auto()

class TensorWrapper(TensorLike):
    def __init__(self, onnx_tensor: OnnxTensor, loop_dims: list[ConcreteLoopIterator], operand: OperandKind, sub_id: int = 0):
        super().__init__(name=onnx_tensor.name, kind=onnx_tensor.kind, _shape = onnx_tensor.shape)
        self._onnx_tensor = onnx_tensor
        self.loop_dims: list[ConcreteLoopIterator] = loop_dims
        self.operand = operand
        self.num_elems = np.prod(onnx_tensor.shape)
        self.sub_id = sub_id
        self._backing_tensor = np.zeros(onnx_tensor.shape)
    
    @property
    def loop_iterators(self) -> list[LoopIterator]:
        return [ldim.iterator for ldim in self.loop_dims]
    
    @property
    def shape(self) -> list[int]:
        return [ldim.extent for ldim in self.loop_dims]
    
    @shape.setter
    def shape(self, new_shape: list[int]) :
        self._shape = new_shape
        self._onnx_tensor.shape = new_shape
    
    def get_loop_dim(self, dim: LoopDim):
        return self.loop_dims[self.loop_iterators.index(dim)]
    
    def __contains__(self, key):
        if isinstance(key, ConcreteLoopIterator):
            return key in self.loop_dims
        elif isinstance(key, LoopDim):
            return key in [ldim.iterator for ldim in self.loop_dims]
        else:
            return False