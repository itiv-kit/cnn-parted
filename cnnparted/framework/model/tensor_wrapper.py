from enum import Enum, auto

import numpy as np
from framework.model.onnx_tensor import OnnxTensor
from framework.model.tiling.iterators import ConcreteLoopIterator, LoopDim, LoopIterator

class OperandKind(Enum):
    INPUT = auto()
    WEIGHT = auto()
    BIAS = auto()
    OUTPUT = auto()

class TensorWrapper(OnnxTensor):
    def __init__(self, onnx_tensor: OnnxTensor, loop_dims: list[ConcreteLoopIterator], operand: OperandKind, sub_id: int | None = None):
        super().__init__(name=onnx_tensor.name, elem_type_map=onnx_tensor.elem_type_map, kind=onnx_tensor.kind, shape=onnx_tensor.shape)
        self._onnx_tensor = onnx_tensor
        self.loop_dims: list[ConcreteLoopIterator] = loop_dims
        self.operand = operand
        self.num_elems = np.prod(onnx_tensor.shape)
        self.sub_id = sub_id
    
    @property
    def loop_iterators(self) -> list[LoopIterator]:
        return [ldim.iterator for ldim in self.loop_dims]
    
    def __contains__(self, key):
        if isinstance(key, ConcreteLoopIterator):
            return key in self.loop_dims
        elif isinstance(key, LoopDim):
            return key in [ldim.iterator for ldim in self.loop_dims]
        else:
            return False