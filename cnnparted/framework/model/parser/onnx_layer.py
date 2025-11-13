from abc import ABC, abstractmethod
from typing import Any

from onnx import AttributeProto, NodeProto

from framework.model.layers.compute_layer import ComputationLayer
from framework.model.layers.rename_layer import RenameLayer
from framework.model.onnx_tensor import OnnxTensor

class OnnxLayerParser(ABC):
    def __init__(self, node: NodeProto, in_shape: list[int], out_shape: list[int]):
        self.node = node
        self.in_shape = in_shape
        self.out_shape = out_shape

    @abstractmethod
    def parse(self, inputs: list[OnnxTensor], output: OnnxTensor) -> ComputationLayer | RenameLayer:
        ...

    def get_int_attr_by_name(self, name: str, attrs: Any, default: int | list[int] | None) -> list[int] | int:
        attr_names = [attr.name for attr in attrs]
        try: 
            idx = attr_names.index(name)
            target_attr = attrs[idx]
            match target_attr.type:
                case AttributeProto.AttributeType.INT:
                    return int(target_attr.i)
                case AttributeProto.AttributeType.INTS:
                    return list(target_attr.ints)
                case _:
                    raise RuntimeError("Desired attribute is not an int!")
        except:
            if default is not None:
                return default
            else:
                raise RuntimeError(f"attrs has no member of {name}: attr_names={attr_names}")