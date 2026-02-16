from typing import NewType
from collections.abc import Callable
from dataclasses import dataclass

from framework.model.tensor_wrapper import TensorWrapper

NodeId = NewType("NodeId", int)
NocCoordinate = NewType("NocCoordinate", tuple[int, ...])
LinkId = NewType("LinkId", tuple[NocCoordinate, NocCoordinate])

#class NocCoordinate:
#    def __init__(self, *coords: tuple[int]):
#        self.coord = coords

@dataclass
class ComputeTask:
    network: str
    layer: str
    start_time: float
    end_time: float

@dataclass
class TransferTask:
    network: str
    tensor_name: str
    start_time: float
    end_time: float

@dataclass
class AllocatedTensor:
    tensor: TensorWrapper
    start_time: float
    end_time: float

@dataclass
class WorkloadInfo:
    accuracy_function: Callable
    input_shape: list[int]
    deadline: int
    frames: int
    onnx_filepath: str
