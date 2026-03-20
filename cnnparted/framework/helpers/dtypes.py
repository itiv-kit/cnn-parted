from typing import NewType
from collections.abc import Callable
from dataclasses import dataclass, field

from framework.model.tensor_wrapper import TensorWrapper
from framework.helpers.config.general_config import WorkloadSpec

NodeId = NewType("NodeId", int)
NocCoordinate = NewType("NocCoordinate", tuple[int, ...])
LinkId = NewType("LinkId", tuple[NocCoordinate, NocCoordinate])

#class NocCoordinate:
#    def __init__(self, *coords: tuple[int]):
#        self.coord = coords


@dataclass
class TransferTask:
    network: str
    tensor_name: str
    start_time: float
    end_time: float
    src_node: NocCoordinate
    dest_node: NocCoordinate
    bandwidth: int

@dataclass
class ComputeTask:
    network: str
    layer: str
    binding: int
    start_time: float
    end_time: float
    loads: list[TransferTask] = field(default_factory=lambda: [])
    stores: list[TransferTask] = field(default_factory=lambda: [])

    def to_dict(self) -> dict:
        return {
            self.layer: {
                "binding": self.binding,
                "start_time": self.start_time,
                "end_time": self.end_time,          
            }
        }

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
    deadline_div: float
    deadline_mul: float
    frames: int
    onnx_filepath: str

    def __init__(self, accuracy_function: Callable, spec: WorkloadSpec, onnx_filepath: str):
        self.accuracy_function = accuracy_function
        self.input_shape = spec.input_size
        self.onnx_filepath = onnx_filepath
        self.deadline = spec.deadline
        self.deadline_div = spec.deadline_div
        self.deadline_mul = spec.deadline_mul
        self.frames = spec.frames
