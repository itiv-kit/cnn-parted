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
class LinkEvent:
    network: str
    tensor_name: str
    link_start_node: int
    link_end_node: int
    start_time: float
    end_time: float
    bandwidth: float
    link_energy: float

    @property
    def link_latency(self):
        return self.end_time - self.start_time

@dataclass
class TransferTask:
    network: str
    tensor_name: str
    start_time: float
    end_time: float
    src_node: NocCoordinate
    dest_node: NocCoordinate
    bandwidth: int
    events: list[LinkEvent]

    @property
    def transfer_latency(self):
        return sum([ev.link_latency for ev in self.events])
    
    @property
    def transfer_energy(self):
        return sum([ev.link_energy for ev in self.events])

@dataclass
class ComputeTask:
    network: str
    frame: int
    layer: str
    binding: int
    start_time: float
    end_time: float
    computation_latency: float
    computation_energy: float
    loads: list[TransferTask] = field(default_factory=lambda: [])
    stores: list[TransferTask] = field(default_factory=lambda: [])

    def to_dict(self) -> dict:
        return {
            self.layer: {
                "frame": self.frame,
                "binding": self.binding,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "computation_latency": self.computation_latency,
                "computation_energy": self.computation_energy,
                "total_latency": self.total_latency,
                "total_energy": self.total_energy
            }
        }
    
    @property
    def communication_latency(self):
        return sum([ld.transfer_latency for ld in self.loads] + [st.transfer_latency for st in self.stores])

    @property
    def communication_energy(self):
        return sum([ld.transfer_energy for ld in self.loads] + [st.transfer_energy for st in self.stores])
    
    @property
    def total_latency(self):
        return self.computation_latency + self.communication_latency
    
    @property
    def total_energy(self):
        return self.computation_energy + self.communication_energy

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
    frames: int | None
    onnx_filepath: str
    identifier: str

    def __init__(self, accuracy_function: Callable, spec: WorkloadSpec, onnx_filepath: str):
        self.accuracy_function = accuracy_function
        self.input_shape = spec.input_size
        self.onnx_filepath = onnx_filepath
        self.deadline = spec.deadline
        self.deadline_div = spec.deadline_div
        self.deadline_mul = spec.deadline_mul
        self.frames = spec.frames
