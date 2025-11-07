from typing import NewType
from dataclasses import dataclass

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