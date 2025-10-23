from typing import NewType
from dataclasses import dataclass

NodeId = NewType("NodeId", int)
NocCoordinate = NewType("NocCoordinate", tuple[int, ...])

@dataclass
class ComputeTask:
    network: str
    layer: str
    start_time: float
    end_time: float

@dataclass
class TransferTask:
    start_time: float
    end_time: float