from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

class TensorKind(Enum):
    INPUT = auto()
    OUTPUT = auto()
    HIDDEN = auto()
    STATIC = auto()


@dataclass
class TensorLike(ABC):
    name: str
    kind: TensorKind
    _shape: list[int]

    @property
    @abstractmethod
    def shape(self) -> list[int]:
        ...
    
    @shape.setter
    @abstractmethod
    def shape(self, new_shape: list[int]):
        ...

    @property
    def is_static(self):
        return self.kind == TensorKind.STATIC
    
    @property
    def is_input(self):
        return self.kind == TensorKind.INPUT

    @property
    def is_output(self):
        return self.kind == TensorKind.OUTPUT
    
    @property
    def is_hidden(self):
        return self.kind == TensorKind.HIDDEN
    
    @property
    def is_initially_available(self):
        return self.kind == TensorKind.INPUT \
                or self.kind == TensorKind.STATIC