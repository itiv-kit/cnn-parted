from abc import ABC, abstractmethod

class LinkModelInterface(ABC):

    @abstractmethod
    def get_latency_ms(self, slice_size : int, frame_rate : int = 0) -> float:
        ...

    @abstractmethod
    def get_pow_cons_mW(self, slice_size : int, frame_rate : int = 0) -> float:
        ...
