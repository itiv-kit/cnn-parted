from abc import ABC, abstractmethod

from framework.model.layer_tracker import LayerBindings
from framework.scheduling.schedules import SystemSchedule
from .context import MultiTenancyContext

class MultiTenancyStrategy(ABC):

    @abstractmethod
    def run(self, ctx: MultiTenancyContext) -> list[tuple[LayerBindings, SystemSchedule]]:
        pass
