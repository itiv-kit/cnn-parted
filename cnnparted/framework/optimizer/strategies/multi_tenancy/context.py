from collections.abc import Callable
from typing import Protocol

from framework.helpers.config_helper import ConfigHelper
from framework.model.onnx_model import OnnxModel
from framework.noc.noc import NocTopology
from framework.model.layer_tracker import LayerBindings
from framework.scheduling import SystemSchedule
from framework.node.node_evaluator import SystemResult


class MultiTenancyStrategy(Protocol):
    def multi_tenancy_opt(self,
                            node_stats: SystemResult,
                            models: dict[str, OnnxModel],
                            noc: NocTopology,
                            conf: ConfigHelper,
                            initial_tracker: LayerBindings,
                            scheduling_algorithm: Callable[...,  SystemSchedule],
                          ) -> tuple[list[SystemSchedule], list[SystemSchedule]]:
        pass


class MultiTenancyContext:
    def __init__(self,
                strategy: MultiTenancyStrategy,
                node_stats: SystemResult, models: dict[str, OnnxModel], 
                noc: NocTopology, conf: ConfigHelper,
                initial_tracker: LayerBindings,
                scheduling_algorithm: Callable[..., SystemSchedule]):
        self.strategy = strategy
        self.node_stats = node_stats
        self.models = models
        self.noc = noc
        self.conf = conf
        self.initial_tracker = initial_tracker
        self.scheduling_algorithm =  scheduling_algorithm
    
    def run(self):
        solutions, invalid_solutions = self.strategy.multi_tenancy_opt(
            node_stats=self.node_stats,
            models=self.models,
            noc = self.noc,
            conf=self.conf,
            initial_tracker=self.initial_tracker,
            scheduling_algorithm=self.scheduling_algorithm,
        )
        return solutions, invalid_solutions
