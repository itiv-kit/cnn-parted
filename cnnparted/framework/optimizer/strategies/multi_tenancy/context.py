from collections.abc import Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from framework.helpers.config_helper import ConfigHelper
from framework.helpers.config.stages_config import SchedulingPriority
from framework.model.onnx_model import OnnxModel
from framework.noc.noc import NocTopology
from framework.model.layer_tracker import LayerBindings
from framework.scheduling import SystemSchedule
from framework.node.node_evaluator import SystemResult

from .herald import HeraldMultiTenancy
from .simulated_annealing import SaMultiTenancy
from .optuna import OptunaMultiTenancy



class MultiTenancyContext:
    def __init__(self, node_stats: SystemResult, models: dict[str, OnnxModel], 
                 noc: NocTopology, conf: ConfigHelper,
                 initial_tracker: LayerBindings,
                 scheduling_algorithm: Callable[ [dict[str, OnnxModel], LayerBindings, SystemResult, NocTopology, SchedulingPriority], SystemSchedule]):
        self.node_stats = node_stats
        self.models = models
        self.noc = noc
        self.conf = conf
        self.initial_tracker = initial_tracker
        self.scheduling_algorithm =  scheduling_algorithm
