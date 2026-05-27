from typing import Callable

from framework.model.layer_tracker import LayerBindings
from framework.scheduling import herald_like
from framework.helpers.config_helper import ConfigHelper
from framework.model.onnx_model import OnnxModel
from framework.noc.noc import NocTopology
from framework.scheduling import SystemSchedule
from framework.node.node_evaluator import SystemResult

class HeraldMultiTenancy:

    def multi_tenancy_opt(self,
                            node_stats: SystemResult,
                            models: dict[str, OnnxModel],
                            noc: NocTopology,
                            conf: ConfigHelper,
                            initial_tracker: LayerBindings,
                            scheduling_algorithm: Callable[..., SystemSchedule],
                          ) -> tuple[list[SystemSchedule], list[SystemSchedule]]:

        system_schedule = herald_like(models, node_stats, noc, conf.mt.optimizer.lbf)

        # Herald combines binding and scheduling, read out values from the schedule
        network_info = {nw_name: {} for nw_name in models.keys()}
        for id, node_schedule in system_schedule.node_schedules.items():
            for task in node_schedule.tasks:
                network_info[task.network][task.layer] = id
        bindings = LayerBindings(network_info)

        # Herald only finds one solution
        return [(bindings, system_schedule)]

