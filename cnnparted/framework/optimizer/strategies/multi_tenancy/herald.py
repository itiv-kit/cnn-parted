from ._strategy import MultiTenancyStrategy
from .context import MultiTenancyContext

from framework.model.layer_tracker import LayerBindings
from framework.scheduling import herald_like

class HeraldMultiTenancy(MultiTenancyStrategy):

    def run(self, ctx: MultiTenancyContext):
        system_schedule = herald_like(ctx.models, ctx.node_stats, ctx.noc, ctx.conf.mt.optimizer.lbf)

        # Herald combines binding and scheduling, read out values from the schedule
        network_info = {nw_name: {} for nw_name in ctx.models.keys()}
        for id, node_schedule in system_schedule.node_schedules.items():
            for task in node_schedule.tasks:
                network_info[task.network][task.layer] = id
        bindings = LayerBindings(network_info)

        # Herald only finds one solution
        return [(bindings, system_schedule)]

