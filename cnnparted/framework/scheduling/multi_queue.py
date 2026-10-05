from framework.noc.noc import NocTopology
from framework.model.layer_tracker import LayerBindings
from framework.helpers.config.stages_config import SchedulingPriority
from framework.model.layers.rename_layer import RenameLayer
from framework.model.layers.compute_layer import ComputationLayer
from framework.model.onnx_model import OnnxModel
from framework.node.node_evaluator import SystemResult
from .schedules import SystemSchedule
from .utils import find_layer_info, SchedulingConstraints, SchedulingDeadlines, ReadyQueue, release_misc_layer

def multi_queue_scheduling(models: dict[str, OnnxModel], bindings: LayerBindings, node_stats: SystemResult,
                           noc: NocTopology, priority: SchedulingPriority) -> SystemSchedule:
    """Modified version of list scheduing. Layers that have no more predecessors are assumed
    to be ready for scheduling and are added to V_ready. From there, these elements are
    written to a queue, where every accelerator has its own queue from which elements are scheduled.
    Compared to the basic list scheduling, this results in a more even utilization of accelerators"""

    schedule = SystemSchedule(node_stats.get_platform_ids(), list(models.keys()), noc )
    model_deadlines = {nw: SchedulingDeadlines(model.deadlines) for nw, model in models.items()}

    # Initialize a lookup to decide when a layer is ready to be scheduled
    scheduling_constraints: dict[str, SchedulingConstraints] = {nw: SchedulingConstraints(model) for nw, model in models.items()}

    model_endtimes = {nw_name: 0.0 for nw_name in models.keys()}
    completed_model_frames = {nw_name: 0 for nw_name in models.keys()}

    # Every platform gets its own pair of ready pools (deadline-constrained / best-effort), so
    # that each platform's next layer can be popped directly instead of rescanning and
    # re-sorting every ready layer into per-platform queues on every iteration.
    platform_ids = node_stats.get_platform_ids()
    constrained_pools = {platform: ReadyQueue(priority, bindings, schedule) for platform in platform_ids}
    unconstrained_pools = {platform: ReadyQueue(priority, bindings, schedule) for platform in platform_ids}
    num_incomplete_constrained = sum(1 for model in models.values() if model.has_constraints)

    def pool_for(network: str, platform: int) -> ReadyQueue:
        pools = constrained_pools if models[network].has_constraints else unconstrained_pools
        return pools[platform]

    def on_ready(network: str, layer: ComputationLayer | RenameLayer):
        """Called exactly once, the moment `layer`'s in_degree reaches zero."""
        if isinstance(layer, ComputationLayer):
            binding = bindings.get_binding(network, layer.layer_name)
            eval_data = find_layer_info(binding, network, layer.layer_name, node_stats)
            pool_for(network, binding).push(network, layer, eval_data, scheduling_constraints[network], model_deadlines)
        else:
            release_misc_layer(models, scheduling_constraints, network, layer, on_ready)

    # Seed the pools once with every layer that starts out without predecessors, instead of
    # rescanning all layers on every iteration of the loop below to find them
    initially_ready = [(network, layer) for network, model in models.items() for layer in model.nodes
                       if scheduling_constraints[network][layer].in_degree == 0]
    for network, layer in initially_ready:
        on_ready(network, layer)

    # While not all nodes have been assigned
    all_scheduled = False
    while not all_scheduled:
        # Schedule at most one layer per platform, picked from whichever pool the deadline
        # gate currently allows
        pools = constrained_pools if num_incomplete_constrained > 0 else unconstrained_pools
        for platform in platform_ids:
            candidate = pools[platform].pop()
            if candidate is None:
                continue

            network, eval_data, layer = candidate
            current_frame = completed_model_frames[network]
            earliest_start = scheduling_constraints[network][layer].earliest_start

            scheduled_task = schedule.schedule_layer(platform, network, current_frame, earliest_start, eval_data, layer)
            model_endtimes[network] = max(model_endtimes[network], scheduled_task.end_time)

            # Before we remove the node, set the earliest start for all successors
            for succ in models[network].successors(layer):
                succ_constr = scheduling_constraints[network][succ]
                succ_constr.earliest_start = max(succ_constr.earliest_start, scheduled_task.end_time)
                succ_constr.in_degree -= 1
                if succ_constr.in_degree == 0:
                    on_ready(network, succ)

            scheduling_constraints[network][layer].is_scheduled = True

        # Logic to schedule a network multiple times
        # If all nodes are scheduled, and the netweork should be exectued multiple times,
        # reset the SchedulingConstraints to re-start scheduling of it
        for network, model in models.items():
            if scheduling_constraints[network].network_is_complete:
                # If not all iterations of this model have been scheduled incement the counter
                if not model_deadlines[network].all_frames_are_done:
                    completed_model_frames[network] += 1
                    model_deadlines[network].mark_frame_complete()
                    schedule.network_finish_times[network].append(model_endtimes[network])
                    if model.has_constraints:
                        num_incomplete_constrained -= 1

                    # If, after the previous increment, we still have more iterations to schedule,
                    # copy the model over to schedule it once more
                    if not model_deadlines[network].all_frames_are_done:
                        scheduling_constraints[network] = SchedulingConstraints(model)
                        if model.has_constraints:
                            num_incomplete_constrained += 1
                        for input_layer in models[network].get_input_layers():
                            # This is needed to reset the earliest start times of the input layers
                            scheduling_constraints[network][input_layer].earliest_start = model_deadlines[network].previous_deadline
                            on_ready(network, input_layer)

        # Check if the scheduling is finished
        all_scheduled = all([sched_constr.network_is_complete for sched_constr in scheduling_constraints.values()])

    return schedule
