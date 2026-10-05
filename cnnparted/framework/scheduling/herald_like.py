import heapq
import random

from framework.noc.noc import NocTopology
from framework.model.layers.rename_layer import RenameLayer
from framework.model.layers.compute_layer import ComputationLayer
from framework.model.onnx_model import OnnxModel
from framework.node.node_evaluator import SystemResult
from .schedules import SystemSchedule
from framework.scheduling.utils import SchedulingConstraints, release_misc_layer

def herald_like(models: dict[str, OnnxModel], node_stats: SystemResult,
                    noc: NocTopology, lbf: float) -> SystemSchedule:
    """This implements a version of the Herald Scheduling and Binding algorith proposed
    by Kwon et al. in Heterogeneous Dataflow Accelerators"""

    schedule = SystemSchedule(node_stats.get_platform_ids(), list(models.keys()), noc )


    network_info = {nw_name: {} for nw_name in models.keys()}

    # Initialize a lookup to decide when a layer is ready to be scheduled
    scheduling_constraints: dict[str, SchedulingConstraints] = {nw: SchedulingConstraints(model) for nw, model in models.items()}

    model_endtimes = {nw_name: 0.0 for nw_name in models.keys()}
    completed_model_frames = {nw_name: 0 for nw_name in models.keys()}

    # Ready ComputationLayers of each network, kept as a min-heap keyed by the layer's
    # construction order (layer.id) - this reproduces "take the first ready layer found while
    # scanning model.nodes" without re-scanning that network's layers every time it is picked
    # as current_network below.
    ready_heaps: dict[str, list[tuple[int, ComputationLayer]]] = {nw: [] for nw in models.keys()}

    def on_ready(network: str, layer: ComputationLayer | RenameLayer):
        """Called exactly once, the moment `layer`'s in_degree reaches zero."""
        if isinstance(layer, ComputationLayer):
            heapq.heappush(ready_heaps[network], (layer.id, layer))
        else:
            release_misc_layer(models, scheduling_constraints, network, layer, on_ready)

    # Seed the heaps once with every layer that starts out without predecessors, instead of
    # rescanning a network's layers from scratch every time it is picked as current_network
    initially_ready = [(network, layer) for network, model in models.items() for layer in model.nodes
                       if scheduling_constraints[network][layer].in_degree == 0]
    for network, layer in initially_ready:
        on_ready(network, layer)

    # While not all nodes have been assigned
    all_scheduled = False
    current_network = list(models.keys())[0]
    while not all_scheduled:
        heap = ready_heaps[current_network]

        if heap:
            # Peek only: the layer stays on the heap unless it is actually scheduled below
            layer = heap[0][1]

            layer_on_platforms = node_stats.get_layer_on_all_platforms(current_network, layer.layer_name)
            edp = [layer_result.edp for id, layer_result in layer_on_platforms]
            min_edp_idx = edp.index(min(edp))
            best_fit_acc_id, best_fit_acc_layer_result = layer_on_platforms[min_edp_idx]

            load_balance_cond = schedule.system_latency \
                                < lbf * (schedule.node_schedules[best_fit_acc_id].cur_time + best_fit_acc_layer_result.latency)

            layer_was_scheduled = False
            current_frame = completed_model_frames[current_network]
            earliest_start = scheduling_constraints[current_network][layer].earliest_start

            if load_balance_cond:
                # Schedule layer on the best fitting accelerator
                scheduled_task = schedule.schedule_layer(best_fit_acc_id, current_network, current_frame, earliest_start, best_fit_acc_layer_result, layer)
                print("Schedule on pref")
                layer_was_scheduled = True

            else:
                # Try to schedule the layer on other accelerators
                layer_on_platforms.pop(min_edp_idx)
                for id, layer_result in layer_on_platforms:
                    load_balance_cond = schedule.system_latency \
                                        < lbf * (schedule.node_schedules[id].cur_time + layer_result.latency)
                    if load_balance_cond:
                        scheduled_task = schedule.schedule_layer(best_fit_acc_id, current_network, current_frame, earliest_start, best_fit_acc_layer_result, layer)
                        layer_was_scheduled = True

            if layer_was_scheduled:
                heapq.heappop(heap)

                # Before we remove the node, set the earliest start for all successors
                for succ in models[current_network].successors(layer):
                    succ_constr = scheduling_constraints[current_network][succ]
                    succ_constr.earliest_start = max(succ_constr.earliest_start, scheduled_task.end_time)
                    succ_constr.in_degree -= 1
                    if succ_constr.in_degree == 0:
                        on_ready(current_network, succ)
                scheduling_constraints[current_network][layer].is_scheduled = True
            else:
                # For the current model we cannot schedule anything, apply depth-first
                # search among other networks. Herald does not mention how this
                # rearrange is implemented, so we try a random other network
                current_network = random.choice(list(models.keys()))

        else:
            # cant do anything on current network
            current_network = random.choice(list(models.keys()))

        all_scheduled = all([sched_constr.network_is_complete for sched_constr in scheduling_constraints.values()])

    # The original herald paper uses a "task swapping" after the initial schedulig. We don't do this here,
    # because exploiting earlier possible scheduling slots is already part of our scheduling
    return schedule
