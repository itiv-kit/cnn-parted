import random
import networkx as nx

from framework.noc.noc import NocTopology
from framework.model.layer_tracker import LayerBindings
from framework.helpers.config.stages_config import SchedulingPriority
from framework.model.layers.rename_layer import RenameLayer
from framework.model.layers.compute_layer import ComputationLayer
from framework.model.onnx_model import OnnxModel
from framework.node.node_evaluator import SystemResult, LayerResult
from .schedules import SystemSchedule
from framework.scheduling.utils import sort_ready_list, get_ready_layers, find_layer_info, SchedulingConstraints

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

    # While not all nodes have been assigned
    all_scheduled = False
    current_network = list(models.keys())[0]
    while not all_scheduled:

        # First we determine all networks of the currently selected network that we can currently schedule
        V_ready: list[tuple[str, ComputationLayer]] = []
        misc_layers: list[tuple[str, RenameLayer]] = []

        for search_layer in models[current_network].nodes:
            #if model.in_degree(layer) == 0:
            if scheduling_constraints[current_network][search_layer].in_degree == 0 and not scheduling_constraints[current_network][search_layer].is_scheduled:
                if isinstance(search_layer, ComputationLayer):
                    V_ready.append( (current_network, search_layer) )
                else:
                    misc_layers.append((current_network, search_layer))

        #print(V_ready)
        if len(V_ready) > 0:
            layer = V_ready[0][1]
            
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
                # Before we remove the node, set the earliest start for all successors
                for succ in models[current_network].successors(layer):
                    scheduling_constraints[current_network][succ].earliest_start = max(
                                                                    scheduling_constraints[current_network][succ].earliest_start, 
                                                                    scheduled_task.end_time
                                                                )
                    scheduling_constraints[current_network][succ].in_degree -= 1
                scheduling_constraints[current_network][layer].is_scheduled = True
            else:
                # For the current model we cannot schedule anything, apply depth-first
                # search among other networks. Herald does not mention how this 
                # rearrange is implemented, so we try a random other network
                current_network = random.choice(list(models.keys())) 

        # Handle all non-computation layers that are ready so that their 
        # successors can be scheduled in the next iteration
        for network, rename_layer in misc_layers:
            for succ in models[network].successors(rename_layer):
                # We assume here that misc layers take ZERO time to execute,
                # for that reason we can just propagate the earliest start information here
                scheduling_constraints[network][succ].earliest_start = max(
                                                                scheduling_constraints[network][succ].earliest_start, 
                                                                scheduling_constraints[network][rename_layer].earliest_start
                                                            )
                scheduling_constraints[network][succ].in_degree -= 1
            scheduling_constraints[network][rename_layer].is_scheduled = True
        
        if len(V_ready) == 0 and len(misc_layers) == 0:
            # cant do anything on current network
            current_network = random.choice(list(models.keys())) 


        all_scheduled = all([sched_constr.network_is_complete for sched_constr in scheduling_constraints.values()])

    # The original herald paper uses a "task swapping" after the initial schedulig. We don't do this here,
    # because exploiting earlier possible scheduling slots is already part of our scheduling
    return schedule
