import copy

from framework.noc.noc import NocTopology
from framework.model.layer_tracker import LayerBindings
from framework.helpers.config.stages_config import SchedulingPriority
from framework.model.layers.rename_layer import RenameLayer
from framework.model.layers.compute_layer import ComputationLayer
from framework.model.onnx_model import OnnxModel
from framework.node.node_evaluator import SystemResult, LayerResult
from .schedules import SystemSchedule
from .utils import sort_ready_list, get_ready_layers, find_layer_info, SchedulingConstraints

def multi_queue_scheduling(models: dict[str, OnnxModel], bindings: LayerBindings, node_stats: SystemResult,  
                           noc: NocTopology, priority: SchedulingPriority) -> SystemSchedule:
    """Modified version of list scheduing. Layers that have no more predecessors are assumed
    to be ready for scheduling and are added to V_ready. From there, these elements are
    written to a queue, where every accelerator has its own queue from which elements are scheduled.
    Compared to the basic list scheduling, this results in a more even utilization of accelerators"""

    type ReadyElement = tuple[str, LayerResult, ComputationLayer, float]

    schedule = SystemSchedule(node_stats.get_platform_ids(), list(models.keys()), noc )

    # Initialize a lookup to decide when a layer is ready to be scheduled
    scheduling_constraints: dict[str, SchedulingConstraints] = {nw: SchedulingConstraints(model) for nw, model in models.items()}

    model_endtimes = {nw_name: 0.0 for nw_name in models.keys()}
    completed_model_frames = {nw_name: 0 for nw_name in models.keys()}

    # While not all nodes have been assigned
    all_scheduled = False
    while not all_scheduled:
        #Determine all nodes that can be scheduled (all that have no predecessor)
        task_queues: dict[int, list[ReadyElement]] = {platform: [] for platform in node_stats.get_platform_ids()}

        V_ready, misc_layers = get_ready_layers(models, bindings, schedule, node_stats, scheduling_constraints)

        # For all operations that are ready, assign them to the respective queues
        # Afterwards, sort the queues according to the priority function
        for ready_layer in V_ready:
            network, _, layer, _, = ready_layer
            binding = bindings.get_binding(network, layer.layer_name)
            task_queues[binding].append(ready_layer)
        
        for task_queue in task_queues.values():
            task_queue = sort_ready_list(task_queue, bindings, schedule, scheduling_constraints, models, priority)

        # Iterate over all queues and schedule one layers for each node
        # We schedule only one layer here to allow updating the priority functions
        for platform, queue in task_queues.items():
            if len(queue) == 0:
                continue

            network, eval_data, layer, introduced_delay = queue.pop(0)
            current_frame = completed_model_frames[network]
            layer_result = find_layer_info(platform, network, layer.layer_name, node_stats)

            earliest_start = scheduling_constraints[network][layer].earliest_start
            scheduled_task = schedule.schedule_layer(platform, network, current_frame, earliest_start, layer_result, layer)
            model_endtimes[network] = max(model_endtimes[network], scheduled_task.end_time)

            # Before we remove the node, set the earliest start for all successors
            for succ in models[network].successors(layer):
                scheduling_constraints[network][succ].earliest_start = max(
                                                                scheduling_constraints[network][succ].earliest_start, 
                                                                scheduled_task.end_time
                                                            )
                scheduling_constraints[network][succ].in_degree -= 1
                #succ.scheduling.earliest_start = max(succ.scheduling.earliest_start, scheduled_task.end_time)

            scheduling_constraints[network][layer].is_scheduled = True
            #models[network].remove_layer(layer)
        
        for network, rename_layer in misc_layers:
            for succ in models[network].successors(rename_layer):
                # We assume here that misc layers take ZERO time to execute,
                # for that reason we can just propagate the earliest start information here
                scheduling_constraints[network][succ].earliest_start = max(
                                                                scheduling_constraints[network][succ].earliest_start, 
                                                                scheduling_constraints[network][rename_layer].earliest_start
                                                            )
                scheduling_constraints[network][succ].in_degree -= 1
                #succ.scheduling.earliest_start = max(succ.scheduling.earliest_start, rename_layer.scheduling.earliest_start)
            scheduling_constraints[network][rename_layer].is_scheduled = True
            #models[network].remove_layer(rename_layer)
        
        # Logic to schedule a network multiple times
        # If all nodes are scheduled, and the netweork should be exectued multiple times,
        # copy it over which enables scheduling it repeatedly
        for network, model in models.items():
            if scheduling_constraints[network].network_is_complete:
                # If not all iterations of this model have been scheduled incement the counter
                if completed_model_frames[network] < model.frames:
                    completed_model_frames[network] += 1
                    schedule.network_finish_times[network].append(model_endtimes[network])

                    # If, after the previous increment, we still have more iterations to schedule,
                    # copy the model over to schedule it once more
                    if completed_model_frames[network] < model.frames:
                        scheduling_constraints[network] = SchedulingConstraints(model)
                        for input_layer in models[network].get_input_layers():
                            # This is needed to reset the earliest start times of the input layers
                            scheduling_constraints[network][input_layer].earliest_start = model_endtimes[network]
                            #input_layer.scheduling.earliest_start = model_endtimes[network] 
        
        #num_remaining_nodes = sum([graph.number_of_nodes() for graph in models.values()])
        #all_scheduled = (num_remaining_nodes == 0)
        all_scheduled = all([sched_constr.network_is_complete for sched_constr in scheduling_constraints.values()])

    return schedule
