import copy

from framework.noc.noc import NocTopology
from framework.helpers.config.stages_config import SchedulingPriority
from framework.model.layers.rename_layer import RenameLayer
from framework.model.layers.compute_layer import ComputationLayer
from framework.model.onnx_model import OnnxModel
from framework.node.node_evaluator import SystemResult, LayerResult
from .schedules import SystemSchedule
from .utils import sort_ready_list, get_ready_layers, find_layer_info

def multi_queue_scheduling(models: dict[str, OnnxModel], node_stats: SystemResult,  noc: NocTopology,
                    priority: SchedulingPriority) -> SystemSchedule:
    """Modified version of list scheduing. Layers that have no more predecessors are assumed
    to be ready for scheduling and are added to V_ready. From there, these elements are
    written to a queue, where every accelerator has its own queue from which elements are scheduled.
    Compared to the basic list scheduling, this results in a more even utilization of accelerators"""

    type ReadyElement = tuple[str, LayerResult, ComputationLayer, float]

    schedule = SystemSchedule(node_stats.get_platform_ids(), list(models.keys()), noc )

    # Backup of models: scheduling modifies the model, so we need a backup
    # incase a model should be scheduled more than once. 
    model_backup = copy.deepcopy(models)
    model_endtimes = {nw_name: 0.0 for nw_name in models.keys()}
    completed_model_frames = {nw_name: 0 for nw_name in models.keys()}

    scheduled_layers = {nw_name: [] for nw_name in models.keys()}

    # While not all nodes have been assigned
    all_scheduled = False
    while not all_scheduled:
        #Determine all nodes that can be scheduled (all that have no predecessor)
        V_ready: list[ReadyElement] = []
        task_queues: dict[int, list[ReadyElement]] = {platform: [] for platform in node_stats.get_platform_ids()}
        misc_layers: list[tuple[str, RenameLayer]] = []

        V_ready, misc_layers = get_ready_layers(models, schedule, node_stats)

        # For all operations that are ready, assign them to the respective queues
        # Afterwards, sort the queues according to the scheduling algorithm
        for ready_layer in V_ready:
            network, _, layer, _, = ready_layer
            task_queues[layer.scheduling.binding].append(ready_layer)
        
        for task_queue in task_queues.values():
            task_queue = sort_ready_list(task_queue, schedule, models, priority)

        # Iterate over all queues and schedule one layers for each node
        # We schedule only one layer here to allow updating the priority functions
        for platform, queue in task_queues.items():
            if len(queue) == 0:
                continue

            network, eval_data, layer, introduced_delay = queue.pop(0)
            layer_result = find_layer_info(platform, network, layer.layer_name, node_stats)
            scheduled_layers[network].append(layer.layer_name)
            scheduled_task = schedule.schedule_layer(platform, network, layer_result, layer)
            model_endtimes[network] = max(model_endtimes[network], scheduled_task.end_time)

            # Before we remove the node, set the earliest start for all successors
            for succ in models[network].successors(layer):
                current_task = schedule.layers[platform][network][layer.layer_name]
                assert current_task == scheduled_task
                succ.scheduling.earliest_start = max(succ.scheduling.earliest_start, current_task.end_time)

            models[network].remove_layer(layer)
            queue.clear()
        
        for network, layer in misc_layers:
            for succ in models[network].successors(layer):
                # We assume here that misc layers take ZERO time to execute,
                # for that reason we can just propagate the earliest start information here
                succ.scheduling.earliest_start = layer.scheduling.earliest_start
            models[network].remove_layer(layer)
        
        # Logic to schedule a network multiple times
        # If all nodes are scheduled, and the netweork should be exectued multiple times,
        # copy it over which enables scheduling it repeatedly
        for network, model in models.items():
            if model.number_of_nodes() == 0:
                completed_model_frames[network] += 1
                # Check if the model has been scheduled enough times
                if completed_model_frames[network] < model.frames:
                    models[network] = copy.deepcopy(model_backup[network])
                    #models[network].frames = remaining_frames
                    for input_layer in models[network].get_input_layers():
                        # This is needed to reset the earliest start times of the input layers
                        input_layer.scheduling.earliest_start = model_endtimes[network] 
        
        num_remaining_nodes = sum([graph.number_of_nodes() for graph in models.values()])
        all_scheduled = (num_remaining_nodes == 0)

    return schedule
