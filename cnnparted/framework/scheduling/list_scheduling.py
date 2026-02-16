import copy

from framework.helpers.config.stages_config import SchedulingPriority
from framework.model.layers.rename_layer import RenameLayer
from framework.model.layers.compute_layer import ComputationLayer
from framework.model.onnx_model import OnnxModel
from framework.node.node_evaluator import SystemResult, LayerResult
from .utils import SystemSchedule

def list_scheduling(models: dict[str, OnnxModel], node_stats: SystemResult, 
                    priority: SchedulingPriority) -> SystemSchedule:
    """Used for MultiTenancy partitioning. After the layers have been assigned to cores,
    we must determine a start time for each layer"""

    # Define helper functions to find layers
    def find_layer_info(id: int, nw_name: str, layer_name: str):
        network = node_stats[id].designs[0].networks[nw_name]
        if layer_name in network:
            return network[layer_name]
        raise RuntimeError(f"Could not find layer {layer_name} in network {nw_name}")

    schedule = SystemSchedule(node_stats.get_platform_ids(), list(models.keys()) )

    # Backup of models: scheduling modifies the model, so we need a backup
    # incase a model should be scheduled more than once. 
    model_backup = copy.deepcopy(models)
    model_endtimes = {nw_name: 0.0 for nw_name in models.keys()}

    # While not all nodes have been assigned
    all_assigned = False
    while not all_assigned:
        #Determine all nodes that can be scheduled (all that have no predecessor)
        V_ready: list[tuple[str, LayerResult, ComputationLayer, float]] = []
        misc_layers: list[tuple[str, RenameLayer]] = []

        # When one or more models have a deadline we start by scheduling only those to 
        # increase the likeliness of successfully scheduling them
        all_constraints_scheduled = not any([model.has_constraints and model.number_of_nodes() > 0 for model in models.values()])

        for network, model in models.items():
            if not all_constraints_scheduled and not model.has_constraints:
                continue

            for layer in model.nodes:
                if model.in_degree(layer) == 0:
                    if isinstance(layer, ComputationLayer):
                            platform = layer.scheduling.binding
                            eval_data = find_layer_info(platform, network, layer.layer_name)
                            introduced_delay = layer.scheduling.earliest_start - schedule.node_schedules[platform].cur_time
                            V_ready.append( (network, eval_data, layer, introduced_delay) )
                    else:
                        misc_layers.append((network, layer))
        
        match priority:
            case SchedulingPriority.EARLIEST_START:
                V_ready.sort(key=lambda data: data[2].scheduling.earliest_start)
            case SchedulingPriority.SHORTEST_JOB:
                V_ready.sort(key=lambda data: data[1].latency)
            case SchedulingPriority.EARLIEST_FINISH:
                V_ready.sort(key=lambda data: data[2].scheduling.earliest_start + data[1].latency)
            case SchedulingPriority.LONGEST_JOB:
                V_ready.sort(key=lambda data: -data[1].latency)
            case SchedulingPriority.IDLE_TIME:
                V_ready.sort(key=lambda data: data[3])
            case SchedulingPriority.DEADLINE:
                V_ready.sort(key=lambda data: 
                                models[data[0]].deadline - schedule.node_schedules[data[2].scheduling.binding].cur_time )
            case _:
                raise RuntimeError(f"Found an unknown priority: {priority}")

        for network, eval_data, layer, introduced_delay in V_ready[:1]:
            binding = layer.scheduling.binding
            layer_result = find_layer_info(binding, network, layer.layer_name)
            layer_name = layer.layer_name

            scheduled_task = schedule.schedule_layer(binding, network, layer_result, layer.scheduling.earliest_start)
            model_endtimes[network] = max(model_endtimes[network], scheduled_task.end_time)

            # Before we remove the node, set the earliest start for all successors
            for succ in models[network].successors(layer):
                current_task = schedule.layers[binding][network][layer_name]
                succ.scheduling.earliest_start = max(succ.scheduling.earliest_start, current_task.end_time)

            models[network].remove_layer(layer)
        
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
            if model.number_of_nodes() == 0 and model.frames >= 1:
                remaining_frames = model.frames - 1
                if remaining_frames > 0:
                    models[network] = copy.deepcopy(model_backup[network])
                    models[network].frames = remaining_frames
                    for input_layer in models[network].get_input_layers():
                        # This is needed to reset the earliest start times of the input layers
                        input_layer.scheduling.earliest_start = model_endtimes[network] 

        num_remaining_nodes = sum([graph.number_of_nodes() for graph in models.values()])
        all_assigned = (num_remaining_nodes == 0)

    return schedule