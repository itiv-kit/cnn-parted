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

    # While not all nodes have been assigned
    all_assigned = False
    while not all_assigned:
        #Determine all nodes that can be scheduled (all that have no predecessor)
        V_ready: list[tuple[str, LayerResult, ComputationLayer, float]] = []
        misc_layers: list[tuple[str, RenameLayer]] = []
        for network, model in models.items():
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
                V_ready = sorted(V_ready, key=lambda data: data[2].scheduling.earliest_start)
            case SchedulingPriority.SHORTEST_JOB:
                V_ready = sorted(V_ready, key=lambda data: data[1].latency)
            case SchedulingPriority.EARLIEST_FINISH:
                V_ready = sorted(V_ready, key=lambda data: data[2].scheduling.earliest_start + data[1].latency)
            case SchedulingPriority.LONGEST_JOB:
                V_ready = sorted(V_ready, key=lambda data: -data[1].latency)
            case SchedulingPriority.IDLE_TIME:
                V_ready = sorted(V_ready, key=lambda data: data[3])
            case _:
                raise RuntimeError(f"Found an unknown {priority}")

        for network, eval_data, layer, introduced_delay in V_ready[:1]:
            binding = layer.scheduling.binding
            layer_result = find_layer_info(binding, network, layer.layer_name)
            layer_name = layer.layer_name

            schedule.assign_layer(binding, network, layer_result, layer.scheduling.earliest_start)

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

        num_remaining_nodes = sum([graph.number_of_nodes() for graph in models.values()])
        all_assigned = (num_remaining_nodes == 0)

    return schedule