import math

from framework.model.layers.rename_layer import RenameLayer
from framework.scheduling.schedules import SystemSchedule
from framework.helpers.config.stages_config import SchedulingPriority
from framework.model.onnx_model import OnnxModel
from framework.model.layers.compute_layer import ComputationLayer
from framework.node.node_evaluator import LayerResult, SystemResult


def find_layer_info(id: int, nw_name: str, layer_name: str, node_stats: SystemResult):
    network = node_stats[id].designs[0].networks[nw_name]
    if layer_name in network:
        return network[layer_name]
    raise RuntimeError(f"Could not find layer {layer_name} in network {nw_name}")


def get_ready_layers(models: dict[str, OnnxModel], schedule: SystemSchedule, node_stats: SystemResult):
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
                        eval_data = find_layer_info(platform, network, layer.layer_name, node_stats)
                        introduced_delay = layer.scheduling.earliest_start - schedule.node_schedules[platform].cur_time
                        V_ready.append( (network, eval_data, layer, introduced_delay) )
                else:
                    misc_layers.append((network, layer))
    return V_ready, misc_layers


def sort_ready_list(ready_list: list[tuple[str, LayerResult, ComputationLayer, float, float]],
                    schedule: SystemSchedule,
                    models: dict[str, OnnxModel], priority: SchedulingPriority):
    match priority:
        case SchedulingPriority.EARLIEST_START:
            ready_list.sort(key=lambda data: data[2].scheduling.earliest_start)
        case SchedulingPriority.SHORTEST_JOB:
            ready_list.sort(key=lambda data: data[1].latency)
        case SchedulingPriority.EARLIEST_FINISH:
            ready_list.sort(key=lambda data: data[2].scheduling.earliest_start + data[1].latency)
        case SchedulingPriority.LONGEST_JOB:
            ready_list.sort(key=lambda data: -data[1].latency)
        case SchedulingPriority.IDLE_TIME:
            ready_list.sort(key=lambda data: data[3])
        case SchedulingPriority.DYNAMIC_DEADLINE:
            ready_list.sort(key=lambda data: 
                            models[data[0]].deadline - schedule.node_schedules[data[2].scheduling.binding].cur_time )
        case SchedulingPriority.EARLIEST_DEADLINE:
            ready_list.sort(key=lambda data: models[data[0]].deadline)
        case _:
            raise RuntimeError(f"Found an unknown priority {priority}")
    
    return ready_list


def validate_schedule(schedule: SystemSchedule, models: dict[str, OnnxModel]):
    are_constraints_fulfilled = True
    network_end_times = schedule.get_network_finish_times()

    # Check that all layers are of each network are scheduled #frames-times
    scheduled_layers = {nw_name: [] for nw_name in models.keys()}
    for platform_schedule in schedule.node_schedules.values():
        for task in platform_schedule.tasks:
            scheduled_layers[task.network].append(task.layer)
    for network, model in models.items():
        for layer in model.nodes:
            if not isinstance(layer, ComputationLayer):
                continue
            target_times_exec = model.frames
            actual_times_exec = scheduled_layers[network].count(layer.layer_name)
            assert target_times_exec == actual_times_exec, f"Layer {layer.layer_name} is started {actual_times_exec} times but should be started {target_times_exec} times"

    # Check that all successors are scheduled after 
    for network, model in models.items():
        for layer in model.nodes:
            if not isinstance(layer, ComputationLayer):
                continue
            layer_task = schedule.layers[layer.scheduling.binding][network][layer.layer_name]
            for succ in model.successors(layer):
                if not isinstance(succ, ComputationLayer):
                    continue
                # Check that all succesors actually start after their
                # predecessor is finished
                succ_task = schedule.layers[succ.scheduling.binding][network][succ.layer_name]
                assert succ_task.start_time >= layer_task.end_time, f"Layer {succ.layer_name} of {network} is not scheduled after predecssor {layer.layer_name}"
        
        # Check if this schedule fulfills the constraints
        if model.deadline != math.inf and network_end_times[network] > model.deadline:
            # Scheduling with a given deadline
            are_constraints_fulfilled = False
        elif model.frames > 1 and network_end_times[network] > 1_000:
            # Scheduling a fixed amount of frames per second (here: 1000ms)
            are_constraints_fulfilled = False
    
    return are_constraints_fulfilled


