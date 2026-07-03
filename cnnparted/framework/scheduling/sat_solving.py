import math

from ortools.sat.python import cp_model

from framework.noc.noc import NocTopology
from framework.model.layer_tracker import LayerBindings
from framework.helpers.config.stages_config import SchedulingPriority
from framework.model.onnx_model import OnnxModel
from framework.node.node_evaluator import SystemResult
from .schedules import SystemSchedule
from framework.scheduling.utils import find_layer_info


def sat_solver_scheduling(models: dict[str, OnnxModel], bindings: LayerBindings, node_stats: SystemResult, 
                    noc: NocTopology, priority: SchedulingPriority) -> SystemSchedule:

    compute_graphs = {nw: model.get_computation_graph() for nw, model in models.items()}

    SCALE = 1000 # Scale time to turn float values into int without too much loss of precision
    model = cp_model.CpModel()

    scaled_exec_time_lut: dict[str, dict[str, int]] = {nw: {} for nw in models.keys()}


    starts_list = []
    ends_list = []
    platform_intervals = {id: [] for id in node_stats.get_platform_ids()}
    starts = {nw: {} for nw in models.keys()}
    ends = {nw: {} for nw in models.keys()}

    for network, compute_graph in compute_graphs.items():
        for node in compute_graph:
            binding = bindings.get_binding(network, node.layer_name)
            layer_data = find_layer_info(binding, network, node.layer_name, node_stats)
            scaled_exec_time_lut[network][node.layer_name] = int(math.ceil(layer_data.latency * SCALE))
    
    max_t = sum([t for layers in scaled_exec_time_lut.values() for t in layers.values()])


    # Model the start and end time for each task as an interval
    for network, compute_graph in compute_graphs.items():
        for node in compute_graph:
            start = model.new_int_var(0, max_t, f"layer_start::{network}::{node.layer_name}")
            end = model.new_int_var(0, max_t, f"layer_end::{network}::{node.layer_name}")
            interval = model.new_interval_var(start, scaled_exec_time_lut[network][node.layer_name], end, 
                                              f"interval::{network}::{node.layer_name}")

            starts_list.append(start)
            ends_list.append(end)
            starts[network][node.layer_name] = start
            ends[network][node.layer_name] = end

            binding = bindings.get_binding(network, node.layer_name)
            platform_intervals[binding].append(interval)
    
    # Enforce that only one task is active on each hardware simultaneously
    for platform, intervals in platform_intervals.items():
        model.add_no_overlap(intervals)

    # Precedence constraints: ensure all predecessors are done before a layer starts
    for network, compute_graph in compute_graphs.items():
        for node in compute_graph:
            start_node = starts[network][node.layer_name]
            
            for pred in compute_graph.predecessors(node):
                end_pred = ends[network][pred.layer_name]
                model.add(start_node >= end_pred)

    # Enforce deadlines for each model
    for network, compute_graph in compute_graphs.items():
        if compute_graph.has_constraints:
            scaled_deadline = math.ceil(compute_graph.deadlines[0] * SCALE)
            output_layers = compute_graph.get_output_layers()
            for output_layer in output_layers:
                output_end = ends[network][output_layer.layer_name]
                model.add(output_end <= scaled_deadline)
    
    # Objective function: Minimize makespan
    makespan = model.new_int_var(0, max_t, "makespan")
    model.add_max_equality(makespan, ends_list)
    model.minimize(makespan)

    # Solve the problem
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.solve(model)
    
    schedule = SystemSchedule(node_stats.get_platform_ids(), models.keys(), noc)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        #print(f"Status: {solver.StatusName(status)}")
        #print(f"makespan:  {solver.Value(makespan) / SCALE:.3f} ms\n")

        #Get the starting times for each task
        platform_tasks_start_ms: dict[int, list[tuple[float, str, str]]] = {id: [] for id in node_stats.get_platform_ids()}
        for network, layer_dict in starts.items():
            for layer_name, start_var in layer_dict.items():
                start_ms = solver.Value(starts[network][layer_name]) / SCALE
                binding = bindings.get_binding(network, layer_name)
                platform_tasks_start_ms[binding].append((start_ms, network, layer_name))

        for platform_tasks in platform_tasks_start_ms.values():
            platform_tasks.sort(key=lambda data: data[0])
        
        # Now that we have a sorted list of all compute tasks for every platform,
        # we translate them into the internal SystemSchedule representation
        for platform, tasks in platform_tasks_start_ms.items():
            for task in tasks:
                start_ms, network, layer_name = task
                binding = bindings.get_binding(network, layer_name)
                layer_data = find_layer_info(binding, network, layer_name, node_stats)

                end_ms = solver.Value(ends[network][layer_name]) / SCALE

                schedule.append_layer(binding, network, 0, layer_name, layer_data, start_ms, end_ms, latency=None)
        
        for network, compute_graph in compute_graphs.items():
            if compute_graph.has_constraints:
                output_layers = compute_graph.get_output_layers()
                output_layers_end_ms = [solver.Value(ends[network][output_layer.layer_name]) / SCALE for output_layer in output_layers]
                schedule.network_finish_times[network].append(max(output_layers_end_ms))
    else:
        schedule.is_feasible = False # no feasible solution was found within time budget

    return schedule



