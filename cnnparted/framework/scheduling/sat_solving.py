import itertools as it
import math

from framework.helpers.config.stages_config import SchedulingPriority
from framework.helpers.dtypes import NocCoordinate
from framework.model.layer_tracker import LayerBindings
from framework.model.onnx_model import OnnxModel
from framework.noc.noc import NocTopology
from framework.node.node_evaluator import SystemResult
from framework.scheduling.utils import find_layer_info
from ortools.sat.python import cp_model

from .schedules import SystemSchedule

SCALE = 1000 # Scale time to turn float values into int without too much loss of precision

def sat_solver_scheduling(models: dict[str, OnnxModel], bindings: LayerBindings, node_stats: SystemResult,
                    noc: NocTopology, priority: SchedulingPriority | None = None, *,
                    num_search_workers: int = 8, max_time_in_seconds: float = 30.0) -> SystemSchedule:

    compute_graphs = {nw: model.get_computation_graph() for nw, model in models.items()}

    model = cp_model.CpModel()

    scaled_exec_time_lut: dict[str, dict[str, int]] = {nw: {} for nw in models.keys()}


    # Model utilization of compute ressources
    compute_starts_list = []
    compute_ends_list = []
    compute_intervals = {id: [] for id in node_stats.get_platform_ids()}
    memory_intervals = {id: [] for id in node_stats.get_platform_ids()}
    # Every network's graph is unrolled once per frame it must run, since each frame gets its
    # own start/end variables and its own deadline
    compute_starts: dict[str, dict[int, dict[str, cp_model.IntVar]]] = {nw: {} for nw in models.keys()}
    compute_ends: dict[str, dict[int, dict[str, cp_model.IntVar]]] = {nw: {} for nw in models.keys()}
    transfer_inp_ends: dict[str, dict[int, dict[str, cp_model.IntVar]]] = {nw: {} for nw in models.keys()}
    transfer_out_ends: dict[str, dict[int, dict[str, cp_model.IntVar]]] = {nw: {} for nw in models.keys()}

    # Model data transfer to and from the accelerators
    transfer_intervals = {id: [] for id in noc.link_ids}

    for network, compute_graph in compute_graphs.items():
        for node in compute_graph:
            binding = bindings.get_binding(network, node.layer_name)
            layer_data = find_layer_info(binding, network, node.layer_name, node_stats)
            scaled_exec_time_lut[network][node.layer_name] = int(math.ceil(layer_data.latency * SCALE))

    # Routes and per-hop transfer durations only depend on a layer's binding, not on which frame
    # is executing, so they're computed once here and reused below - both to avoid recomputing
    # them per frame and so max_t (below) can size itself using them too.
    route_lut: dict[str, dict[str, list]] = {nw: {} for nw in models}
    hop_duration_tick_inp_lut: dict[str, dict[str, int]] = {nw: {} for nw in models}
    hop_duration_tick_out_lut: dict[str, dict[str, int]] = {nw: {} for nw in models}
    for network, compute_graph in compute_graphs.items():
        for node in compute_graph:
            platform = bindings.get_binding(network, node.layer_name)
            start_dram = noc.get_nearest_dram_node(platform)
            visited_nodes = noc.singlecast_routing(start_dram, platform)
            route_lut[network][node.layer_name] = list(it.pairwise([n.coord for n in visited_nodes]))

            num_bytes_inp = sum([inp.num_elems for inp in node.inputs]) #assume 8bit quantization
            hop_duration_inp, _, _ = noc.default_link.eval(num_bytes_inp) #TODO: Correct links
            hop_duration_tick_inp_lut[network][node.layer_name] = max(math.ceil(hop_duration_inp * SCALE), 1)

            num_bytes_out = node.output.num_elems #assume 8bit quantization
            hop_duration_out, _, _ = noc.default_link.eval(num_bytes_out)
            hop_duration_tick_out_lut[network][node.layer_name] = max(math.ceil(hop_duration_out * SCALE), 1)

    # max_t bounds every variable in the model, including the transfer-chain variables below -
    # not just compute time - or those chains could be clamped below what a valid schedule needs,
    # making the model spuriously infeasible regardless of deadlines.
    max_t = sum(
        compute_graph.frames * sum(
            scaled_exec_time_lut[network][node.layer_name]
            + len(route_lut[network][node.layer_name]) * (hop_duration_tick_inp_lut[network][node.layer_name]
                                                           + hop_duration_tick_out_lut[network][node.layer_name])
            for node in compute_graph
        )
        for network, compute_graph in compute_graphs.items()
    )


    # Model the start and end time for each compute task, once per frame, as an interval
    for network, compute_graph in compute_graphs.items():
        for frame in range(compute_graph.frames):
            compute_starts[network][frame] = {}
            compute_ends[network][frame] = {}
            for node in compute_graph:
                start = model.new_int_var(0, max_t, f"layer_start::{network}::{frame}::{node.layer_name}")
                end = model.new_int_var(0, max_t, f"layer_end::{network}::{frame}::{node.layer_name}")
                interval = model.new_interval_var(start, scaled_exec_time_lut[network][node.layer_name], end,
                                                  f"interval::{network}::{frame}::{node.layer_name}")

                compute_starts_list.append(start)
                compute_ends_list.append(end)
                compute_starts[network][frame][node.layer_name] = start
                compute_ends[network][frame][node.layer_name] = end

                binding = bindings.get_binding(network, node.layer_name)
                compute_intervals[binding].append(interval)

    # Model the data transfers, reusing the routes/per-hop durations computed above
    for network, compute_graph in compute_graphs.items():
        for frame in range(compute_graph.frames):
            transfer_inp_ends[network][frame] = {}
            transfer_out_ends[network][frame] = {}

        for node in compute_graph:
            route = route_lut[network][node.layer_name]
            hop_duration_tick_inp = hop_duration_tick_inp_lut[network][node.layer_name]
            hop_duration_tick_out = hop_duration_tick_out_lut[network][node.layer_name]

            for frame in range(compute_graph.frames):
                prev_end = 0
                for hop_idx, link in enumerate(route):
                    s = model.new_int_var(0, max_t, f"xfer_{network}_{frame}_{node.layer_name}_hop{hop_idx}_start")
                    e = model.new_int_var(0, max_t, f"xfer_{network}_{frame}_{node.layer_name}_hop{hop_idx}_end")
                    iv = model.new_interval_var(s, hop_duration_tick_inp, e, f"xfer_{network}_{frame}_{node.layer_name}_hop{hop_idx}")

                    model.add(s >= prev_end)
                    transfer_intervals[link].append(iv)
                    prev_end = e

                transfer_inp_ends[network][frame][node.layer_name] = e
                model.add(compute_starts[network][frame][node.layer_name] >= e)

                prev_end = 0
                for hop_idx, link in enumerate(route):
                    s_out = model.new_int_var(0,  max_t, f"xfer_out_{network}_{frame}_{node.layer_name}_hop{hop_idx}_start")
                    e_out = model.new_int_var(0,  max_t, f"xfer_out_{network}_{frame}_{node.layer_name}_hop{hop_idx}_end")
                    iv_out = model.new_interval_var(s_out, hop_duration_tick_out, e_out, f"xfer_out_{network}_{frame}_{node.layer_name}_hop{hop_idx}")
                    model.add(s_out >= prev_end)
                    transfer_intervals[link].append(iv_out)
                    prev_end = e_out

                transfer_out_ends[network][frame][node.layer_name] = e_out
                for succ in compute_graph.successors(node):
                    #successors can only start after transfer is complete
                    model.add(compute_starts[network][frame][succ.layer_name] >= e_out)

    # Model the tensor lifetimes
    for network, compute_graph in compute_graphs.items():
        for frame in range(compute_graph.frames):
            for node in compute_graph:

                # Lifetime of the inputs:
                #   From: Arrival of input data at the accelerator
                #   Until: End of computation
                inp_transfer_end = transfer_inp_ends[network][frame][node.layer_name]
                compute_end = compute_ends[network][frame][node.layer_name]
                size_var = model.new_int_var(0, max_t, f"size_input_data_{network}_{frame}_{node.layer_name}")
                model.add(size_var == compute_end - inp_transfer_end)
                memory_interval = model.new_interval_var(inp_transfer_end, size_var, compute_end,
                                                         f"input_data_{network}_{frame}_{node.layer_name}")
                binding = bindings.get_binding(network, node.layer_name)
                num_bytes_inp = sum([inp.num_elems for inp in node.inputs]) #assume 8bit quantization
                memory_intervals[binding].append((memory_interval, num_bytes_inp))

                # Lifetime of the output:
                #   From: Start of computation
                #   Until: Transfer from accelerator is initiated
                compute_start = compute_starts[network][frame][node.layer_name]
                transfer_out_end = transfer_out_ends[network][frame][node.layer_name]
                size_var = model.new_int_var(0, max_t, f"size_output_data_{network}_{frame}_{node.layer_name}")
                model.add(size_var == transfer_out_end - compute_start)
                memory_interval = model.new_interval_var(compute_start, size_var, transfer_out_end,
                                                         f"ouput_data_{network}_{frame}_{node.layer_name}")
                binding = bindings.get_binding(network, node.layer_name)
                num_bytes_out = node.output.num_elems
                memory_intervals[binding].append((memory_interval, num_bytes_out))

    for platform, memory in memory_intervals.items():
        ivs = [iv for iv, _ in memory]
        demands = [d for _, d in memory]
        model.add_cumulative(ivs, demands, 200_000_000 // 8)

    # Enforce that only one task is active on each hardware simultaneously
    for platform, intervals in compute_intervals.items():
        model.add_no_overlap(intervals)

    # Precedence constraints: ensure all predecessors are done before a layer starts
    for network, compute_graph in compute_graphs.items():
        for frame in range(compute_graph.frames):
            for node in compute_graph:
                start_node = compute_starts[network][frame][node.layer_name]

                for pred in compute_graph.predecessors(node):
                    end_pred = compute_ends[network][frame][pred.layer_name]
                    model.add(start_node >= end_pred)

    # Enforce deadlines for each model, once per frame
    for network, compute_graph in compute_graphs.items():
        if compute_graph.has_constraints:
            output_layers = compute_graph.get_output_layers()
            for frame in range(compute_graph.frames):
                scaled_deadline = math.ceil(compute_graph.deadlines[frame] * SCALE)
                for output_layer in output_layers:
                    output_end = compute_ends[network][frame][output_layer.layer_name]
                    model.add(output_end <= scaled_deadline)

    # A network's next frame is only released once the current frame's deadline has passed
    for network, compute_graph in compute_graphs.items():
        if compute_graph.frames > 1:
            root_layers = [node for node in compute_graph if compute_graph.in_degree(node) == 0]
            for frame in range(1, compute_graph.frames):
                scaled_prev_deadline = math.ceil(compute_graph.deadlines[frame - 1] * SCALE)
                for root_layer in root_layers:
                    model.add(compute_starts[network][frame][root_layer.layer_name] >= scaled_prev_deadline)

    # Objective function: Minimize makespan
    makespan = model.new_int_var(0, max_t, "makespan")
    model.add_max_equality(makespan, compute_ends_list)
    model.minimize(makespan)

    # Solve the problem
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = num_search_workers
    solver.parameters.max_time_in_seconds = max_time_in_seconds
    status = solver.solve(model)

    schedule = SystemSchedule(node_stats.get_platform_ids(), list(models.keys()), noc)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        #print(f"Status: {solver.StatusName(status)}")
        #print(f"makespan:  {solver.Value(makespan) / SCALE:.3f} ms\n")

        #Get the starting times for each task
        platform_tasks_start_ms: dict[int, list[tuple[float, str, int, str]]] = {id: [] for id in node_stats.get_platform_ids()}
        for network, frame_dict in compute_starts.items():
            for frame, layer_dict in frame_dict.items():
                for layer_name, start_var in layer_dict.items():
                    start_ms = solver.Value(start_var) / SCALE
                    binding = bindings.get_binding(network, layer_name)
                    platform_tasks_start_ms[binding].append((start_ms, network, frame, layer_name))

        for platform_tasks in platform_tasks_start_ms.values():
            platform_tasks.sort(key=lambda data: data[0])

        # Now that we have a sorted list of all compute tasks for every platform,
        # we translate them into the internal SystemSchedule representation
        for platform, tasks in platform_tasks_start_ms.items():
            for task in tasks:
                start_ms, network, frame, layer_name = task
                binding = bindings.get_binding(network, layer_name)
                layer_data = find_layer_info(binding, network, layer_name, node_stats)

                end_ms = solver.Value(compute_ends[network][frame][layer_name]) / SCALE

                compute_task = schedule.append_layer(binding, network, frame, layer_name, layer_data, start_ms, end_ms, latency=None)

                # append_layer only records the compute task and attach communication
                # bookkeeping (latency/energy) used by greedy scheduler.
                # it's used here purely for reporting, not fed back into the solve.
                layer_node = compute_graphs[network].node_by_layer_name(layer_name)
                load_task = schedule.schedule_input_transfers(binding, network, compute_task, layer_node)
                store_task = schedule.schedule_output_transfers(binding, network, compute_task, layer_node)
                schedule.transfer_tasks.append(load_task)
                schedule.transfer_tasks.append(store_task)
                compute_task.loads.append(load_task)
                compute_task.stores.append(store_task)

        for network, compute_graph in compute_graphs.items():
            if compute_graph.has_constraints:
                output_layers = compute_graph.get_output_layers()
                for frame in range(compute_graph.frames):
                    output_layers_end_ms = [solver.Value(compute_ends[network][frame][output_layer.layer_name]) / SCALE for output_layer in output_layers]
                    schedule.network_finish_times[network].append(max(output_layers_end_ms))
    else:
        schedule.is_feasible = False # no feasible solution was found within time budget

    return schedule


