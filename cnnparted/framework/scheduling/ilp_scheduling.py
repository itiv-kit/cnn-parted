import heapq
import itertools as it

from framework.helpers.config.stages_config import SchedulingPriority
from framework.model.layer_tracker import LayerBindings
from framework.model.onnx_model import OnnxModel
from framework.noc.noc import NocTopology
from framework.node.node_evaluator import SystemResult
from framework.scheduling.utils import find_layer_info
from ortools.linear_solver import pywraplp

from .schedules import SystemSchedule


def ilp_scheduling(models: dict[str, OnnxModel], bindings: LayerBindings, node_stats: SystemResult,
                    noc: NocTopology, priority: SchedulingPriority | None = None, *,
                    objective: str = "makespan", hard_rt: bool = False,
                    solver_name: str = "CBC", max_time_in_seconds: float = 30.0) -> SystemSchedule:
    """MILP scheduler implementing the "ILP Scheduling After GA Mapping" formulation of
    Hoffman et al., "MuDSE: GA-ILP-based Framework for Automated Deployment of Multiple DNNs
    on Heterogeneous Mixed-Criticality Systems".
    """

    compute_graphs = {nw: model.get_computation_graph() for nw, model in models.items()}

    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"Could not create ortools MIP solver '{solver_name}'")

    # Eq. 11 (E_{i,p}) and the cost to send/receive a layer's data to/from the nearest DRAM
    # node, which plays the role of the paper's "central bus" (C^S_{i,p}, C^R_{i,p}).
    exec_time_lut: dict[str, dict[str, float]] = {nw: {} for nw in models}
    send_cost_lut: dict[str, dict[str, float]] = {nw: {} for nw in models}
    recv_cost_lut: dict[str, dict[str, float]] = {nw: {} for nw in models}

    for network, compute_graph in compute_graphs.items():
        for node in compute_graph:
            binding = bindings.get_binding(network, node.layer_name)
            layer_data = find_layer_info(binding, network, node.layer_name, node_stats)
            exec_time_lut[network][node.layer_name] = layer_data.latency

            start_dram = noc.get_nearest_dram_node(binding)
            inp_route = list(it.pairwise([n.coord for n in noc.singlecast_routing(start_dram, binding)]))
            out_route = list(it.pairwise([n.coord for n in noc.singlecast_routing(binding, start_dram)]))

            num_bytes_inp = sum([inp.num_elems for inp in node.inputs]) #assume 8bit quantization
            num_bytes_out = node.output.num_elems

            hop_duration_inp, _, _ = noc.default_link.eval(num_bytes_inp) #TODO: Correct links
            hop_duration_out, _, _ = noc.default_link.eval(num_bytes_out)

            recv_cost_lut[network][node.layer_name] = hop_duration_inp * len(inp_route)
            send_cost_lut[network][node.layer_name] = hop_duration_out * len(out_route)

    # Eq. 6-10 (with the corrected receive-cost indexing described above): a task's total
    # communication overhead, charged only when it isn't co-located with a data-sharing
    # neighbor, plus always for external input/output (no on-chip predecessor/successor).
    comm_cost_lut: dict[str, dict[str, float]] = {nw: {} for nw in models}
    for network, compute_graph in compute_graphs.items():
        for node in compute_graph:
            binding = bindings.get_binding(network, node.layer_name)
            preds = list(compute_graph.predecessors(node))
            succs = list(compute_graph.successors(node))

            needs_recv = not preds or any(bindings.get_binding(network, p.layer_name) != binding for p in preds)
            needs_send = not succs or any(bindings.get_binding(network, s.layer_name) != binding for s in succs)

            a_R = recv_cost_lut[network][node.layer_name] if needs_recv else 0.0  # Eq. 9
            a_S = send_cost_lut[network][node.layer_name] if needs_send else 0.0  # Eq. 8
            comm_cost_lut[network][node.layer_name] = a_S + a_R  # Eq. 10

    # big-M / variable upper bound: any value >= the largest possible finish time
    max_t = sum(
        compute_graph.frames * sum(
            exec_time_lut[network][node.layer_name] + comm_cost_lut[network][node.layer_name]
            for node in compute_graph
        )
        for network, compute_graph in compute_graphs.items()
    ) + 1.0
    big_m = max_t

    # Eq. 2-3: t^s_i, t^f_i for every (network, frame, layer) task instance. Every network's
    # graph is unrolled once per frame it must run, since each frame gets its own start/end
    # variables and its own deadline.
    t_s: dict[str, dict[int, dict[str, pywraplp.Variable]]] = {nw: {} for nw in models}
    t_f: dict[str, dict[int, dict[str, pywraplp.Variable]]] = {nw: {} for nw in models}
    task_platform: dict[tuple[str, int, str], int] = {}

    for network, compute_graph in compute_graphs.items():
        for frame in range(compute_graph.frames):
            t_s[network][frame] = {}
            t_f[network][frame] = {}
            for node in compute_graph:
                start = solver.NumVar(0, max_t, f"t_s::{network}::{frame}::{node.layer_name}")
                finish = solver.NumVar(0, max_t, f"t_f::{network}::{frame}::{node.layer_name}")
                solver.Add(finish == start + exec_time_lut[network][node.layer_name]
                                    + comm_cost_lut[network][node.layer_name])  # Eq. 3

                t_s[network][frame][node.layer_name] = start
                t_f[network][frame][node.layer_name] = finish
                task_platform[(network, frame, node.layer_name)] = bindings.get_binding(network, node.layer_name)

    # Precedence constraints (Eq. 4): ensure all predecessors are done before a layer starts
    for network, compute_graph in compute_graphs.items():
        for frame in range(compute_graph.frames):
            for node in compute_graph:
                for pred in compute_graph.predecessors(node):
                    solver.Add(t_s[network][frame][node.layer_name] >= t_f[network][frame][pred.layer_name])

    # A networks next frame can only start once the current frames deadline has passed.
    # This mirrors sat_solver_scheduling; the paper itself has no concept of frames.
    for network, compute_graph in compute_graphs.items():
        if compute_graph.frames > 1:
            root_layers = [node for node in compute_graph if compute_graph.in_degree(node) == 0]
            for frame in range(1, compute_graph.frames):
                prev_deadline = compute_graph.deadlines[frame - 1]
                for root_layer in root_layers:
                    solver.Add(t_s[network][frame][root_layer.layer_name] >= prev_deadline)

    # Eq. 5: mutual exclusion for tasks sharing a platform. 
    # A boolean y is only created for pairs that are actually mapped to the same platform,
    tasks_by_platform: dict[int, list[tuple[str, int, str]]] = {}
    for key, platform in task_platform.items():
        tasks_by_platform.setdefault(platform, []).append(key)

    # M must be large enough to make the *inactive* side of the disjunction valid for any
    # value t_f_j can actually take - and that's NOT bounded by platform p's own local
    # workload: a task's precedence chain can pass through *other* platforms first and
    # accumulate delay there before ever reaching p (e.g. a network whose layers/tiles are
    # spread across several accelerators). 
    # Issues due to precision related to the big_m formulation are not handled here. Instead of 
    # directly using the solvers values, they are used as priority functions for a list scheduling.
    for platform, keys in tasks_by_platform.items():
        for idx, (nw_i, fr_i, ln_i) in enumerate(keys):
            s_i, f_i = t_s[nw_i][fr_i][ln_i], t_f[nw_i][fr_i][ln_i]
            for nw_j, fr_j, ln_j in keys[idx + 1:]:
                s_j, f_j = t_s[nw_j][fr_j][ln_j], t_f[nw_j][fr_j][ln_j]
                y = solver.BoolVar(f"y::{platform}::{nw_i}#{fr_i}#{ln_i}::{nw_j}#{fr_j}#{ln_j}")
                solver.Add(s_i >= f_j - big_m * (1 - y))  # Eq. 12
                solver.Add(s_j >= f_i - big_m * y)        # Eq. 13

    # Per-task deadlines D_i, needed for the soft-RT/deadline-miss objectives and hard_rt
    output_layer_names: dict[str, list[str]] = {}
    task_deadline: dict[tuple[str, int, str], float] = {}
    for network, compute_graph in compute_graphs.items():
        if compute_graph.has_constraints:
            output_layers = compute_graph.get_output_layers()
            output_layer_names[network] = [layer.layer_name for layer in output_layers]
            for frame in range(compute_graph.frames):
                deadline = compute_graph.deadlines[frame]
                for output_layer in output_layers:
                    task_deadline[(network, frame, output_layer.layer_name)] = deadline

    if hard_rt:
        if not task_deadline:
            raise ValueError("hard_rt=True requires at least one network with RT constraints")
        for (network, frame, layer_name), deadline in task_deadline.items():
            solver.Add(t_f[network][frame][layer_name] <= deadline)  # Eq. 40

    # Objective function (Sec. III-F/G)
    all_finish_vars = [
        t_f[network][frame][node.layer_name]
        for network, compute_graph in compute_graphs.items()
        for frame in range(compute_graph.frames)
        for node in compute_graph
    ]

    match objective:
        case "sum_finish":  # Eq. 30: minimize the sum of all finish times
            solver.Minimize(solver.Sum(all_finish_vars))
        case "makespan":  # Eq. 31/32: minimize the maximum finish time
            makespan = solver.NumVar(0, max_t, "makespan")
            for finish in all_finish_vars:
                solver.Add(makespan >= finish)
            solver.Minimize(makespan)
        case "soft_rt":  # Eq. 33-35: minimize total overrun time past each deadline
            if not task_deadline:
                raise ValueError("objective='soft_rt' requires at least one network with RT constraints")
            overrun_vars = []
            for (network, frame, layer_name), deadline in task_deadline.items():
                overrun = solver.NumVar(0, max_t, f"overrun::{network}::{frame}::{layer_name}")
                solver.Add(overrun >= t_f[network][frame][layer_name] - deadline)
                overrun_vars.append(overrun)
            solver.Minimize(solver.Sum(overrun_vars))
        case "deadline_misses":  # Eq. 36-39: minimize the number of missed deadlines
            if not task_deadline:
                raise ValueError("objective='deadline_misses' requires at least one network with RT constraints")
            miss_vars = []
            for (network, frame, layer_name), deadline in task_deadline.items():
                miss = solver.BoolVar(f"miss::{network}::{frame}::{layer_name}")
                finish = t_f[network][frame][layer_name]
                solver.Add(finish <= deadline + big_m * miss)              # Eq. 37
                solver.Add(finish >= deadline + 1e-6 - big_m * (1 - miss))  # Eq. 38
                miss_vars.append(miss)
            solver.Minimize(solver.Sum(miss_vars))
        case _:
            raise ValueError(f"Unknown ILP scheduling objective: {objective}")

    solver.SetTimeLimit(int(max_time_in_seconds * 1000))
    status = solver.Solve()

    schedule = SystemSchedule(node_stats.get_platform_ids(), list(models.keys()), noc)

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        # Reconstruct the reported schedule via a forward simulation instead of reading
        # t_s/t_f's raw solution_value()s directly. A MIP solver only guarantees its
        # constraints hold up to a numerical tolerance, and for a model whose coefficients
        # span from single-digit per-layer costs to a big-M sized from the whole system's
        # workload, that tolerance can become visible at the millisecond scale  The ILP's real
        # output is the *combinatorial* decision - which order same-platform tasks
        # run in - not the continuous timestamps. We use each task's solved start time
        # only as a priority for that ordering, then re-derive exact start/end times
        # deterministically, which is exact by construction and needs no tolerance.
        solved_start: dict[tuple[str, int, str], float] = {
            key: t_s[key[0]][key[1]][key[2]].solution_value() for key in task_platform
        }

        # A network's frame>1 root layers may not start before the previous frame's
        # deadline (mirrors the constraint added above, Sec. II).
        release_time: dict[tuple[str, int], float] = {
            (network, frame): compute_graph.deadlines[frame - 1]
            for network, compute_graph in compute_graphs.items() if compute_graph.frames > 1
            for frame in range(1, compute_graph.frames)
        }

        remaining_preds: dict[tuple[str, int, str], int] = {}
        ready: list[tuple[float, str, int, str]] = []
        for (network, frame, layer_name), platform in task_platform.items():
            node = compute_graphs[network].node_by_layer_name(layer_name)
            num_preds = compute_graphs[network].in_degree(node)
            remaining_preds[(network, frame, layer_name)] = num_preds
            if num_preds == 0:
                heapq.heappush(ready, (solved_start[(network, frame, layer_name)], network, frame, layer_name))

        reported_end: dict[tuple[str, int, str], float] = {}
        platform_cur_time: dict[int, float] = {id: 0.0 for id in node_stats.get_platform_ids()}

        while ready:
            _, network, frame, layer_name = heapq.heappop(ready)
            platform = task_platform[(network, frame, layer_name)]
            node = compute_graphs[network].node_by_layer_name(layer_name)

            pred_ready = max(
                [reported_end[(network, frame, pred.layer_name)] for pred in compute_graphs[network].predecessors(node)]
                + [release_time.get((network, frame), 0.0)],
                default=0.0,
            )
            start = max(pred_ready, platform_cur_time[platform])
            end = start + exec_time_lut[network][layer_name]
            reported_end[(network, frame, layer_name)] = end
            platform_cur_time[platform] = end

            binding = bindings.get_binding(network, layer_name)
            layer_data = find_layer_info(binding, network, layer_name, node_stats)
            compute_task = schedule.append_layer(binding, network, frame, layer_name, layer_data, start, end, latency=None)

            load_task = schedule.schedule_input_transfers(binding, network, compute_task, node)
            store_task = schedule.schedule_output_transfers(binding, network, compute_task, node)
            schedule.transfer_tasks.append(load_task)
            schedule.transfer_tasks.append(store_task)
            compute_task.loads.append(load_task)
            compute_task.stores.append(store_task)

            for succ in compute_graphs[network].successors(node):
                key = (network, frame, succ.layer_name)
                remaining_preds[key] -= 1
                if remaining_preds[key] == 0:
                    heapq.heappush(ready, (solved_start[key], network, frame, succ.layer_name))

        for network, compute_graph in compute_graphs.items():
            if compute_graph.has_constraints:
                for frame in range(compute_graph.frames):
                    ends = [reported_end[(network, frame, layer_name)] for layer_name in output_layer_names[network]]
                    schedule.network_finish_times[network].append(max(ends))
    else:
        schedule.is_feasible = False  # no feasible solution was found within the time budget

    return schedule
