import math
import operator
import random
from functools import partial

import networkx as nx
from deap import base, creator, tools, gp, algorithms

from framework.noc.noc import NocTopology
from framework.model.layer_tracker import LayerBindings
from framework.helpers.config.stages_config import SchedulingPriority
from framework.model.layers.rename_layer import RenameLayer
from framework.model.layers.compute_layer import ComputationLayer
from framework.model.onnx_model import OnnxModel
from framework.node.node_evaluator import SystemResult
from .schedules import SystemSchedule
from .utils import SchedulingConstraints, SchedulingDeadlines, find_layer_info, release_misc_layer

EPS = 1e-6


def _protected_div(a: float, b: float) -> float:
    return a / b if abs(b) > EPS else 1.0


def _build_primitive_set() -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", 4)
    pset.renameArguments(ARG0="latency_ratio", ARG1="idle_ratio",
                         ARG2="critical_path_ratio", ARG3="deadline_urgency")
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(_protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(min, 2)
    pset.addEphemeralConstant(f"rand_const_{id(pset)}", partial(random.uniform, -2.0, 2.0))
    return pset


def _ensure_deap_types():
    if not hasattr(creator, "GPFitnessMin"):
        creator.create("GPFitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "GPIndividual"):
        creator.create("GPIndividual", gp.PrimitiveTree, fitness=creator.GPFitnessMin)


class _PriorityFeatures:
    """Computes the same kind of relative/structural scheduling features used in
    reinforcement_learning.py (latency relative to the current ready set, idle time, remaining
    critical-path length, deadline urgency) as scalar terminals for the GP-evolved formula,
    rather than a fixed-size observation vector for a neural net."""

    def __init__(self, models: dict[str, OnnxModel], bindings: LayerBindings, node_stats: SystemResult):
        self.models = models
        self.bindings = bindings
        self.node_stats = node_stats
        self.critical_path_len, self.network_critical_path_total = self._compute_critical_path_lengths()

        total_latency = EPS
        for network, model in models.items():
            for node in model.nodes:
                if isinstance(node, ComputationLayer):
                    binding = bindings.get_binding(network, node.layer_name)
                    total_latency += model.frames * find_layer_info(binding, network, node.layer_name, node_stats).latency
        # A deadline miss should dominate the fitness ordering over any amount of makespan
        # improvement, so scale the penalty to this instance's own total workload
        self.deadline_miss_penalty = 10.0 * total_latency

    def _compute_critical_path_lengths(self) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """For each layer, the length (sum of latencies) of the longest path from that layer to
        any sink of its network, including its own latency - a static topology property,
        computed once. See the identical helper in reinforcement_learning.py for more detail."""
        path_len: dict[str, dict[str, float]] = {}
        network_total: dict[str, float] = {}
        for network, model in self.models.items():
            own_latency: dict[str, float] = {}
            for node in model.nodes:
                if isinstance(node, ComputationLayer):
                    binding = self.bindings.get_binding(network, node.layer_name)
                    own_latency[node.layer_name] = find_layer_info(binding, network, node.layer_name, self.node_stats).latency
                else:
                    own_latency[node.layer_name] = 0.0

            layer_path_len: dict[str, float] = {}
            for node in reversed(list(nx.topological_sort(model))):
                successor_lens = [layer_path_len[succ.layer_name] for succ in model.successors(node)]
                layer_path_len[node.layer_name] = own_latency[node.layer_name] + (max(successor_lens) if successor_lens else 0.0)

            path_len[network] = layer_path_len
            network_total[network] = max(layer_path_len.values(), default=0.0)

        return path_len, network_total

    def __call__(self, network: str, latency: float, layer: ComputationLayer,
                scheduling_constraints: dict[str, SchedulingConstraints],
                model_deadlines: dict[str, SchedulingDeadlines],
                schedule: SystemSchedule, mean_latency: float) -> tuple[float, float, float, float]:
        binding = self.bindings.get_binding(network, layer.layer_name)
        earliest_start = scheduling_constraints[network][layer].earliest_start
        cur_time = schedule.node_schedules[binding].cur_time
        has_constraints = self.models[network].has_constraints

        latency_ratio = latency / (mean_latency + EPS)
        idle_ratio = _clip((earliest_start - cur_time) / (latency + EPS), -5.0, 5.0)

        critical_path_len = self.critical_path_len[network][layer.layer_name]
        critical_path_ratio = critical_path_len / (self.network_critical_path_total[network] + EPS)

        if has_constraints:
            deadline_urgency = _clip(
                (model_deadlines[network].active_deadline - schedule.system_latency) / (critical_path_len + EPS),
                0.0, 5.0)
        else:
            deadline_urgency = 5.0 # sentinel: plenty of slack

        return latency_ratio, idle_ratio, critical_path_ratio, deadline_urgency


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _run_schedule(models: dict[str, OnnxModel], bindings: LayerBindings, node_stats: SystemResult,
                  noc: NocTopology, score_fn, features: _PriorityFeatures) -> tuple[SystemSchedule, int]:
    """Runs one full list-scheduling pass using `score_fn` as the priority function: at each
    step, every ready layer is scored and the lowest-scoring one is scheduled next. Structurally
    the same ready-set/deadline-gate machinery as list_scheduling.py, just with a pluggable
    scoring function instead of a fixed SchedulingPriority. Returns the schedule and how many
    frame deadlines were missed, for use as the GP fitness."""
    schedule = SystemSchedule(node_stats.get_platform_ids(), list(models.keys()), noc)
    model_deadlines = {nw: SchedulingDeadlines(model.deadlines) for nw, model in models.items()}
    scheduling_constraints: dict[str, SchedulingConstraints] = {nw: SchedulingConstraints(model) for nw, model in models.items()}
    model_endtimes = {nw: 0.0 for nw in models.keys()}
    completed_model_frames = {nw: 0 for nw in models.keys()}
    constrained_ready: list[tuple[str, float, ComputationLayer]] = []
    unconstrained_ready: list[tuple[str, float, ComputationLayer]] = []
    num_incomplete_constrained = sum(1 for model in models.values() if model.has_constraints)
    num_deadline_misses = 0

    def on_ready(network: str, layer: ComputationLayer | RenameLayer):
        if isinstance(layer, ComputationLayer):
            binding = bindings.get_binding(network, layer.layer_name)
            latency = find_layer_info(binding, network, layer.layer_name, node_stats).latency
            pool = constrained_ready if models[network].has_constraints else unconstrained_ready
            pool.append((network, latency, layer))
        else:
            release_misc_layer(models, scheduling_constraints, network, layer, on_ready)

    initially_ready = [(network, layer) for network, model in models.items() for layer in model.nodes
                       if scheduling_constraints[network][layer].in_degree == 0]
    for network, layer in initially_ready:
        on_ready(network, layer)

    all_scheduled = False
    while not all_scheduled:
        active_ready = constrained_ready if num_incomplete_constrained > 0 else unconstrained_ready

        if active_ready:
            mean_latency = sum(latency for _, latency, _ in active_ready) / len(active_ready)
            best_idx, best_score = None, None
            for idx, (network, latency, layer) in enumerate(active_ready):
                feats = features(network, latency, layer, scheduling_constraints, model_deadlines, schedule, mean_latency)
                try:
                    score = float(score_fn(*feats))
                except (ZeroDivisionError, OverflowError, ValueError, TypeError):
                    score = math.inf
                if not math.isfinite(score):
                    score = math.inf
                if best_score is None or score < best_score:
                    best_score, best_idx = score, idx

            network, latency, layer = active_ready.pop(best_idx)
            binding = bindings.get_binding(network, layer.layer_name)
            layer_result = find_layer_info(binding, network, layer.layer_name, node_stats)
            current_frame = completed_model_frames[network]
            earliest_start = scheduling_constraints[network][layer].earliest_start

            scheduled_task = schedule.schedule_layer(binding, network, current_frame, earliest_start, layer_result, layer)
            model_endtimes[network] = max(model_endtimes[network], scheduled_task.end_time)

            for succ in models[network].successors(layer):
                succ_constr = scheduling_constraints[network][succ]
                succ_constr.earliest_start = max(succ_constr.earliest_start, scheduled_task.end_time)
                succ_constr.in_degree -= 1
                if succ_constr.in_degree == 0:
                    on_ready(network, succ)

            scheduling_constraints[network][layer].is_scheduled = True

        # Logic to schedule a network multiple times, mirroring list_scheduling.py
        for nw, model in models.items():
            if scheduling_constraints[nw].network_is_complete:
                if not model_deadlines[nw].all_frames_are_done:
                    completed_model_frames[nw] += 1
                    model_deadlines[nw].mark_frame_complete()
                    schedule.network_finish_times[nw].append(model_endtimes[nw])
                    if model_endtimes[nw] > model_deadlines[nw].previous_deadline:
                        num_deadline_misses += 1
                    if model.has_constraints:
                        num_incomplete_constrained -= 1

                    if not model_deadlines[nw].all_frames_are_done:
                        scheduling_constraints[nw] = SchedulingConstraints(model)
                        if model.has_constraints:
                            num_incomplete_constrained += 1
                        for input_layer in models[nw].get_input_layers():
                            scheduling_constraints[nw][input_layer].earliest_start = model_deadlines[nw].previous_deadline
                            on_ready(nw, input_layer)

        all_scheduled = all(sc.network_is_complete for sc in scheduling_constraints.values())

    return schedule, num_deadline_misses


def _evaluate(individual, pset: gp.PrimitiveSet, models: dict[str, OnnxModel], bindings: LayerBindings,
             node_stats: SystemResult, noc: NocTopology, features: _PriorityFeatures) -> tuple[float]:
    score_fn = gp.compile(expr=individual, pset=pset)
    schedule, num_misses = _run_schedule(models, bindings, node_stats, noc, score_fn, features)
    return (schedule.system_latency + features.deadline_miss_penalty * num_misses,)


def save_priority_formula(individual: gp.PrimitiveTree, path: str):
    with open(path, "w") as f:
        f.write(str(individual))


def load_priority_formula(path: str, pset: gp.PrimitiveSet):
    with open(path) as f:
        expr_str = f.read()
    individual = gp.PrimitiveTree.from_string(expr_str, pset)
    return gp.compile(expr=individual, pset=pset)


def gp_scheduling(models: dict[str, OnnxModel], bindings: LayerBindings, node_stats: SystemResult,
                  noc: NocTopology, priority: SchedulingPriority | None = None, *,
                  population_size: int = 30, generations: int = 20, max_tree_depth: int = 5,
                  cxpb: float = 0.7, mutpb: float = 0.2, seed: int | None = None,
                  verbose: bool = False, save_path: str | None = None,
                  load_path: str | None = None) -> SystemSchedule:

    _ensure_deap_types()
    pset = _build_primitive_set()
    features = _PriorityFeatures(models, bindings, node_stats)

    if load_path is not None:
        best_score_fn = load_priority_formula(load_path, pset)
    else:
        if seed is not None:
            random.seed(seed)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.GPIndividual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", _evaluate, pset=pset, models=models, bindings=bindings,
                         node_stats=node_stats, noc=noc, features=features)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_depth))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_depth))

        population = toolbox.population(n=population_size)
        hall_of_fame = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", min)
        stats.register("avg", lambda vals: sum(vals) / len(vals))

        algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations,
                            stats=stats, halloffame=hall_of_fame, verbose=verbose)

        best_score_fn = gp.compile(expr=hall_of_fame[0], pset=pset)
        if verbose:
            print(f"Best evolved priority formula: {hall_of_fame[0]}")

        if save_path is not None:
            save_priority_formula(hall_of_fame[0], save_path)

    schedule, _ = _run_schedule(models, bindings, node_stats, noc, best_score_fn, features)
    return schedule
