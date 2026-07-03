import math
import os
import copy
from collections import defaultdict
from dataclasses import dataclass
import itertools as it
import time
from collections.abc import Callable
import statistics
import cloudpickle

import ray
from loguru import logger


import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import optuna
from optuna.samplers.nsgaii import UniformCrossover
import tqdm

from framework.optimizer.strategies.multi_tenancy import (MultiTenancyContext,
                                                          HeraldMultiTenancy, 
                                                          SaMultiTenancy, 
                                                          OptunaMultiTenancy,
                                                          PydeapMultiTenancy)

from framework.model.layer_tracker import LayerBindings
from framework.noc.noc import NocTopology
from framework.model.onnx_model import OnnxModel
from framework.model.layers.compute_layer import ComputationLayer
from framework.optimizer.optimizer import Optimizer
from framework.graph_analyzer import GraphAnalyzer
from framework.helpers.config_helper import ConfigHelper
from framework.node.node_evaluator import SystemResult
from framework.scheduling import (list_scheduling, multi_queue_scheduling, sat_solver_scheduling,
                                   SystemSchedule)


class MultiTenancyPartitioning(Optimizer):
    def __init__(self, work_dir: str, node_stats: SystemResult, ga: GraphAnalyzer, global_conf: ConfigHelper, noc: NocTopology):
        self.work_dir = work_dir
        self.plot_dir = os.path.join(work_dir, "figures")
        self.ga = ga
        self.conf = global_conf
        self.system_result = node_stats
        self.noc = noc

        os.makedirs(self.plot_dir, exist_ok=True)

    def optimize(self):
        algorithm = MultiTenancyProblem(self.work_dir, self.system_result, self.ga, self.conf, self.noc)
        solutions, invalid_solutions = algorithm.minimize()
        return solutions, invalid_solutions


@dataclass
class ScheduleResult:
    schedule: SystemSchedule
    tracker: LayerBindings
    is_valid: bool

def scheduling_wrapper(trial: optuna.Trial, models: dict[str, OnnxModel], scheduling_algorithm,
                        node_stats: SystemResult, noc: NocTopology, scheduling_priority, bindings: list[int]) -> ScheduleResult:
    """Wrapper to simplify parallelizing scheduling"""

    # --- SCHEDULING ---
    tracker: LayerBindings = bindings #trial.user_attrs["bindings"]
    system_schedule: SystemSchedule = scheduling_algorithm(models, tracker, node_stats, noc, priority=scheduling_priority)

    # --- VALIDATION ---
    are_constraints_fulfilled = validate_schedule(system_schedule, tracker, models)
    return ScheduleResult(system_schedule, tracker, are_constraints_fulfilled)

@ray.remote
def scheduling_worker(trial, models_id, node_stats_id, noc_id, priority_id, scheduling_algorithm):
    """
    Worker function: Executes the scheduling and validation for one trial.
    """
    # Pull references from shared memory (Zero-copy read)
    base_models = models_id
    node_stats = node_stats_id
    noc = noc_id
    scheduling_priority = priority_id

    # --- SCHEDULING ---
    tracker: LayerBindings = trial.user_attrs["bindings"]

    system_schedule: SystemSchedule = scheduling_algorithm(
        base_models, tracker, node_stats, noc, scheduling_priority
    )

    # --- VALIDATION ---
    are_constraints_fulfilled = validate_schedule(system_schedule, tracker, base_models)

    return ScheduleResult(system_schedule, tracker, are_constraints_fulfilled)

@ray.remote
def scoring_worker(trial: optuna.Trial, models_id, node_stats_id, noc_id, priority_id, cost_fns_id, scheduling_algorithm, bindings_id):
    """
    Worker function: Given a trial, perform all cost evaluations
    """
    # Pull references from shared memory
    base_models = models_id
    node_stats = node_stats_id
    noc = noc_id
    scheduling_priority = priority_id
    cost_fns: list[Callable[[], float]] = cost_fns_id
    tracker: LayerBindings = bindings_id

    cost_terms = []
    sched = None
    for cost_fn in cost_fns:
        if cost_fn.__name__ == "scheduling_wrapper":
            sched: ScheduleResult = cost_fn(trial, base_models, scheduling_algorithm, node_stats, noc, scheduling_priority, tracker)
            cost = sched.schedule.system_latency
        else:
            try:
                cost = cost_fn(tracker)
            except TypeError:
                cost = cost_fn(tracker, base_models)
        cost_terms.append(cost) 
    return cost_terms, sched


    

class MultiTenancyProblem():
    def __init__(self, work_dir: str, node_stats: SystemResult, ga: GraphAnalyzer, global_conf: ConfigHelper, noc: NocTopology):
        self.work_dir = work_dir
        self.plot_dir = os.path.join(work_dir, "figures")
        self.optuna_out_dir = os.path.join(work_dir, "optuna")
        self.node_stats = node_stats
        self.node_ids = node_stats.get_platform_ids()
        self.ga = ga
        self.noc = noc
        self.models = copy.deepcopy(ga.models)
        self.conf = global_conf

        # Determine scheduling algorithms
        match self.conf.mt.scheduling:
            case "list":
                self.scheduling_algorithm = list_scheduling
            case "multi_queue":
                self.scheduling_algorithm = multi_queue_scheduling
            case "sat_solver":
                self.scheduling_algorithm = sat_solver_scheduling
        
        # Parse execution graphs into LayerBindings and generate random initial binding
        network_info = defaultdict(lambda: defaultdict(int))
        node_ids = self.node_stats.get_platform_ids()
        for network, model in ga.models.items():
            for layer in model.nodes:
                if isinstance(layer, ComputationLayer):
                    network_info[network][layer.layer_name] = np.random.choice(node_ids)

        self.layer_tracker = LayerBindings(network_info)
        self.num_nodes = self.layer_tracker.get_num_layers()

    
    
    def minimize(self) -> tuple[list[SystemSchedule], list[SystemSchedule]]:
        # Distribute workload between platforms
        logger.debug(f"Optimizing multitenancy problem with {self.conf.mt.strategy}")
        
        match self.conf.mt.strategy:
            case "optuna":
                strategy = OptunaMultiTenancy()
                #solutions, invalid_solutions = self._optuna_optimizer()
            case "sa":
                strategy = SaMultiTenancy()
                #solutions, invalid_solutions =  self._simulated_annealing()
            case "herald":
                strategy = HeraldMultiTenancy()
                #solutions, invalid_solutions = self._herald_like()
            case "deap":
                strategy = PydeapMultiTenancy()
            case _:
                raise RuntimeError(f"Unknown algorith: {self.conf.mt.strategy}")

        # Perform the optimization based on the specified strategy
        ctx = MultiTenancyContext(
            strategy=strategy,
            node_stats=self.node_stats,
            models=self.models,
            noc=self.noc,
            initial_tracker=self.layer_tracker,
            conf=self.conf,
            scheduling_algorithm=self.scheduling_algorithm,
        )
        solutions, invalid_solutions = ctx.run()

        if len(solutions) > 0:
            # Log some statistics
            system_runtimes: list[float] = []
            system_energy: list[float] = []
            for schedule in solutions:
                system_runtimes.append( max([sched.cur_time for sched in schedule.node_schedules.values()]) )
                system_energy.append(schedule.get_system_energy(self.node_stats))
            
            min_runtime_idx = system_runtimes.index(min(system_runtimes))
            min_runtime_latency = system_runtimes[min_runtime_idx]
            min_runtime_energy = system_energy[min_runtime_idx]
            min_energy_idx = system_energy.index(min(system_energy))
            min_energy_latency = system_runtimes[min_energy_idx]
            min_energy_energy = system_energy[min_energy_idx]
            logger.info(f"Configuration with minimum system runtime (ID: {min_runtime_idx}):\n\tRuntime: {min_runtime_latency}\n\tEnergy: {min_runtime_energy}")
            logger.info(f"Configuration with minimum system energy (ID: {min_energy_idx}):\n\tRuntime: {min_energy_latency}\n\tEnergy: {min_energy_energy}")
            logger.info(f"Median system runtime: {statistics.median(system_runtimes)}")

        else:
            logger.warning("No valid solutions for multi-tenancy scheduling were found")

        return solutions, invalid_solutions


