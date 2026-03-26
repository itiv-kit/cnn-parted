import math
import os
import copy
from collections import defaultdict
from dataclasses import dataclass
import sys
import itertools as it
import time

import ray
from loguru import logger

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import optuna
import tqdm

from framework.model.layer_tracker import LayerBindings
from framework.noc.noc import NocTopology
from framework.model.onnx_model import OnnxModel
from framework.model.layers.compute_layer import ComputationLayer
from framework.optimizer.optimizer import Optimizer
from framework.graph_analyzer import GraphAnalyzer
from framework.helpers.config_helper import ConfigHelper
from framework.node.node_evaluator import SystemResult
from framework.scheduling import (list_scheduling, multi_queue_scheduling, 
                                  herald_like, validate_schedule, SystemSchedule)


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
        solutions = algorithm.minimize()
        return solutions


@dataclass
class ScheduleResult:
    schedule: SystemSchedule
    tracker: LayerBindings
    is_valid: bool

def _scheduling_wrapper(trial: optuna.Trial, models: dict[str, OnnxModel], scheduling_algorithm,
                        node_stats: SystemResult, noc: NocTopology, scheduling_priority) -> ScheduleResult:
    """Wrapper to simplify parallelizing scheduling"""

    # --- SCHEDULING ---
    tracker: LayerBindings = trial.user_attrs["bindings"]
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
            
        # Determine which scoring metric is used to measure fairness of latency allocation
        match self.conf.mt.fairness_score:
            case "gini":
                self.score_latency_fairness = self._score_latency_fairness_gini
                self.fairness_direction = ["minimize"]
            case "jain":
                self.score_latency_fairness = self._score_latency_fairness_jain
                self.fairness_direction = ["maximize"]
            case "max_min_space":
                self.score_latency_fairness = self._score_latency_fairness_max_min_space
                self.fairness_direction = ["minimize"]
            case "std_dev":
                self.score_latency_fairness = self._score_latency_fairness_std_dev
                self.fairness_direction = ["minimize"]
            case "no":
                self.score_latency_fairness = None
                self.fairness_direction = []
        
        # Parse execution graphs into LayerBindings and generate random initial binding
        network_info = defaultdict(lambda: defaultdict(int))
        node_ids = self.node_stats.get_platform_ids()
        for network, model in ga.models.items():
            for layer in model.nodes:
                if isinstance(layer, ComputationLayer):
                    network_info[network][layer.layer_name] = np.random.choice(node_ids)

        self.layer_tracker = LayerBindings(network_info)
        self.num_nodes = self.layer_tracker.get_num_layers()
        #self.initial_schedule = self.scheduling_algorithm(self.models, self.node_stats, priority=self.conf.mt.scheduling_priority)

        # SA configuration
        #self.initial_cost = self._calc_cost(self.layer_tracker, self.initial_schedule)
        self.num_iter = self.num_nodes * 100
        #self.t_start = self.initial_cost*8
        self.convergence_counter = 0
        self.cost_history = []
    
    def _herald_like(self) -> list[tuple[LayerBindings, SystemSchedule]]:
        system_schedule = herald_like(self.models, self.node_stats, self.noc, self.conf.mt.optimizer.lbf)

        # Herald combines binding and scheduling, read out values from the schedule
        network_info = {nw_name: {} for nw_name in self.models.keys()}
        for id, node_schedule in system_schedule.node_schedules.items():
            for task in node_schedule.tasks:
                network_info[task.network][task.layer] = id
        bindings = LayerBindings(network_info)

        # Herald only finds one solution
        return [(bindings, system_schedule)]
                
    
    def _simulated_annealing(self) -> list[tuple[LayerBindings, SystemSchedule]]:
        """Optimize the assigment of layers to platforms using simulated annealing.
        This optimizer only supports single-objective optimization."""

        old_cost = self.initial_cost
        best_cost = np.inf
        best_state = None
        best_schedule = None
        current_state = copy.deepcopy(self.layer_tracker)
        improvement_counter = self.convergence_counter
        convergence = False
        rng = np.random.default_rng()

        def _random_modify(self, current_state: LayerBindings) -> LayerBindings:
            layer_tracker = copy.deepcopy(current_state)
            action_space = [layer_tracker.swap_layers, layer_tracker.move_layer, layer_tracker.swap_platforms]
            action = rng.choice(a=action_space, p=[0.45, 0.45, 0.1])
            match action:
                case layer_tracker.swap_layers:
                    platforms_to_swap = rng.choice(self.node_ids, size=2, replace=False)
                    layers_to_swap = []
                    for platform in platforms_to_swap:
                        layers = np.array(layer_tracker.get_all_layers_on_platform(platform))
                        if layers.size == 0:
                            return layer_tracker
                        layers_to_swap.append(rng.choice(layers))
                    action(layers_to_swap)
                case layer_tracker.move_layer:
                    # Move one layer to other platform
                    platform_from, platform_to = rng.choice(self.node_ids, size=2, replace=False)
                    layer_candidates = np.array(layer_tracker.get_all_layers_on_platform(platform_from))
                    if layer_candidates.size == 0:
                        return layer_tracker
                    layer_to_move = rng.choice(layer_candidates)
                    action(layer_to_move, platform_to)
                case layer_tracker.swap_platforms:
                    platform_a, platform_b = rng.choice(self.node_ids, size=2, replace=False)
                    action(platform_a, platform_b)
            return layer_tracker

        while not convergence:
            solution_improved = False
            for i in range(self.num_iter):
                t = self.t_start / (i+1)
                new_state = _random_modify(current_state)

                node_schedules = self.scheduling_algorithm(self.models, new_state, self.node_stats, self.noc, priority=self.conf.mt.scheduling_priority)
                
                new_cost = self._calc_cost(new_state, node_schedules)
                
                if rng.random() <= np.exp( -(new_cost - old_cost) / t ):
                    current_state = new_state
                    old_cost = new_cost
                    
                    # Check for new optimum
                    self.cost_history.append(new_cost)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_state = copy.deepcopy(new_state)
                        best_schedule = copy.deepcopy(node_schedules)
                        solution_improved = True
            
            #Check convergence: No better solution in three cooling cycles
            if solution_improved:
                improvement_counter = self.convergence_counter
            else:
                improvement_counter -= 1

            if improvement_counter <= 0:
                convergence = True
            
        fig = px.line(self.cost_history, title="EDP History for SA")
        fig.update_yaxes(title_text="EDP")
        fig.update_xaxes(title_text="Iteration")
        fig.write_html(os.path.join(self.plot_dir, "edp_history_sa.html"))

        # Result is a list for compatibility with Multi-Objective Opt
        result = (best_state, best_schedule)
        return [result]

    
    def _optuna_optimizer(self) -> list[tuple[LayerBindings, SystemSchedule]]:
        """Optimize the assigment of layers to platforms a specified Optuna sampler.
        This aims to optize energy and latency as a multi-objective optimization.
        """

        optuna_cfg = self.conf.mt.optimizer

        match self.conf.mt.optimizer.sampler:
            case "grid":
                sampler = optuna.samplers.GridSampler()
            case "tpe":
                sampler = optuna.samplers.TPESampler()
            case "cmaes":
                sampler = optuna.samplers.CmaEsSampler()
            case "nsga2":
                sampler = optuna.samplers.NSGAIISampler(seed=42, population_size=optuna_cfg.population_size, 
                                                        mutation_prob=optuna_cfg.mutation_prob, crossover_prob=optuna_cfg.crossover_prob)
            case "random":
                sampler = optuna.samplers.RandomSampler()
            case "qmc":
                sampler = optuna.samplers.QMCSampler()
            case "gp":
                sampler = optuna.samplers.GPSampler()
            case _:
                raise RuntimeError(f"Unknow selection for Optuna sampler, found: {self.conf.mt.optimizer.sampler}")
        
        # Setup information for optuna. For every layer, sample from available 
        # platform ids to determine on which platform it should be executed
        platform_ids = self.conf.system.platform_ids
        work_layer_tracker = copy.deepcopy(self.layer_tracker)
        num_trials = self.conf.mt.optimizer.n_trials
        binding_granularity = self.conf.mt.binding_granularity
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.trial.Trial):
            bindings: list[int] = []

            match binding_granularity:
                case "tile":
                    for i, (nw, layer, current_binding) in enumerate(work_layer_tracker):
                        tile_bind = trial.suggest_categorical(f"tile_binding_{i}", 
                                                               choices=platform_ids)
                        bindings.append(tile_bind)
                case "layer":
                    layer_id_binding: dict[int, int] = {}
                    for i, (nw, layer, current_binding) in enumerate(work_layer_tracker):
                        layer_id = self.models[nw].node_by_layer_name(layer).id
                        if layer_id not in layer_id_binding:
                            layer_bind = trial.suggest_categorical(f"layer_binding_{i}",
                                                                   choices=platform_ids)
                        else:
                            layer_bind = layer_id_binding[layer_id]
                        bindings.append(layer_bind)
                case "network":
                    network_bind = {}
                    for i, nw in enumerate(self.models.keys()):
                        network_bind[nw] = trial.suggest_categorical(f"network_binding_{i}",
                                                                choices=platform_ids)
                    for (nw, layer, current_binding) in work_layer_tracker:
                        bindings.append(network_bind[nw])
                    
            work_layer_tracker.set_all_bindings(bindings)

            # Add information to be later retrieved
            trial.set_user_attr("bindings", copy.deepcopy(work_layer_tracker))

            if self.score_latency_fairness is not None:
                return (
                        self._score_binding_energy(work_layer_tracker) + self._score_tensor_transfer_energy(work_layer_tracker),
                        self._score_binding_latency(work_layer_tracker),
                        self.score_latency_fairness(work_layer_tracker),
                        )
            else:
                return (
                        self._score_binding_energy(work_layer_tracker) + self._score_tensor_transfer_energy(work_layer_tracker),
                        self._score_binding_latency(work_layer_tracker),
                        )

        # Start the optimization, setup everything for Optuna run
        os.makedirs(self.optuna_out_dir, exist_ok=True)
        storage_db = f"sqlite:///{self.optuna_out_dir}/optuna_allocation_opt.db"
        study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize"] + self.fairness_direction, 
                                    study_name="optuna_allocation_opt",) # storage=storage_db
        
        start_time_optuna = time.perf_counter()
        study.optimize(objective, n_trials=num_trials, n_jobs=1, show_progress_bar=True)
        end_time_optuna = time.perf_counter()
        logger.info(f"Runtime of task binding: {end_time_optuna - start_time_optuna}")
        
        # Plotting to make sense of the results
        #fig: go.Figure = optuna.visualization.plot_pareto_front(study, target_names=["Energy", "Latency"], 
        #                                                        targets=lambda t: (t.values[0], t.values[1]),
        #                                                        include_dominated_trials=False)
        #file_name = "pareto_points_optuna"
        #fig.write_html(os.path.join(self.plot_dir, file_name+".html"))
        #fig.write_image(os.path.join(self.plot_dir, file_name+".png"))

        for idx, target in [(0, "Energy"), (1, "Latency"), ]:
            fig: go.Figure = optuna.visualization.plot_optimization_history(study, target_name=target, target=lambda t: t.values[idx])
            file_name = f"history_{target.lower()}_optuna"
            fig.write_html(os.path.join(self.plot_dir, file_name+".html"))
            fig.write_image(os.path.join(self.plot_dir, file_name+".png"))

        logger.debug(f"Multi-Tenancy optimization complete, found {len(study.best_trials)} Parto-optimal solutions")
        logger.info("Performing scheduling of pareto-optimal bindings")

        ### Perform scheduling for pareto-optimal solutions
        results: list[tuple[LayerBindings, SystemSchedule]] = []
        invalid_results: list[tuple[LayerBindings, SystemSchedule]] = []

        #sched_function = partial(_scheduling_wrapper, models=self.models, scheduling_algorithm=self.scheduling_algorithm, 
        #                        noc=self.noc, node_stats=self.node_stats, 
        #                        scheduling_priority=self.conf.mt.scheduling_priority)
        ## Old, single-threaded approach
        #for trial in tqdm.tqdm(study.best_trials):
        #    sched_res = sched_function(trial)
        
        #    if sched_res.is_valid:
        #        results.append( (sched_res.tracker, sched_res.schedule) )
        #    else:
        #        invalid_results.append( (sched_res.tracker, sched_res.schedule) )
        #breakpoint()
        
        # #########################################################
        # ------------------ WIP Ray integration ------------------
        # #########################################################
        # 1. Initialize Ray
        scheduling_start_time = time.perf_counter()
        ray.init(num_cpus=self.conf.mt.num_jobs, 
                 )
        
        # 2. Put static objects in the Shared Memory Object Store
        # This prevents the 100MB 'models' from being copied 1000 times
        models_ref = ray.put(self.models)
        node_stats_ref = ray.put(self.node_stats)
        noc_ref = ray.put(self.noc)
        priority_ref = ray.put(self.conf.mt.scheduling_priority)

        # Decide if scheduling should be performed for all bindings or just Pareto-optimal
        # NOTE: This is very bad for performance
        if self.conf.mt.schedule_all_bindings:
            bindings_to_schedule = study.trials
        else:
            bindings_to_schedule = study.best_trials
        
        task_ids = [
            scheduling_worker.remote(
                trial, models_ref, node_stats_ref, noc_ref, priority_ref, self.scheduling_algorithm
            )
            for trial in bindings_to_schedule
        ]
        
        final_results: list[ScheduleResult] = []
        with tqdm.tqdm(total=len(task_ids), desc="Finalizing Schedules") as pbar:
            remaining = task_ids
            while remaining:
                # Get the next finished task
                done, remaining = ray.wait(remaining, num_returns=1)
                try:
                    result = ray.get(done[0])
                except Exception as e:
                    print(f"WORKER ERROR: {e}")
                final_results.append(result)
                pbar.update(1)
        
        # Shutdown Ray when done to free up shared memory
        ray.shutdown()

        scheduling_end_time = time.perf_counter()
        logger.info(f"Scheduling runtime: {scheduling_end_time-scheduling_start_time}")
        
        for res in final_results:
            if res.is_valid:
                results.append( (res.tracker, res.schedule) )
            else:
                invalid_results.append( (res.tracker, res.schedule) )
        # #########################################################
        # ------------------ END Ray integration ------------------
        # #########################################################

        
        # Log some statistics about deadline success rates
        @dataclass
        class DeadlineStats:
            success: int = 0
            partial_violation: int = 0
            end_violation: int = 0
            end_success: int = 0

        deadline_stats = {network: DeadlineStats() for network in self.models.keys()}
        all_partial_succeeded = []
        all_fully_succeeded = []
        for _, schedule in results + invalid_results:
            network_frame_endtimes = schedule.network_finish_times
            partial_succ = True
            fully_succ = True
            for nw, frame_endtimes in network_frame_endtimes.items():
                
                # Check if every single frame finishes with the deadline
                for frame_endtime, deadline in zip(frame_endtimes, self.models[nw].deadlines, strict=True):
                    if frame_endtime > deadline:
                        deadline_stats[nw].partial_violation += 1
                        fully_succ = False
                        break
                
                if frame_endtimes[-1] > 1000 and self.models[nw].has_frame_constraint:
                    # Check if the last frame is finished before 1s elapsed
                    partial_succ = False
                    fully_succ = False
                    deadline_stats[nw].end_violation += 1
                else:
                    # If the last frame is successful, we count it as a success
                    # This is a relaxed definition
                    partial_succ = True
                    deadline_stats[nw].end_success += 1

            all_fully_succeeded.append(fully_succ)
            all_partial_succeeded.append(partial_succ)
                
        num_all_succ = all_partial_succeeded.count(True)
        num_sched = len(all_partial_succeeded)
        logger.info(f"All deadlines were wet in {num_all_succ}/{num_sched} cases ({num_all_succ/num_sched*100:.2f}%)")
        for nw, stats in deadline_stats.items():
            if self.models[nw].has_constraints:
                logger.info(f"End-frame success for {nw} in {stats.end_success}/{num_sched} cases ({stats.end_success/(num_sched)*100:.2f}%)")

        return results
    
    def minimize(self):
        # Distribute workload between platforms
        logger.debug(f"Optimizing multitenancy problem with {self.conf.mt.algorithm}")

        match self.conf.mt.algorithm:
            case "optuna":
                solutions = self._optuna_optimizer()
            case "sa":
                solutions =  self._simulated_annealing()
            case "herald":
                solutions = self._herald_like()

        # Log some statistics
        system_runtimes: list[float] = []
        system_energy: list[float] = []
        for bindings, schedule in solutions:
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

        return solutions

    def _score_edp(self, tracker: LayerBindings) -> float:
        # For each layer we want to find the most efficient platform
        total_edp = 0.0    
        for nw, layer, bind in tracker:
            # TODO Currently assumes that there is only one design
            layer_data = self.node_stats[bind].designs[0].networks[nw].layers[layer]
            total_edp += layer_data.edp
        return total_edp
    
    def _score_even_finish(self, schedules: SystemSchedule):
        sched_finish_times = [sched.cur_time for sched in schedules.node_schedules.values()]
        latest_finish = max(sched_finish_times)
        total_idling_at_end = 0.0
        for sched in schedules.node_schedules.values():
            total_idling_at_end += (latest_finish - sched.cur_time)
        return total_idling_at_end
    
    def _score_idle_times(self, schedules: SystemSchedule) -> float:
        # Prevent idle times during execution
        idle_times = [sched.get_idle_time() for sched in schedules.node_schedules.values()]
        return sum(idle_times)
    
    def _score_binding_energy(self, tracker: LayerBindings) -> float:
        # For each layer we want to find the most efficient platform
        total_energy = 0.0    
        for nw, layer, bind in tracker:
            # TODO Currently assumes that there is only one design
            layer_data = self.node_stats[bind].designs[0].networks[nw].layers[layer]
            total_energy += layer_data.energy
        return total_energy
    
    def _score_tensor_transfer_energy(self, tracker: LayerBindings) -> float:
        total_energy = 0.0    
        for nw, layer, bind in tracker:
            # Besides executing the layer, consider the energy to transfer the input and output
            # tensors between the platform and the DRAM
            platform_coord = self.noc.id_to_node[bind].coord
            num_hops = self.noc.get_min_hops_to_dram(platform_coord)
            model_layer = self.models[nw].node_by_layer_name(layer)
            tensor_num_elem = [math.prod(tensor.shape) for tensor in model_layer.inputs + [model_layer.output]]
            tensor_transfer_eval = [self.noc.default_link.eval(num_elem) for num_elem in tensor_num_elem]
            tensor_transfer_energy = [energy * num_hops for (latency, energy, slice_size) in tensor_transfer_eval]
            total_energy += sum(tensor_transfer_energy)
        return total_energy
    
    def _score_cuts(self, tracker: LayerBindings) -> int:
        cut_cost = 0
        for network, model in self.models.items():
            for layer in model.nodes:
                if not isinstance(layer, ComputationLayer):
                    continue
                layer_binding = tracker.network_info[network][layer.layer_name]
                all_computation_succs = model.computation_successors(layer)
                found_cuts = 0
                for succ in all_computation_succs:
                    succ_binding = tracker.network_info[network][succ.layer_name]
                    if layer_binding != succ_binding:
                        found_cuts += 1
                if all_computation_succs:
                    cut_cost += (found_cuts // len(all_computation_succs))
        return cut_cost

    def _score_binding_latency(self, tracker: LayerBindings) -> float:
        # For each layer we want to find the most efficient platform
        total_latency = 0.0    
        for nw, layer, bind in tracker:
            # TODO Currently assumes that there is only one design
            layer_data = self.node_stats[bind].designs[0].networks[nw].layers[layer]
            total_latency += layer_data.latency
        return total_latency
    
    def _score_max_latency(self, schedules: SystemSchedule) -> float:
        latencies = [sched.cur_time for sched in schedules.node_schedules.values()]
        return max(latencies)
    
    def _score_static_max_latency(self, tracker:LayerBindings) -> float:
        latencies  = {id: 0 for id in self.conf.system.platform_ids}
        for nw, layer, binding in tracker:
            layer_data = self.node_stats[binding].designs[0].networks[nw].layers[layer]
            latencies[binding] += layer_data.latency
        return max(latencies.values())
    
    def _score_latency_fairness_jain(self, tracker: LayerBindings) -> float:
        """Calculate jains fairness index -> to be maximized"""
        latencies  = {id: 0 for id in self.conf.system.platform_ids}
        for nw, layer, binding in tracker:
            layer_data = self.node_stats[binding].designs[0].networks[nw].layers[layer]
            latencies[binding] += layer_data.latency * self.models[nw].frames
        lats = list(latencies.values())
        
        x = sum(lats)**2
        y = self.conf.system.num_platforms * (sum([lat**2 for lat in lats])) 
        J = x / y
        return J
    
    def _score_latency_fairness_std_dev(self, tracker: LayerBindings) -> float:
        """Calculate std deviation of latency on different platforms"""
        latencies  = {id: 0 for id in self.conf.system.platform_ids}
        for nw, layer, binding in tracker:
            layer_data = self.node_stats[binding].designs[0].networks[nw].layers[layer]
            latencies[binding] += layer_data.latency * self.models[nw].frames
        lats = list(latencies.values())
        mean = (1/len(lats)) * sum(lats)
        var = (1/len(lats)) * sum( (lat - mean)**2 for lat in latencies) 
        return math.sqrt(var)
    
    def _score_latency_fairness_gini(self, tracker: LayerBindings) -> float:
        """Calculate Gini coefficient to measure equality of latency on different platforms"""
        latencies  = {id: 0 for id in self.conf.system.platform_ids}
        for nw, layer, binding in tracker:
            layer_data = self.node_stats[binding].designs[0].networks[nw].layers[layer]
            latencies[binding] += layer_data.latency * self.models[nw].frames
        lats = list(latencies.values())

        mean = (1/len(lats)) * sum(lats)
        x = sum([abs(li - lj) for li, lj in it.permutations(lats, 2)])
        y = 2 * self.conf.system.num_platforms**2*mean
        gini = x / y
        return gini
    
    def _score_latency_fairness_max_min_space(self, tracker: LayerBindings) -> float:
        """Calculate difference of max and min accelerators to score equality of latency on different platforms"""
        latencies  = {id: 0.0 for id in self.conf.system.platform_ids}
        for nw, layer, binding in tracker:
            layer_data = self.node_stats[binding].designs[0].networks[nw].layers[layer]
            latencies[binding] += layer_data.latency * self.models[nw].frames
        lats = list(latencies.values())

        return max(lats) - min(lats)

    def _calc_cost(self, tracker: LayerBindings, schedules: SystemSchedule) -> float:
        edp_cost = self._score_edp(tracker)
        util_cost = self._score_idle_times(schedules)
        lat_cost = self._score_max_latency(schedules)

        return edp_cost**0.4 +  util_cost**2.5 + lat_cost*2
    

