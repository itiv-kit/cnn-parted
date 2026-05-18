import math
import os
import copy
from collections import defaultdict
from dataclasses import dataclass
import itertools as it
import time
import sys
import pdb
from collections.abc import Callable
import statistics
import cloudpickle

import ray
from loguru import logger


import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import optuna
from optuna.samplers.nsgaii import BLXAlphaCrossover, UniformCrossover
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

    
    def _optuna_optimizer(self) -> tuple[list[tuple[LayerBindings, SystemSchedule]], list[tuple[LayerBindings, SystemSchedule]]]:
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
                sampler = optuna.samplers.NSGAIISampler(population_size=optuna_cfg.population_size, 
                                                        crossover=UniformCrossover(swapping_prob=0.5),
                                                        mutation_prob=optuna_cfg.mutation_prob, crossover_prob=optuna_cfg.crossover_prob)
            case "nsga3":
                sampler = optuna.samplers.NSGAIIISampler(population_size=optuna_cfg.population_size, 
                                                        crossover=UniformCrossover(swapping_prob=0.5),
                                                        mutation_prob=optuna_cfg.mutation_prob, crossover_prob=optuna_cfg.crossover_prob)
            case "random":
                sampler = optuna.samplers.RandomSampler()
            case "qmc":
                sampler = optuna.samplers.QMCSampler()
            case "gp":
                sampler = optuna.samplers.GPSampler()
            case _:
                raise RuntimeError(f"Unknow selection for Optuna sampler, found: {self.conf.mt.optimizer.sampler}")
        
        # Configuration of cost terms to be calculated
        # Includes optimization direction, function to be called and pretty-print name
        cost_term_mapping = {
            "energy":           ("minimize", self._score_binding_energy, "Cum. Energy"),
            "latency":          ("minimize", self._score_binding_latency, "Cum. Latency"),
            "gini":             ("minimize", self._score_latency_fairness_gini, "Gini"),
            "jain":             ("maximize", self._score_latency_fairness_jain, "Jain"),
            "max_min_space":    ("minimize", self._score_latency_fairness_max_min_space, "Max-Min Space"),
            "std_dev":          ("minimize", self._score_latency_fairness_std_dev, "Std. dev"),
            "crit_path":        ("minimize", self._score_scheduling_proxy, "Critical Path"),
            "scheduling":       ("minimize",  scheduling_wrapper, "Makespan")
        }

        cost_terms = []
        cost_fns = []
        opt_direction = []
        for cost_term in self.conf.mt.cost_terms:
            opt_dir, fn, name = cost_term_mapping[cost_term]
            cost_terms.append(name)
            cost_fns.append(fn)
            opt_direction.append(opt_dir)

            
        # Setup information for optuna. For every layer, sample from available 
        # platform ids to determine on which platform it should be executed
        platform_ids = self.conf.system.platform_ids
        work_layer_tracker = copy.deepcopy(self.layer_tracker)
        num_trials = self.conf.mt.optimizer.n_trials
        binding_granularity = self.conf.mt.binding_granularity
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _suggest_task_bindings(trial: optuna.trial.Trial) -> list[int]:
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
                    
            return bindings


        def objective(trial: optuna.trial.Trial):
            bindings = _suggest_task_bindings(trial)
            work_layer_tracker.set_all_bindings(bindings)
            trial.set_user_attr("bindings", copy.deepcopy(work_layer_tracker))

            cost_values = []
            for cost_fn in cost_fns:
                try:
                    cost = cost_fn(work_layer_tracker)
                except TypeError:
                    cost = cost_fn(work_layer_tracker, self.models)
                cost_values.append(cost)

            return cost_values

        # Start the optimization, setup everything for Optuna run
        # We only create the study here, the optimization is performed later depending on configuration:
        #   1) Perform scheduling for ALL trials, use ask-and-tell interface
        #   2) Perform scheduling only for pareto optimal task allocations
        os.makedirs(self.optuna_out_dir, exist_ok=True)

        results: list[tuple[LayerBindings, SystemSchedule]] = []
        invalid_results: list[tuple[LayerBindings, SystemSchedule]] = []

        study = optuna.create_study(sampler=sampler, directions=opt_direction, 
                                    study_name="optuna_allocation_opt",) # storage=storage_db

        
        # #########################################################
        # ------------------ WIP Ray integration ------------------
        # #########################################################

        # 1. Initialize Ray
        scheduling_start_time = time.perf_counter()
        ray.init(num_cpus=max(self.conf.mt.num_jobs, 1), ignore_reinit_error=True)
        
        # 2. Put static objects in the Shared Memory Object Store
        models_ref = ray.put(self.models)
        node_stats_ref = ray.put(self.node_stats)
        noc_ref = ray.put(self.noc)
        priority_ref = ray.put(self.conf.mt.scheduling_priority)
        cost_fns_ref = ray.put(cost_fns)



        if "scheduling" in self.conf.mt.cost_terms:    #self.conf.mt.schedule_all_bindings:
            # Scheduling must be evaluated for every Optuna trial
            # Pipelined implementation for ray: whenever a task finishes we immediately add a new one
            # Scheduling is very compute intesive, so we try to minimize runtime by doing so.
            task_mapping: dict[ray.ObjectRef, optuna.Trial] = {} 
            all_schedules: dict[int, ScheduleResult] = {}       # {trial_number: ScheduleResult}
            finished_count = 0
            pbar = tqdm.tqdm(total=num_trials, desc="Pipelined Optimization")

            # A. Prime the pipeline: Fill all CPU cores with initial tasks
            for _ in range(self.conf.mt.num_jobs):
                if finished_count + len(task_mapping) < num_trials:
                    trial = study.ask()
                    bindings = _suggest_task_bindings(trial)
                    work_layer_tracker.set_all_bindings(bindings)
                    bindings_ref = ray.put(copy.deepcopy(work_layer_tracker))
                    
                    # Dispatch to Ray immediately
                    t_id = scoring_worker.remote(
                        trial,
                        models_ref, 
                        node_stats_ref, 
                        noc_ref, 
                        priority_ref, 
                        cost_fns_ref,
                        self.scheduling_algorithm,
                        bindings_ref
                    )
                    task_mapping[t_id] = trial
            
            # B. The Continuous Loop: As one task finishes, start one more
            while task_mapping:
                # Wait for any single task to finish
                ready_ids, _ = ray.wait(list(task_mapping.keys()), num_returns=1)
                
                for t_id in ready_ids:
                    # 1. Retrieve the Ray result and the linked Optuna trial
                    try:
                        result: tuple[list[float], ScheduleResult] = ray.get(t_id) 
                    except Exception as e:
                        result = e
                    
                    # Remove from task tracking before cleanup
                    trial = task_mapping.pop(t_id, None)
                    
                    if trial is None:
                        continue
                    
                    # 2. Handle results and update Optuna
                    if isinstance(result, Exception):
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        logger.error(f"Trial {trial.number} failed: {result}")
                    else:
                        study.tell(trial, result[0])
                        
                        #all_schedules[trial.number] = result[1]
                        schedule_file = os.path.join(self.optuna_out_dir, f"schedule_trial_{trial.number}.pkl")
                        all_schedules[trial.number] = schedule_file
                        with open(schedule_file, 'wb') as f:
                            cloudpickle.dump(result[1], f)  # Store only ScheduleResult to disk
                    
                    finished_count += 1
                    pbar.update(1)

                    # 3. Refill the pipeline: Ask Optuna for the NEXT trial 
                    if (finished_count + len(task_mapping)) < num_trials:
                        new_trial = study.ask()
                        bindings = _suggest_task_bindings(new_trial)
                        work_layer_tracker.set_all_bindings(bindings)
                        bindings_ref = ray.put(copy.deepcopy(work_layer_tracker))
                        new_t_id = scoring_worker.remote(
                            new_trial,
                            models_ref, 
                            node_stats_ref, 
                            noc_ref, 
                            priority_ref, 
                            cost_fns_ref,
                            self.scheduling_algorithm,
                            bindings_ref
                        )
                        
                        task_mapping[new_t_id] = new_trial
            
            pbar.close()
            ray.shutdown()

            # 5. --- Post-Processing: Filtering Pareto Front ---
            logger.info(f"Found {len(study.best_trials)} Pareto-optimal solutions")
            for trial in study.best_trials:
                #res = all_schedules[trial.number]
                schedule_file = all_schedules[trial.number]
                with open(schedule_file, "rb") as f:
                   res: ScheduleResult = cloudpickle.load(f)
                if res and res.is_valid:
                    results.append((res.tracker, res.schedule))
                elif res:
                    invalid_results.append((res.tracker, res.schedule))



        else:
            # Scheduling is only performed for Pareto-optimal task allocations
            study.optimize(objective, n_trials=num_trials, n_jobs=1, show_progress_bar=True)
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
                    obj_ref = done[0]
                    try:
                        result = ray.get(obj_ref)
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


        # Plotting to make sense of the results
        fig: go.Figure = optuna.visualization.plot_pareto_front(study, target_names=["Energy", "Latency"], 
                                                                targets=lambda t: (t.values[0], t.values[1]),
                                                                include_dominated_trials=False)

        fig.update_layout(
            title_text="",
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(
                size=26,
            ),
        )
        fig.update_xaxes(
            tickfont_size=22,
        )
        fig.update_yaxes(
            tickfont_size=22,
        )

        file_name = "pareto_points_optuna"
        fig.write_html(os.path.join(self.plot_dir, file_name+".html"))
        fig.write_image(os.path.join(self.plot_dir, file_name+".png"))

        for idx, target in enumerate(cost_terms):
            fig: go.Figure = optuna.visualization.plot_optimization_history(study, target_name=target, target=lambda t: t.values[idx])
            fig.update_layout(
                title_text="",
                margin=dict(l=20, r=20, t=20, b=20),
                font=dict(
                    size=26,
                ),
            )
            fig.update_xaxes(
                tickfont_size=22,
            )
            fig.update_yaxes(
                tickfont_size=22,
            )
            file_name = f"history_{target.lower()}_optuna"
            fig.write_html(os.path.join(self.plot_dir, file_name+".html"))
            fig.write_image(os.path.join(self.plot_dir, file_name+".png"))

        logger.debug(f"Multi-Tenancy optimization complete, found {len(study.best_trials)} Parto-optimal solutions")
        logger.info("Performing scheduling of pareto-optimal bindings")
        
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

        return results, invalid_results
    
    def minimize(self):
        # Distribute workload between platforms
        logger.debug(f"Optimizing multitenancy problem with {self.conf.mt.algorithm}")

        match self.conf.mt.algorithm:
            case "optuna":
                solutions, invalid_solutions = self._optuna_optimizer()
            case "sa":
                solutions =  self._simulated_annealing()
            case "herald":
                solutions = self._herald_like()

        if len(solutions) > 0:
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
            logger.info(f"Median system runtime: {statistics.median(system_runtimes)}")

        else:
            logger.warning("No valid solutions for multi-tenancy scheduling were found")

        return solutions, invalid_solutions

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

    def _score_total_binding_energy(self, tracker: LayerBindings) -> float:
        return self._score_binding_energy(tracker) + self._score_tensor_transfer_energy(tracker)
    
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
            latencies[binding] += layer_data.latency #* self.models[nw].frames
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
            latencies[binding] += layer_data.latency #* self.models[nw].frames
        lats = list(latencies.values())
        mean = (1/len(lats)) * sum(lats)
        var = (1/len(lats)) * sum( (lat - mean)**2 for lat in latencies) 
        return math.sqrt(var)
    
    def _score_latency_fairness_gini(self, tracker: LayerBindings) -> float:
        """Calculate Gini coefficient to measure equality of latency on different platforms"""
        latencies  = {id: 0 for id in self.conf.system.platform_ids}
        for nw, layer, binding in tracker:
            layer_data = self.node_stats[binding].designs[0].networks[nw].layers[layer]
            latencies[binding] += layer_data.latency #* self.models[nw].frames
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
            latencies[binding] += layer_data.latency #* self.models[nw].frames
        lats = list(latencies.values())

        return max(lats) - min(lats)
    
    def _score_scheduling_proxy(self, tracker: LayerBindings, models: dict[str, OnnxModel]):
        # Determine when the last accelerator finishes
        latest_platform_finish_time = self._score_static_max_latency(tracker)

        # Determine when the last network finishes
        network_finish_time: dict[str, float] = {}
        for network, model in models.items():
            layer_finish_time: dict[str, float] = {layer.layer_name: 0 for layer in model.nodes}
            for layer in model.nodes:

                # Set the base value for the layer finish time: the time to process it
                if isinstance(layer, ComputationLayer):
                    # assumption that only computation layers have an execution time
                    layer_binding = tracker.get_binding(network, layer.layer_name)
                    layer_finish_time[layer.layer_name] = self.node_stats[layer_binding].design.networks[network].layers[layer.layer_name].latency
                else:
                    layer_finish_time[layer.layer_name] = 0

                pred_latencies = []
                for pred in model.predecessors(layer):
                    pred_latencies.append(layer_finish_time[pred.layer_name])

                if len(pred_latencies) > 0:
                    layer_finish_time[layer.layer_name] += max(pred_latencies)

            network_finish_time[network] = max(layer_finish_time.values())
        
        # The network that takes longest to execute
        latest_network_finish_time = max(network_finish_time.values())

        return max(latest_platform_finish_time, latest_network_finish_time)

    def _score_task_clumping(self, tracker: LayerBindings):
        task_counts = {}
        for nw, layer, binding in tracker:
            if binding not in task_counts:
                task_counts[binding] = 0
            task_counts[binding] += 1
        
        square_counts = [task_count**2 for task_count in task_counts.values()]
        task_density_cost = sum(square_counts)
        return task_density_cost



    def _calc_cost(self, tracker: LayerBindings, schedules: SystemSchedule) -> float:
        edp_cost = self._score_edp(tracker)
        util_cost = self._score_idle_times(schedules)
        lat_cost = self._score_max_latency(schedules)

        return edp_cost**0.4 +  util_cost**2.5 + lat_cost*2
    

