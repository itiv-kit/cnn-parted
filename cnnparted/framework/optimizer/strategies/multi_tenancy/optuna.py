from dataclasses import dataclass
from collections.abc import Callable
import copy
import os
import time

import tqdm
import ray
import optuna
from optuna.samplers.nsgaii import UniformCrossover
import cloudpickle
from loguru import logger
import plotly.graph_objects as go

from framework.optimizer.strategies.multi_tenancy.context import MultiTenancyContext

from ._strategy import MultiTenancyStrategy
from framework.model.layer_tracker import LayerBindings
from framework.noc.noc import NocTopology
from framework.model.onnx_model import OnnxModel
from framework.node.node_evaluator import SystemResult
from framework.scheduling import (validate_schedule, SystemSchedule)
                        

from common import (_score_binding_energy, _score_binding_latency, 
                    _score_latency_fairness_gini,
                    _score_latency_fairness_jain,
                    _score_latency_fairness_max_min_space,
                    _score_latency_fairness_std_dev,
                    _score_scheduling_proxy,)

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




class OptunaMultiTenancy(MultiTenancyStrategy):
    def run(self, ctx: MultiTenancyContext):
        """Optimize the assigment of layers to platforms a specified Optuna sampler.
        This aims to optize energy and latency as a multi-objective optimization.
        """
        optuna_out_dir = os.path.join(ctx.conf.general.work_dir, "optuna")
        plot_dir = os.path.join(ctx.conf.general.work_dir, "figures")


        optuna_cfg = ctx.conf.mt.optimizer

        match ctx.conf.mt.optimizer.sampler:
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
                raise RuntimeError(f"Unknow selection for Optuna sampler, found: {ctx.conf.mt.optimizer.sampler}")
        
        # Configuration of cost terms to be calculated
        # Includes optimization direction, function to be called and pretty-print name
        cost_term_mapping = {
            "energy":           ("minimize", _score_binding_energy, "Cum. Energy"),
            "latency":          ("minimize", _score_binding_latency, "Cum. Latency"),
            "gini":             ("minimize", _score_latency_fairness_gini, "Gini"),
            "jain":             ("maximize", _score_latency_fairness_jain, "Jain"),
            "max_min_space":    ("minimize", _score_latency_fairness_max_min_space, "Max-Min Space"),
            "std_dev":          ("minimize", _score_latency_fairness_std_dev, "Std. dev"),
            "crit_path":        ("minimize", _score_scheduling_proxy, "Critical Path"),
            "scheduling":       ("minimize",  scheduling_wrapper, "Makespan")
        }

        cost_terms = []
        cost_fns = []
        opt_direction = []
        for cost_term in ctx.conf.mt.cost_terms:
            opt_dir, fn, name = cost_term_mapping[cost_term]
            cost_terms.append(name)
            cost_fns.append(fn)
            opt_direction.append(opt_dir)

            
        # Setup information for optuna. For every layer, sample from available 
        # platform ids to determine on which platform it should be executed
        platform_ids = ctx.conf.system.platform_ids
        work_layer_tracker = copy.deepcopy(ctx.initial_tracker)
        num_trials = ctx.conf.mt.optimizer.n_trials
        binding_granularity = ctx.conf.mt.binding_granularity
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
                        layer_id = ctx.models[nw].node_by_layer_name(layer).id
                        if layer_id not in layer_id_binding:
                            layer_bind = trial.suggest_categorical(f"layer_binding_{i}",
                                                                   choices=platform_ids)
                        else:
                            layer_bind = layer_id_binding[layer_id]
                        bindings.append(layer_bind)
                case "network":
                    network_bind = {}
                    for i, nw in enumerate(ctx.models.keys()):
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
                    cost = cost_fn(work_layer_tracker, ctx.models)
                cost_values.append(cost)

            return cost_values

        # Start the optimization, setup everything for Optuna run
        # We only create the study here, the optimization is performed later depending on configuration:
        #   1) Perform scheduling for ALL trials, use ask-and-tell interface
        #   2) Perform scheduling only for pareto optimal task allocations
        os.makedirs(optuna_out_dir, exist_ok=True)

        results: list[tuple[LayerBindings, SystemSchedule]] = []
        invalid_results: list[tuple[LayerBindings, SystemSchedule]] = []

        study = optuna.create_study(sampler=sampler, directions=opt_direction, 
                                    study_name="optuna_allocation_opt",) # storage=storage_db

        
        # #########################################################
        # ------------------ WIP Ray integration ------------------
        # #########################################################

        # 1. Initialize Ray
        scheduling_start_time = time.perf_counter()
        ray.init(num_cpus=max(ctx.conf.mt.num_jobs, 1), ignore_reinit_error=True)
        
        # 2. Put static objects in the Shared Memory Object Store
        models_ref = ray.put(ctx.models)
        node_stats_ref = ray.put(ctx.node_stats)
        noc_ref = ray.put(ctx.noc)
        priority_ref = ray.put(ctx.conf.mt.scheduling_priority)
        cost_fns_ref = ray.put(cost_fns)



        if "scheduling" in ctx.conf.mt.cost_terms:
            # Scheduling must be evaluated for every Optuna trial
            # Pipelined implementation for ray: whenever a task finishes we immediately add a new one
            # Scheduling is very compute intesive, so we try to minimize runtime by doing so.
            task_mapping: dict[ray.ObjectRef, optuna.Trial] = {} 
            all_schedules: dict[int, ScheduleResult] = {}       # {trial_number: ScheduleResult}
            finished_count = 0
            pbar = tqdm.tqdm(total=num_trials, desc="Pipelined Optimization")

            # A. Prime the pipeline: Fill all CPU cores with initial tasks
            for _ in range(ctx.conf.mt.num_jobs):
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
                        ctx.scheduling_algorithm,
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
                        schedule_file = os.path.join(optuna_out_dir, f"schedule_trial_{trial.number}.pkl")
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
                            ctx.scheduling_algorithm,
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
                    trial, models_ref, node_stats_ref, noc_ref, priority_ref, ctx.scheduling_algorithm
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
        fig.write_html(os.path.join(plot_dir, file_name+".html"))
        fig.write_image(os.path.join(plot_dir, file_name+".png"))

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
            fig.write_html(os.path.join(plot_dir, file_name+".html"))
            fig.write_image(os.path.join(plot_dir, file_name+".png"))

        logger.debug(f"Multi-Tenancy optimization complete, found {len(study.best_trials)} Parto-optimal solutions")
        logger.info("Performing scheduling of pareto-optimal bindings")
        
        # Log some statistics about deadline success rates
        @dataclass
        class DeadlineStats:
            success: int = 0
            partial_violation: int = 0
            end_violation: int = 0
            end_success: int = 0

        deadline_stats = {network: DeadlineStats() for network in ctx.models.keys()}
        all_partial_succeeded = []
        all_fully_succeeded = []
        for _, schedule in results + invalid_results:
            network_frame_endtimes = schedule.network_finish_times
            partial_succ = True
            fully_succ = True
            for nw, frame_endtimes in network_frame_endtimes.items():
                
                # Check if every single frame finishes with the deadline
                for frame_endtime, deadline in zip(frame_endtimes, ctx.models[nw].deadlines, strict=True):
                    if frame_endtime > deadline:
                        deadline_stats[nw].partial_violation += 1
                        fully_succ = False
                        break
                
                if frame_endtimes[-1] > 1000 and ctx.models[nw].has_frame_constraint:
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
            if ctx.models[nw].has_constraints:
                logger.info(f"End-frame success for {nw} in {stats.end_success}/{num_sched} cases ({stats.end_success/(num_sched)*100:.2f}%)")

        return results, invalid_results

        