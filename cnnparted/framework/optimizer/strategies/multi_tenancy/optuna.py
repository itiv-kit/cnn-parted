from dataclasses import dataclass
from collections.abc import Callable
import copy
import os
import time
import itertools
from pathlib import Path
import pickle

import tqdm
import ray
import optuna
from optuna.samplers.nsgaii import UniformCrossover
import cloudpickle
from loguru import logger
import plotly.graph_objects as go

from framework.model.layer_tracker import LayerBindings
from framework.noc.noc import NocTopology
from framework.model.onnx_model import OnnxModel
from framework.node.node_evaluator import SystemResult
from framework.scheduling import validate_schedule, SystemSchedule
from framework.scheduling.deadline_eval import deadline_evaluation
from framework.scheduling.dispatch import run_scheduling, SchedulingConfig
from framework.scheduling.utils import SchedulingProblem

from framework.helpers.config_helper import ConfigHelper

from .common import get_scorer

@dataclass
class ScheduleResult:
    schedule: SystemSchedule
    tracker: LayerBindings
    is_valid: bool

def scheduling_wrapper(models: dict[str, OnnxModel], node_stats: SystemResult, noc: NocTopology,
                        config: SchedulingConfig, bindings: LayerBindings) -> ScheduleResult:
    """Wrapper to simplify parallelizing scheduling"""

    # --- SCHEDULING ---
    tracker: LayerBindings = bindings
    problem = SchedulingProblem(models, node_stats, noc)
    system_schedule: SystemSchedule = run_scheduling(problem, tracker, config)

    # --- VALIDATION ---
    are_constraints_fulfilled = validate_schedule(system_schedule, tracker, models)
    return ScheduleResult(system_schedule, tracker, are_constraints_fulfilled)

@ray.remote
def scheduling_worker(trial, problem_id, config_id):
    """
    Worker function: Executes the scheduling and validation for one trial.
    """
    # Pull references from shared memory (Zero-copy read)
    problem: SchedulingProblem = problem_id
    config: SchedulingConfig = config_id

    # --- SCHEDULING ---
    tracker: LayerBindings = trial.user_attrs["bindings"]

    system_schedule: SystemSchedule = run_scheduling(problem, tracker, config)

    # --- VALIDATION ---
    are_constraints_fulfilled = validate_schedule(system_schedule, tracker, problem.models)

    return ScheduleResult(system_schedule, tracker, are_constraints_fulfilled)

@ray.remote
def scoring_worker(problem_id, cost_fns_id, config_id, bindings_id):
    """
    Worker function: Given a trial, perform all cost evaluations
    """
    # Pull references from shared memory
    problem: SchedulingProblem = problem_id
    cost_fns: list[Callable[[], float]] = cost_fns_id
    config: SchedulingConfig = config_id
    tracker: LayerBindings = bindings_id

    cost_terms = []
    sched = None
    for cost_fn in cost_fns:
        if cost_fn.__name__ == "scheduling_wrapper":
            sched: ScheduleResult = cost_fn(problem.models, problem.node_stats, problem.noc, config, tracker)
            cost = sched.schedule.system_latency
        else:
            cost = cost_fn(problem.node_stats, tracker, problem.models, problem.noc)
        cost_terms.append(cost)
    return cost_terms, sched




class OptunaMultiTenancy:

    def multi_tenancy_opt(self,
                            node_stats: SystemResult,
                            models: dict[str, OnnxModel],
                            noc: NocTopology,
                            conf: ConfigHelper,
                            initial_tracker: LayerBindings,
                          ) -> tuple[list[SystemSchedule], list[SystemSchedule]]:
        """Optimize the assigment of layers to platforms a specified Optuna sampler.
        This aims to optize energy and latency as a multi-objective optimization.
        """
        optuna_out_dir = os.path.join(conf.general.work_dir, "optuna")

        optuna_cfg = conf.mt.optimizer

        match conf.mt.optimizer.sampler:
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
                raise RuntimeError(f"Unknow selection for Optuna sampler, found: {conf.mt.optimizer.sampler}")
        
        # Configuration of cost terms to be calculated
        # Includes optimization direction, function to be called and pretty-print name
        cost_term_mapping = {
            "energy":           ("minimize", "Cum. Energy"),
            "latency":          ("minimize", "Cum. Latency"),
            "gini":             ("minimize", "Gini"),
            "jain":             ("minimize", "Jain"),
            "max_min_space":    ("minimize", "Max-Min Space"),
            "std_dev":          ("minimize", "Std. dev"),
            "crit_path":        ("minimize", "Critical Path"),
            "scheduling":       ("minimize", "Makespan")
        }

        cost_names = []
        cost_fns = []
        opt_direction = []
        for cost_term in conf.mt.cost_terms:
            opt_dir, name = cost_term_mapping[cost_term]

            fn = scheduling_wrapper if cost_term == "scheduling" else get_scorer(cost_term)
            cost_names.append(name)
            cost_fns.append(fn)
            opt_direction.append(opt_dir)

            
        # Setup information for optuna. For every layer, sample from available 
        # platform ids to determine on which platform it should be executed
        platform_ids = conf.system.platform_ids
        work_layer_tracker = copy.deepcopy(initial_tracker)
        num_trials = conf.mt.optimizer.n_trials
        binding_granularity = conf.mt.binding_granularity
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _suggest_task_bindings(trial: optuna.trial.Trial) -> list[int]:
            bindings: list[int] = []
            match binding_granularity:
                case "tile":
                    for i, (nw, layer, current_binding) in enumerate(work_layer_tracker):
                        tile_bind = trial.suggest_categorical(f"tile_binding_{i}", 
                                                               choices=platform_ids)
                        bindings.append(tile_bind)
                case "network":
                    network_bind = {}
                    for i, nw in enumerate(models.keys()):
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
                cost = cost_fn(node_stats=node_stats, tracker=work_layer_tracker, models=models, noc=noc)
                cost_values.append(cost)

            return cost_values

        # Start the optimization, setup everything for Optuna run
        # We only create the study here, the optimization is performed later depending on configuration:
        #   1) Perform scheduling for ALL trials, use ask-and-tell interface
        #   2) Perform scheduling only for pareto optimal task allocations
        os.makedirs(optuna_out_dir, exist_ok=True)

        results: list[SystemSchedule] = []
        invalid_results: list[SystemSchedule] = []

        study = optuna.create_study(sampler=sampler, directions=opt_direction, 
                                    study_name="optuna_allocation_opt",) # storage=storage_db

        
        # #########################################################
        # ------------------ WIP Ray integration ------------------
        # #########################################################

        # 1. Initialize Ray
        scheduling_start_time = time.perf_counter()
        num_cpus = max(conf.mt.num_jobs, 1)
        max_in_flight = 2 * num_cpus
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
        
        # 2. Put static objects in the Shared Memory Object Store
        problem_ref = ray.put(SchedulingProblem(models, node_stats, noc))
        config_ref = ray.put(conf.mt.scheduling)
        cost_fns_ref = ray.put(cost_fns)


        if "scheduling" in conf.mt.cost_terms:
            # Scheduling must be evaluated for every Optuna trial
            # Pipelined implementation for ray: whenever a task finishes we immediately add a new one
            # Scheduling is very compute intesive, so we try to minimize runtime by doing so.
            task_mapping: dict[ray.ObjectRef, optuna.Trial] = {} 
            all_schedules: dict[int, str] = {}
            finished_count = 0
            pbar = tqdm.tqdm(total=num_trials, desc="Pipelined Optimization")

            # A. Prime the pipeline: Fill all CPU cores with initial tasks
            for _ in range(conf.mt.num_jobs):
                if finished_count + len(task_mapping) < num_trials:
                    trial = study.ask()
                    bindings = _suggest_task_bindings(trial)
                    work_layer_tracker.set_all_bindings(bindings)
                    bindings_ref = ray.put(copy.deepcopy(work_layer_tracker))
                    
                    # Dispatch to Ray immediately
                    t_id = scoring_worker.remote(
                        problem_ref,
                        cost_fns_ref,
                        config_ref,
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
                            problem_ref,
                            cost_fns_ref,
                            config_ref,
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
                    results.append(res.schedule)
                elif res:
                    invalid_results.append(res.schedule)



        else:
            # Scheduling is only performed for Pareto-optimal task allocations
            study.optimize(objective, n_trials=num_trials, n_jobs=1, show_progress_bar=True)
            bindings_to_schedule = study.best_trials
            results_dir = Path(optuna_out_dir)
            final_results: list[ScheduleResult] = []

            trial_iter = enumerate(bindings_to_schedule)
            in_flight: dict = {}   # ObjectRef -> trial index

            # prime the pipeline
            for idx, trial in itertools.islice(trial_iter, max_in_flight):
                ref = scheduling_worker.remote(trial, problem_ref, config_ref)
                in_flight[ref] = idx

            result_paths: dict[int, Path] = {}

            with tqdm.tqdm(total=len(bindings_to_schedule), desc="Finalizing Schedules") as pbar:
                while in_flight:
                    done, _ = ray.wait(list(in_flight.keys()), num_returns=1)
                    obj_ref = done[0]
                    idx = in_flight.pop(obj_ref)   # remove and recover the correct index

                    try:
                        result = ray.get(obj_ref)
                        path = results_dir / f"result_{idx:04d}.pkl"
                        with open(path, "wb") as f:
                            pickle.dump(result, f)
                        result_paths[idx] = path
                    except Exception as e:
                        print(f"WORKER ERROR: {e}")
                    finally:
                        del obj_ref
                        pbar.update(1)

                    next_item = next(trial_iter, None)
                    if next_item is not None:
                        next_idx, next_trial = next_item
                        ref = scheduling_worker.remote(next_trial, problem_ref, config_ref)
                        in_flight[ref] = next_idx
            
            # Shutdown Ray when done to free up shared memory
            ray.shutdown()

            for idx in sorted(result_paths.keys()):
                with open(result_paths[idx], "rb") as f:
                    res = pickle.load(f)
                if res.is_valid:
                    results.append(res.schedule)
                else:
                    invalid_results.append(res.schedule)
                result_paths[idx].unlink(missing_ok=True)

            scheduling_end_time = time.perf_counter()
            logger.info(f"Scheduling runtime: {scheduling_end_time-scheduling_start_time}")
            
            #for res in final_results:
            #    if res.is_valid:
            #        results.append(res.schedule)
            #    else:
            #        invalid_results.append(res.schedule)
            # #########################################################
            # ------------------ END Ray integration ------------------
            # #########################################################



        logger.debug(f"Multi-Tenancy optimization complete, found {len(study.best_trials)} Parto-optimal solutions")
        
        self.plot_results(conf, study, cost_names)
        deadline_evaluation(models, results, invalid_results)

        return results, invalid_results

    def plot_results(self, conf: ConfigHelper, study: optuna.Study, cost_names: list[str]):
        # Plotting to make sense of the results
        plot_dir = os.path.join(conf.general.work_dir, "figures")

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

        for idx, target in enumerate(cost_names):
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
        