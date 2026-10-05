from framework.helpers.config.stages_config import (
    ListSchedulingConfig, MultiQueueSchedulingConfig, HeraldConfig,
    SatSolverSchedulingConfig, IlpSchedulingConfig, RlSchedulingConfig, GpSchedulingConfig,
)
from framework.model.layer_tracker import LayerBindings
from .schedules import SystemSchedule
from .utils import SchedulingProblem
from .list_scheduling import list_scheduling
from .multi_queue import multi_queue_scheduling
from .herald_like import herald_like
from .sat_solving import sat_solver_scheduling
from .ilp_scheduling import ilp_scheduling
from .reinforcement_learning import rl_scheduling
from .genetic_programming import gp_scheduling

SchedulingConfig = (ListSchedulingConfig | MultiQueueSchedulingConfig | HeraldConfig
                   | SatSolverSchedulingConfig | IlpSchedulingConfig | RlSchedulingConfig | GpSchedulingConfig)


def run_scheduling(problem: SchedulingProblem, bindings: LayerBindings, config: SchedulingConfig) -> SystemSchedule:
    """Uniform entry point for every scheduling algorithm in this package
    """
    match config:
        case ListSchedulingConfig():
            return list_scheduling(problem.models, bindings, problem.node_stats, problem.noc, config.priority)
        case MultiQueueSchedulingConfig():
            return multi_queue_scheduling(problem.models, bindings, problem.node_stats, problem.noc, config.priority)
        case HeraldConfig():
            # herald_like determines its own layer-to-platform bindings dynamically (via EDP), so
            # unlike every other algorithm here it never reads `bindings` at all.
            return herald_like(problem.models, problem.node_stats, problem.noc, config.lbf)
        case SatSolverSchedulingConfig():
            return sat_solver_scheduling(problem.models, bindings, problem.node_stats, problem.noc,
                                         num_search_workers=config.num_search_workers,
                                         max_time_in_seconds=config.max_time_in_seconds)
        case IlpSchedulingConfig():
            return ilp_scheduling(problem.models, bindings, problem.node_stats, problem.noc,
                                  objective=config.objective, hard_rt=config.hard_rt,
                                  solver_name=config.solver, max_time_in_seconds=config.max_time_in_seconds)
        case RlSchedulingConfig():
            return rl_scheduling(problem.models, bindings, problem.node_stats, problem.noc,
                                 total_timesteps=config.total_timesteps, ready_window=config.ready_window,
                                 seed=config.seed, save_path=config.save_path, load_path=config.load_path)
        case GpSchedulingConfig():
            return gp_scheduling(problem.models, bindings, problem.node_stats, problem.noc,
                                 population_size=config.population_size, generations=config.generations,
                                 max_tree_depth=config.max_tree_depth, cxpb=config.cxpb, mutpb=config.mutpb,
                                 seed=config.seed, verbose=config.verbose,
                                 save_path=config.save_path, load_path=config.load_path)
        case _:
            raise RuntimeError(f"Unknown scheduling config type: {type(config)}")
