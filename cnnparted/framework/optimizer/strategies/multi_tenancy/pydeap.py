from dataclasses import dataclass
from collections.abc import Callable
from functools import partial
import array
import statistics
import itertools as it
import copy
import os
import cProfile

import numpy as np
import tqdm
import ray
import cloudpickle
from loguru import logger
import plotly.graph_objects as go
import random

from deap import base, creator, tools, algorithms

from framework.model.layer_tracker import LayerBindings
from framework.noc.noc import NocTopology
from framework.model.onnx_model import OnnxModel
from framework.node.node_evaluator import SystemResult
from framework.scheduling import (validate_schedule, SystemSchedule)

from framework.helpers.config_helper import ConfigHelper
from .common import scoring_dispatch
from .optuna import ScheduleResult


@ray.remote
def scheduling_worker(binding, models_id, node_stats_id, noc_id, priority_id, scheduling_algorithm):
    """
    Worker function: Executes the scheduling and validation for one trial.
    """
    # Pull references from shared memory (Zero-copy read)
    base_models = models_id
    node_stats = node_stats_id
    noc = noc_id
    scheduling_priority = priority_id

    # --- SCHEDULING ---
    tracker: LayerBindings = binding

    system_schedule: SystemSchedule = scheduling_algorithm(
        base_models, tracker, node_stats, noc, scheduling_priority
    )

    # --- VALIDATION ---
    are_constraints_fulfilled = validate_schedule(system_schedule, tracker, base_models)

    return ScheduleResult(system_schedule, tracker, are_constraints_fulfilled)


class PydeapMultiTenancy:
    def multi_tenancy_opt(self,
                            node_stats: SystemResult,
                            models: dict[str, OnnxModel],
                            noc: NocTopology,
                            conf: ConfigHelper,
                            initial_tracker: LayerBindings,
                            scheduling_algorithm: Callable[..., SystemSchedule],
                          ) -> tuple[list[SystemSchedule], list[SystemSchedule]]:

        # Notes:
        # - create a class of "name", inheriting for <base>
        # - Creates attributes for this class
        num_costs = len(conf.mt.cost_terms)
        creator.create("Fitness", base.Fitness, weights=list(it.repeat(-1.0, num_costs)))
        creator.create("Individual", array.array, typecode="I", fitness=creator.Fitness)

        num_attrs = len(initial_tracker)
        num_indiv = conf.mt.optimizer.population_size
        num_generations = conf.mt.optimizer.num_gen
        prob_crossover = conf.mt.optimizer.crossover_prob
        prob_mut = conf.mt.optimizer.mutation_prob

        para_mu = int(num_indiv // 2) #num of indiv taken from prev generation
        para_lambda = num_indiv #num indiv in generation

        def suggest_valid_allocation():
            return random.choice(node_stats.get_platform_ids())
        
        toolbox = base.Toolbox()
        toolbox.register("attr_binding", suggest_valid_allocation)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_binding, n=num_attrs)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def binding_mutation(individual, valid_bindings, indpb):
            """Mutate an individual by replacing attributes with probability *indpb*,
            with a random choice drawn from all valid bindings.
            Modified version of mutUniformInt

            :param individual: Individual to be mutated
            :param valid_binddings: Values form which the bindings will be chosen
            :returns: A tuple of one individual
            """

            size = len(individual)
            for i in range(size):
                if random.random() < indpb:
                    individual[i] = random.choice(valid_bindings)
            return individual,

        # Setup for evaluation
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)

        cost_fns = []
        for idx, cost_term in enumerate(conf.mt.cost_terms):
            fn = partial(scoring_dispatch, cost_term)
            cost_fns.append(fn)

            stats.register(f"min_{cost_term}", lambda values, i=idx: np.min([v[i] for v in values]))
            stats.register(f"med_{cost_term}", lambda values, i=idx: np.median([v[i] for v in values]))


        def score_allocation(ind: list, cost_fns: list[Callable]):
            cost_terms = []
            initial_tracker.set_all_bindings(ind),
            for cost_fn in cost_fns:
                cost = cost_fn(node_stats, initial_tracker,  models, noc)
                cost_terms.append(cost) 
            return cost_terms

        toolbox.register("mate", tools.cxUniform, indpb=prob_crossover)
        toolbox.register("mutate", binding_mutation, valid_bindings=node_stats.get_platform_ids(), indpb=prob_mut)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", score_allocation, cost_fns=cost_fns)

        pop = toolbox.population(n=num_indiv)
        pareto = tools.ParetoFront()

        _, log = algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu=para_mu,
            lambda_=para_lambda,
            cxpb=prob_crossover,
            mutpb=prob_mut,
            ngen=num_generations,
            stats=stats,
            halloffame=pareto,
            verbose=False,
        )
        
        self.plot_statistics(conf, log)
        self.plot_pareto_front(conf, pareto)

        # Perform scheduling for pareto optimal solutions
        ray.init(num_cpus=max(conf.mt.num_jobs, 1), ignore_reinit_error=True)
        results: list[SystemSchedule] = []
        invalid_results: list[SystemSchedule] = []

        # Put static objects in the Shared Memory Object Store
        models_ref = ray.put(models)
        node_stats_ref = ray.put(node_stats)
        noc_ref = ray.put(noc)
        priority_ref = ray.put(conf.mt.scheduling_priority)

        task_ids = [
            scheduling_worker.remote(
                copy.deepcopy(initial_tracker).set_all_bindings(binding), 
                models_ref, 
                node_stats_ref, 
                noc_ref, 
                priority_ref, 
                scheduling_algorithm
            )
            for binding in pareto
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

        for res in final_results:
            if res.is_valid:
                results.append(res.schedule)
            else:
                invalid_results.append(res.schedule)
        return results, invalid_results
        

    def plot_statistics(self, conf: ConfigHelper, log: tools.Logbook):
        # Plotting of all statistics over trials
        for hdr in log.header[2:]:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            data = log.select(hdr)
            fig.add_trace(go.Scatter(y=data, mode="lines", name=hdr))
            fig.update_yaxes(
                title=hdr,
                title_font_size=20,
            )
            fig.update_xaxes(
                title="Num. Gen.",
                title_font_size=20,
            )
            fig.write_image(os.path.join(conf.general.work_dir, "figures", f"{hdr}.pdf"))

    
    def plot_pareto_front(self, conf: ConfigHelper, pareto: tools.ParetoFront):
        # Plotting of pareto front
        fitness_values = [ind.fitness.values for ind in pareto]
        num_objectives = len(fitness_values[0]) if fitness_values else 0
        
        if num_objectives == 2:
            # 2D Scatter plot
            obj1 = [f[0] for f in fitness_values]
            obj2 = [f[1] for f in fitness_values]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=obj1, y=obj2,
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Pareto Front"
            ))
            fig.update_layout(
                title="Pareto Front",
                xaxis_title=conf.mt.cost_terms[0],
                yaxis_title=conf.mt.cost_terms[1],
                template="plotly_white",
            )
        elif num_objectives == 3:
            # 3D Scatter plot
            obj1 = [f[0] for f in fitness_values]
            obj2 = [f[1] for f in fitness_values]
            obj3 = [f[2] for f in fitness_values]
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=obj1, y=obj2, z=obj3,
                mode="markers",
                marker=dict(size=8, color="red"),
                name='Pareto Front'
            ))
            fig.update_layout(
                title="Pareto Front",
                scene=dict(
                    xaxis_title=conf.mt.cost_terms[0],
                    yaxis_title=conf.mt.cost_terms[1],
                    zaxis_title=conf.mt.cost_terms[2],
                ),
                template="plotly_white",
            )
        else:
            # Parallel coordinates plot for 4D+
            dimensions = [
                dict(label=conf.mt.cost_terms[i], values=tuple(f[i] for f in fitness_values))
                for i in range(num_objectives)
            ]
            fig = go.Figure(data=
                go.Parcoords(
                    dimensions=dimensions,
                    line=dict(color='red', showscale=False),
                )
            )
            fig.update_layout(
                title="Pareto Front",
                template="plotly_white",
            )
        
        fig.write_image(os.path.join(conf.general.work_dir, "figures", "pareto_front.pdf"))