import copy
import os

import numpy as np
import plotly.express as px

from .common import _score_edp, _score_idle_times, _score_max_latency

from framework.model.layer_tracker import LayerBindings
from framework.scheduling import SystemSchedule
from framework.scheduling.dispatch import run_scheduling
from framework.scheduling.utils import SchedulingProblem
from framework.helpers.config_helper import ConfigHelper
from framework.model.onnx_model import OnnxModel
from framework.noc.noc import NocTopology
from framework.node.node_evaluator import SystemResult

class SaMultiTenancy:
    def multi_tenancy_opt(self,
                            node_stats: SystemResult,
                            models: dict[str, OnnxModel],
                            noc: NocTopology,
                            conf: ConfigHelper,
                            initial_tracker: LayerBindings,
                          ) -> tuple[list[SystemSchedule], list[SystemSchedule]]:
        """Optimize the assigment of layers to platforms using simulated annealing.
        This optimizer only supports single-objective optimization."""

        problem = SchedulingProblem(models, node_stats, noc)
        initial_schedule = run_scheduling(problem, initial_tracker, conf.mt.scheduling)
        num_nodes = initial_tracker.get_num_layers()
        initial_cost = self._calc_cost(initial_tracker, initial_schedule)
        num_iter = num_nodes * 100
        t_start = initial_cost*8
        convergence_counter = 0
        cost_history = []

        node_ids = node_stats.get_platform_ids()
        old_cost = initial_cost
        best_cost = np.inf
        current_state = copy.deepcopy(initial_tracker)
        improvement_counter = convergence_counter
        convergence = False
        rng = np.random.default_rng()

        # Default init best values
        best_state = current_state
        best_schedule = initial_schedule

        def _random_modify(current_state: LayerBindings) -> LayerBindings:
            layer_tracker = copy.deepcopy(current_state)
            action_space = [layer_tracker.swap_layers, layer_tracker.move_layer, layer_tracker.swap_platforms]
            action = rng.choice(a=action_space, p=[0.45, 0.45, 0.1])
            match action:
                case layer_tracker.swap_layers:
                    platforms_to_swap = rng.choice(node_ids, size=2, replace=False)
                    layers_to_swap = []
                    for platform in platforms_to_swap:
                        layers = np.array(layer_tracker.get_all_layers_on_platform(platform))
                        if layers.size == 0:
                            return layer_tracker
                        layers_to_swap.append(rng.choice(layers))
                    action(layers_to_swap)
                case layer_tracker.move_layer:
                    # Move one layer to other platform
                    platform_from, platform_to = rng.choice(node_ids, size=2, replace=False)
                    layer_candidates = np.array(layer_tracker.get_all_layers_on_platform(platform_from))
                    if layer_candidates.size == 0:
                        return layer_tracker
                    layer_to_move = rng.choice(layer_candidates)
                    action(layer_to_move, platform_to)
                case layer_tracker.swap_platforms:
                    platform_a, platform_b = rng.choice(node_ids, size=2, replace=False)
                    action(platform_a, platform_b)
            return layer_tracker

        while not convergence:
            solution_improved = False
            for i in range(num_iter):
                t = t_start / (i+1)
                
                # Modify the binding and score it
                new_state = _random_modify(current_state)
                node_schedules: SystemSchedule = run_scheduling(problem, new_state, conf.mt.scheduling)
                new_cost = self._calc_cost(new_state, node_schedules)
                
                if rng.random() <= np.exp( -(new_cost - old_cost) / t ):
                    current_state = new_state
                    old_cost = new_cost
                    
                    # Check for new optimum
                    cost_history.append(new_cost)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_state = copy.deepcopy(new_state)
                        best_schedule: SystemSchedule = copy.deepcopy(node_schedules)
                        solution_improved = True
            
            #Check convergence: No better solution in three cooling cycles
            if solution_improved:
                improvement_counter = convergence_counter
            else:
                improvement_counter -= 1

            if improvement_counter <= 0:
                convergence = True
            
        fig = px.line(cost_history, title="EDP History for SA")
        fig.update_yaxes(title_text="EDP")
        fig.update_xaxes(title_text="Iteration")
        fig.write_html(os.path.join(conf.general.work_dir, "edp_history_sa.html")) #TODO: Correct dir

        # Result is a list for compatibility with Multi-Objective Opt
        return [best_schedule], [] #TODO

    def _calc_cost(self, tracker: LayerBindings, schedules: SystemSchedule) -> float:
        edp_cost =  _score_edp(tracker)
        util_cost = _score_idle_times(schedules)
        lat_cost =  _score_max_latency(schedules)

        return edp_cost**0.4 +  util_cost**2.5 + lat_cost*2