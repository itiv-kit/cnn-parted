from math import inf
import os
import copy
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import optuna
import tqdm

from framework.model.onnx_model import OnnxModel
from framework.model.layers.compute_layer import ComputationLayer
from framework.optimizer.optimizer import Optimizer
from framework.graph_analyzer import GraphAnalyzer
from framework.helpers.config_helper import ConfigHelper
from framework.node.node_evaluator import SystemResult
from framework.scheduling import list_scheduling, multi_queue_scheduling, SystemSchedule


class MultiTenancyPartitioning(Optimizer):
    def __init__(self, work_dir: str, node_stats: SystemResult, ga: GraphAnalyzer, global_conf: ConfigHelper):
        self.work_dir = work_dir
        self.plot_dir = os.path.join(work_dir, "figures")
        self.ga = ga
        self.conf = global_conf
        self.system_result = node_stats

        os.makedirs(self.plot_dir, exist_ok=True)

    def optimize(self):
        self._gen_initial_binding()

        algorithm = MultiTenancyProblem(self.work_dir, self.system_result, self.ga, self.conf)
        solutions = algorithm.minimize()
        return solutions

    def _gen_initial_binding(self):
        node_ids = self.system_result.get_platform_ids()
        for network, model in self.ga.models.items():
            for layer in model.nodes:
                layer.scheduling.binding = np.random.choice(node_ids)


@dataclass
class LayerTracker:
    """
    This class is used to keep track of all layers of all networks that are being evaluated. 
    For each layer, it keeps the name of the layer and the binding, i.e. the assignment to a platform.
    Detailed layer stats, like energy or latency, must be retrieved from the SystemResult class.
    """
    network_info: dict[str, dict[str, int]]

    def __iter__(self):
        return iter([ (nw,layer_name,binding) for nw, layers in self.network_info.items() for layer_name, binding in layers.items()])

    def get_all_layers_on_platform(self, platform: int) -> list[tuple[str, str]]:
        nw_layer = [ (nw,layer_name) for nw, layers in self.network_info.items() for layer_name, binding in layers.items() if binding == platform]
        return nw_layer
    
    def get_all_layer_names(self) -> list[tuple[str, str]]:
        nw_layer = [ (nw,layer_name) for nw, layers in self.network_info.items() for layer_name, binding in layers.items()]
        return nw_layer
    
    def get_num_layers(self) -> int:
        layer_ctr = [ len(layers) for layers in self.network_info.values() ]
        return sum(layer_ctr)
    
    def set_all_bindings(self, new_bindings: list[int]):
        """Takes a list[int] where each entry denotes the platform on which a layer should be executed"""
        idx = 0
        for nw, layers in self.network_info.items():
            for layer in layers:
                self.network_info[nw][layer] = new_bindings[idx]
                idx += 1

    def swap_layers(self, nws_layers: list[tuple[str,str]]):
        """Swap the binding of two layers that are on different platforms"""
        bindings = np.array([self.network_info[nw][layer] for nw, layer in nws_layers])
        (first_nw, first_layer), (second_nw, second_layer) = nws_layers
        first_b, second_b = bindings
        self.network_info[first_nw][first_layer] = second_b
        self.network_info[second_nw][second_layer] = first_b
    
    def move_layer(self, nw_layer: tuple[str,str], to: int):
        """Move one layer to a specified platform"""
        nw, layer = nw_layer
        self.network_info[nw][layer] = to
    
    def swap_platforms(self, platform_a: int, platform_b: int):
        """Swap all layers assigned to two differnt platforms"""
        for nw, layer_bindings in self.network_info.items():
            for layer_name, binding in layer_bindings.items():
                if binding == platform_a:
                    self.network_info[nw][layer_name] = platform_b
                elif binding == platform_b:
                    self.network_info[nw][layer_name] = platform_a


class MultiTenancyProblem():
    def __init__(self, work_dir: str, node_stats: SystemResult, ga: GraphAnalyzer, global_conf: ConfigHelper):
        self.work_dir = work_dir
        self.plot_dir = os.path.join(work_dir, "figures")
        self.optuna_out_dir = os.path.join(work_dir, "optuna")
        self.node_stats = node_stats
        self.node_ids = node_stats.get_platform_ids()
        self.ga = ga
        self.models = copy.deepcopy(ga.models)
        self.conf = global_conf
        match self.conf.mt.scheduling:
            case "list":
                self.scheduling_algorithm = list_scheduling
            case "multi_queue":
                self.scheduling_algorithm = multi_queue_scheduling
        
        # Parse execution graphs into LayerTracker
        network_info = defaultdict(lambda: defaultdict(int))
        for network, model in ga.models.items():
            for layer in model.nodes:
                if isinstance(layer, ComputationLayer):
                    network_info[network][layer.layer_name] = layer.scheduling.binding

        self.layer_tracker = LayerTracker(network_info)
        self.num_nodes = self.layer_tracker.get_num_layers()
        self.initial_schedule = self.scheduling_algorithm(copy.deepcopy(ga.models), self.node_stats, priority=self.conf.mt.scheduling_priority)

        # SA configuration
        self.initial_cost = self._calc_cost(self.layer_tracker, self.initial_schedule)
        self.num_iter = self.num_nodes * 100
        self.t_start = self.initial_cost*8
        self.convergence_counter = 0
        self.cost_history = []
    
    def _simulated_annealing(self) -> list[tuple[LayerTracker, SystemSchedule]]:
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

        def _random_modify(self, current_state: LayerTracker):
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

                new_models = copy.deepcopy(self.models)
                self._write_binding_to_graph(new_state, new_models)
                node_schedules = self.scheduling_algorithm(new_models, self.node_stats, priority=self.conf.mt.scheduling_priority)
                
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

    
    def _optuna_optimizer(self) -> list[tuple[LayerTracker, SystemSchedule]]:
        """Optimize the assigment of layers to platforms a specified Optuna sampler.
        This aims to optize energy and latency as a multi-objective optimization.
        """

        match self.conf.mt.optimizer.sampler:
            case "grid":
                sampler = optuna.samplers.GridSampler()
            case "tpe":
                sampler = optuna.samplers.TPESampler()
            case "cmaes":
                sampler = optuna.samplers.CmaEsSampler()
            case "nsga2":
                sampler = optuna.samplers.NSGAIISampler(seed=42)
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
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.trial.Trial):
            bindings = []
            for i, (nw, layer, current_binding) in enumerate(work_layer_tracker):
                layer_bind = trial.suggest_categorical(f"binding_{i}", 
                                                       choices=platform_ids)
                bindings.append(layer_bind)
            work_layer_tracker.set_all_bindings(bindings)

            # Add information to be later retrieved
            trial.set_user_attr("bindings", copy.deepcopy(work_layer_tracker))

            return (self._score_total_energy(work_layer_tracker), 
                    self._score_total_latency(work_layer_tracker),)

        # Start the optimization, setup everything for Optuna run
        os.makedirs(self.optuna_out_dir, exist_ok=True)
        study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize",], )
        study.optimize(objective, n_trials=num_trials, n_jobs=1, show_progress_bar=True)
        
        # Plotting to make sense of the results
        #fig: go.Figure = optuna.visualization.plot_pareto_front(study, target_names=["Energy", "Latency"], 
        #                                                        targets=lambda t: (t.values[0], t.values[1]),
        #                                                        include_dominated_trials=False)
        #file_name = "pareto_points_optuna"
        #fig.write_html(os.path.join(self.plot_dir, file_name+".html"))
        #fig.write_image(os.path.join(self.plot_dir, file_name+".png"))

        for idx, target in [(0, "Energy"), (1, "Latency")]:
            fig: go.Figure = optuna.visualization.plot_optimization_history(study, target_name=target, target=lambda t: t.values[idx])
            file_name = f"history_{target.lower()}_optuna"
            fig.write_html(os.path.join(self.plot_dir, file_name+".html"))
            fig.write_image(os.path.join(self.plot_dir, file_name+".png"))

        logger.debug(f"Multi-Tenancy optimization complete, found {len(study.best_trials)} Parto-optimal solutions")
        logger.info("Performing scheduling of pareto-optimal bindings")

        ### Perform scheduling for pareto-optimal solutions
        results: list[tuple[LayerTracker, SystemSchedule]] = []
        for trial in tqdm.tqdm(study.best_trials):
            new_models = copy.deepcopy(self.models)
            self._write_binding_to_graph(trial.user_attrs["bindings"], new_models)
            node_schedules = self.scheduling_algorithm(new_models, self.node_stats, priority=self.conf.mt.scheduling_priority)
            results.append( (trial.user_attrs["bindings"], node_schedules) )
        
        
        # For every network, figure out at which time it finishes execution
        # Needed to check if deadlines have been met or not
        deadline_stats = {network: [0,0] for network in self.models.keys()}
        all_succeeded = []
        for _, schedule in results:
            all_succeeded.append(True)
            network_finishes = schedule.get_network_finish_times()
            for nw, end in network_finishes.items():
                if (deadline := self.models[nw].deadline) and deadline != inf:
                    if end <= deadline:
                        #success
                        deadline_stats[nw][0] += 1
                    else:
                        #failure
                        all_succeeded[-1] = False
                        deadline_stats[nw][1] += 1
        logger.info(f"All deadlines were wet in {all_succeeded.count(True)}/{len(all_succeeded)} cases ({all_succeeded.count(True)/len(all_succeeded)*100:.2f}%)")
        for nw, (success, fail) in deadline_stats.items():
            if self.models[nw].deadline != inf:
                logger.info(f"Deadline for {nw} was met in {success}/{fail+success} cases ({success/(fail+success)*100:.2f}%)")
        
        # Log some statistics
        system_runtimes: list[float] = []
        system_energy: list[float] = []
        for bindings, schedule in results:
            system_runtimes.append( max([sched.cur_time for sched in schedule.node_schedules.values()]) )
            system_energy.append(self._score_total_energy(bindings))
        
        min_runtime_idx = system_runtimes.index(min(system_runtimes))
        min_runtime_latency = system_runtimes[min_runtime_idx]
        min_runtime_energy = system_energy[min_runtime_idx]
        min_energy_idx = system_energy.index(min(system_energy))
        min_energy_latency = system_runtimes[min_energy_idx]
        min_energy_energy = system_energy[min_energy_idx]
        logger.info(f"Configuration with minimum system runtime (ID: {min_runtime_idx}):\n\tRuntime: {min_runtime_latency}\n\tEnergy: {min_runtime_energy}")
        logger.info(f"Configuration with minimum system energy (ID: {min_energy_idx}):\n\tRuntime: {min_energy_latency}\n\tEnergy: {min_energy_energy}")
            
        return results

    
    def minimize(self):
        # Distribute workload between platforms
        logger.debug(f"Optimizing multitenancy problem with {self.conf.mt.algorithm}")
        if self.conf.mt.algorithm == "optuna":
            solutions = self._optuna_optimizer()
        elif self.conf.mt.algorithm == "sa":
            solutions =  self._simulated_annealing()

        # Move binding to graphs
        #for solution in solutions:
        #    for network, layer, binding in solution:
        #        self.execution_graphs[network].reduced_graph.nodes(data=True)[layer]["binding"] = binding
        
        # For node_stats: when a layer was not bound to a specific platform, reset scheduling information
        #for platform, data in self.node_stats.platforms.items():
        #    for nw_name, nw_data in data.designs[0].networks.items():
        #        for layer_name, layer_data in nw_data.layers.items():
        #            binding = best_solution.network_info[nw_name][layer_name]
        #            if binding != platform:
        #                layer_data.reset_scheduling()


        #TODO: Between binding and scheduling, the design must be fixed and a new node_stats must be created with the single design

        return solutions

    def _score_edp(self, tracker: LayerTracker) -> float:
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
    
    def _score_total_energy(self, tracker: LayerTracker) -> float:
        # For each layer we want to find the most efficient platform
        total_energy = 0.0    
        for nw, layer, bind in tracker:
            # TODO Currently assumes that there is only one design
            layer_data = self.node_stats[bind].designs[0].networks[nw].layers[layer]
            total_energy += layer_data.energy
        return total_energy

    def _score_total_latency(self, tracker: LayerTracker) -> float:
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
    
    def _score_even_latency(self, tracker: LayerTracker) -> float:
        """The total time of operations assigned to each platform should be roughly equal"""
        ideal_runtimes = {platform: 0.0 for platform in self.conf.system.platform_ids}
        for nw, layer, bind in tracker:
            # TODO Currently assumes that there is only one design
            layer_data = self.node_stats[bind].designs[0].networks[nw].layers[layer]
            ideal_runtimes[bind] += layer_data.latency
        max_runtime = max(ideal_runtimes.values())
        cost = 0.0
        for runtime in ideal_runtimes.values():
            cost += (max_runtime - runtime)
        return cost

    def _calc_cost(self, tracker: LayerTracker, schedules: SystemSchedule) -> float:
        edp_cost = self._score_edp(tracker)
        util_cost = self._score_idle_times(schedules)
        lat_cost = self._score_max_latency(schedules)

        return edp_cost**0.4 +  util_cost**2.5 + lat_cost*2
    
    def _write_binding_to_graph(self, tracker: LayerTracker, models: dict[str, OnnxModel]):
        for nw, layer, binding in tracker:
            node = models[nw].node_by_layer_name(layer)
            node.scheduling.binding = binding

