import os
import copy
import itertools as it
from collections import defaultdict
from dataclasses import dataclass
import time

import numpy as np
import tqdm
import plotly.express as px
import plotly.graph_objects as go

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from framework.optimizer.optimizer import Optimizer
from framework.optimizer.partitioning_problem import PartitioningProblem
from framework.optimizer.config.partitioning_opt_config import PartitioningOptConfig
from framework.graph_analyzer import GraphAnalyzer
from framework.helpers.config_helper import ConfigHelper
from framework.node.node_evaluator import SystemResult
from framework.model.execution_graph import ExecutionGraph
from framework.model.scheduling import SystemSchedule, list_scheduling, PlatformSchedule


# What this should do:
#   - given be a random, initial partitioning of the networks
#   - try to optimize EDP by moving partitions between accelerators

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
        for network, exec_graph in self.ga.exec_graphs.items():
            compute_graph = exec_graph.reduced_graph
            num_nodes = compute_graph.number_of_nodes()
            node_ids = self.system_result.get_platform_ids()
            initial_bindings = np.random.choice(node_ids, size=num_nodes)
            for node_name, binding in zip(compute_graph.nodes, initial_bindings):
                compute_graph.nodes[node_name]["binding"] = binding


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

    def get_all_layers_on_platform(self, platform: int):
        nw_layer = [ (nw,layer_name) for nw, layers in self.network_info.items() for layer_name, binding in layers.items() if binding == platform]
        return nw_layer
    
    def get_all_layer_names(self):
        nw_layer = [ (nw,layer_name) for nw, layers in self.network_info.items() for layer_name, binding in layers.items()]
        return nw_layer
    
    def get_num_layers(self):
        layer_ctr = [ len(layers) for layers in self.network_info.values() ]
        return sum(layer_ctr)
    
    def set_all_bindings(self, new_bindings: list[int]):
        idx = 0
        for nw, layers in self.network_info.items():
            for layer in layers:
                self.network_info[nw][layer] = new_bindings[idx]
                idx += 1

    
    # Swap the binding of two layers that are on different platforms
    def swap_layers(self, nws_layers: list[tuple[str,str]]):
        bindings = np.array([self.network_info[nw][layer] for nw, layer in nws_layers])
        (first_nw, first_layer), (second_nw, second_layer) = nws_layers
        first_b, second_b = bindings
        self.network_info[first_nw][first_layer] = second_b
        self.network_info[second_nw][second_layer] = first_b
    
    # Move one layer to another platform
    def move_layer(self, nw_layer: tuple[str,str], to: int):
        nw, layer = nw_layer
        self.network_info[nw][layer] = to
    
    # Swap ALL layers assigned to two different platforms
    def swap_platforms(self, platform_a: int, platform_b: int):
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
        self.execution_graphs = copy.deepcopy(ga.exec_graphs)
        self.rng = np.random.default_rng()
        self.conf = global_conf

        # Parse execution graphs into LayerTracker
        network_info = defaultdict(lambda: defaultdict(int))
        for network, graph in ga.exec_graphs.items():
            for node, data in graph.reduced_graph.nodes(data=True):
                network_info[network][node] = data["binding"]

        self.layer_tracker = LayerTracker(network_info)
        self.num_nodes = self.layer_tracker.get_num_layers()
        self.initial_schedule = list_scheduling(copy.deepcopy(ga.exec_graphs), self.node_stats, priority=self.conf.mt.scheduling)

        # SA configuration
        self.initial_cost = self._calc_cost(self.layer_tracker, self.initial_schedule)
        self.num_iter = self.num_nodes * 100
        self.t_start = self.initial_cost*8
        self.convergence_counter = 0
        self.cost_history = []
    
    def _simulated_annealing(self) -> list[tuple[LayerTracker, SystemSchedule]]:
        old_cost = self.initial_cost
        best_cost = np.inf
        best_state = None
        best_schedule = None
        current_state = copy.deepcopy(self.layer_tracker)
        improvement_counter = self.convergence_counter
        convergence = False

        while not convergence:
            solution_improved = False
            for i in range(self.num_iter):
                t = self.t_start / (i+1)
                new_state = self._random_modify(current_state)

                new_exec_graphs = copy.deepcopy(self.execution_graphs)
                self._write_binding_to_graph(new_state, new_exec_graphs)
                
                node_schedules = list_scheduling(new_exec_graphs, self.node_stats, priority=self.conf.mt.scheduling)
                
                new_cost = self._calc_cost(new_state, node_schedules)
                
                if self.rng.random() <= np.exp( -(new_cost - old_cost) / t ):
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

        match self.conf.mt.optimizer.sampler:
            case "grid":
                sampler = optuna.samplers.GridSampler()
            case "tpe":
                sampler = optuna.samplers.TPESampler()
            case "cmaes":
                sampler = optuna.samplers.CmaEsSampler()
            case "nsga2":
                sampler = optuna.samplers.NSGAIISampler()
            case "random":
                sampler = optuna.samplers.RandomSampler()
            case "qmc":
                sampler = optuna.samplers.QMCSampler()
            case "gp":
                sampler = optuna.samplers.GPSampler()
            case _:
                raise RuntimeError(f"Unknow selection for Optuna sampler, found: {self.conf.mt.optimizer.sampler}")
        
        # We want optuna to suggest continuous ints from [0, num_platforms-1], 
        # but afterwards we have to convert them to the correct id
        platform_lut = {mapping: platform_id for platform_id, mapping in zip(self.conf.system.platform_ids, range(0, self.conf.system.num_platforms)) }
        work_layer_tracker = copy.deepcopy(self.layer_tracker)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.trial.Trial):
            bindings_int = []
            for i in range(self.num_nodes):
                layer_bind = trial.suggest_int(f"binding_{i}", low=0, high=self.conf.system.num_platforms-1)
                bindings_int.append(layer_bind)

            bindings = [platform_lut[b] for b in bindings_int]
            work_layer_tracker.set_all_bindings(bindings)

            new_exec_graphs = copy.deepcopy(self.execution_graphs) #TODO How to remove deepcopy?
            self._write_binding_to_graph(work_layer_tracker, new_exec_graphs)
            node_schedules = list_scheduling(new_exec_graphs, self.node_stats, priority=self.conf.mt.scheduling)
            
            # Add information to be later retrieved
            trial.set_user_attr("schedule", node_schedules)
            trial.set_user_attr("bindings", copy.deepcopy(work_layer_tracker))

            return (self._score_edp(work_layer_tracker), 
                    self._score_utilization(node_schedules), 
                    self._score_max_latency(node_schedules))
        

        # Start the optimization, setup everything for Optuna run
        os.makedirs(self.optuna_out_dir, exist_ok=True)
        storage=JournalStorage(JournalFileBackend(file_path=os.path.join(self.optuna_out_dir,"journal.log")))
        study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize", "minimize"], )

        study.optimize(objective, n_trials=self.conf.mt.optimizer.n_trials, n_jobs=1, show_progress_bar=True)
        
        # Plotting to make sense of the results
        fig: go.Figure = optuna.visualization.plot_pareto_front(study, target_names=["EDP", "Latency"], targets=lambda t: (t.values[0], t.values[2]))
        file_name = "paretos_edp_latency_optuna"
        fig.write_html(os.path.join(self.plot_dir, file_name+".html"))

        fig_hv: go.Figure = optuna.visualization.plot_hypervolume_history(study, [0,0,0])
        file_name_hv = "hypervolume_optuna"
        fig_hv.write_html(os.path.join(self.plot_dir, file_name_hv+".html"))

        # Update the layer tracker with the results #TODO Multi-objective opt
        trial_least_idle = max(study.best_trials, key=lambda t: t.values[0])
        trial_least_idle_params = [platform_lut[b] for b in trial_least_idle.params.values()]
        work_layer_tracker.set_all_bindings(trial_least_idle_params)

        results: list[tuple[LayerTracker, SystemSchedule]] = []
        for trial in study.best_trials:
            results.append( (trial.user_attrs["bindings"], trial.user_attrs["schedule"]) )

        return results

    
    def _random_modify(self, current_state: LayerTracker):
        layer_tracker = copy.deepcopy(current_state)
        action_space = [layer_tracker.swap_layers, layer_tracker.move_layer, layer_tracker.swap_platforms]
        action = self.rng.choice(a=action_space, p=[0.45, 0.45, 0.1])

        match action:
            case layer_tracker.swap_layers:
                platforms_to_swap = self.rng.choice(self.node_ids, size=2, replace=False)
                layers_to_swap = []
                for platform in platforms_to_swap:
                    layers = np.array(layer_tracker.get_all_layers_on_platform(platform))
                    if layers.size == 0:
                        return layer_tracker
                    layers_to_swap.append(self.rng.choice(layers))
                action(layers_to_swap)

            case layer_tracker.move_layer:
                # Move one layer to other platform
                platform_from, platform_to = self.rng.choice(self.node_ids, size=2, replace=False)
                layer_candidates = np.array(layer_tracker.get_all_layers_on_platform(platform_from))
                if layer_candidates.size == 0:
                    return layer_tracker
                layer_to_move = self.rng.choice(layer_candidates)
                action(layer_to_move, platform_to)

            case layer_tracker.swap_platforms:
                platform_a, platform_b = self.rng.choice(self.node_ids, size=2, replace=False)
                action(platform_a, platform_b)

        return layer_tracker


    def minimize(self):
        # Distribute workload between platforms
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
    
    def _score_utilization(self, schedules: SystemSchedule) -> float:
        # Prevent idle times during execution
        idle_times = [sched.get_idle_time() for sched in schedules.node_schedules.values()]
        return sum(idle_times)
    
    def _score_max_latency(self, schedules: SystemSchedule) -> float:
        latencies = [sched.cur_time for sched in schedules.node_schedules.values()]
        return max(latencies)

    def _calc_cost(self, tracker: LayerTracker, schedules: SystemSchedule) -> float:
        edp_cost = self._score_edp(tracker)
        util_cost = self._score_utilization(schedules)
        lat_cost = self._score_max_latency(schedules)

        return edp_cost**0.4 +  util_cost**2.5 + lat_cost*2
    
    def _write_binding_to_graph(self, tracker: LayerTracker, exec_graphs: dict[str, ExecutionGraph]):
        for nw, layer, binding in tracker:
            exec_graphs[nw].reduced_graph.nodes[layer]["binding"] = binding

