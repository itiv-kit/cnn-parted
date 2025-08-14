import os
import copy
import itertools as it
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import tqdm
import plotly.express as px


from framework.optimizer.optimizer import Optimizer
from framework.optimizer.partitioning_problem import PartitioningProblem
from framework.optimizer.config.partitioning_opt_config import PartitioningOptConfig
from framework.graph_analyzer import GraphAnalyzer
from framework.helpers.config_helper import ConfigHelper
from framework.node.node_evaluator import SystemResult
from framework.model.execution_graph import ExecutionGraph
from framework.model.scheduling import list_scheduling, PlatformSchedule


# What this should do:
#   - given be a random, initial partitioning of the networks
#   - try to optimize EDP by moving partitions between accelerators

class MultiTenancyPartitioning(Optimizer):
    def __init__(self, work_dir: str, node_stats: SystemResult, ga: GraphAnalyzer):
        self.work_dir = work_dir
        self.plot_dir = os.path.join(work_dir, "figures")
        self.ga = ga
        self.system_result = node_stats

        os.makedirs(self.plot_dir, exist_ok=True)

    def optimize(self):
        self._gen_initial_binding() #not it

        algorithm = MultiTenancyProblem(self.work_dir, self.system_result, self.ga) #not it
        node_stats = algorithm.minimize()
        return node_stats

    
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
    network: list[str]
    layer: dict[str, int]

    def __iter__(self):
        return zip(self.network, list(self.layer.keys()), list(self.layer.values()))

    def get_all_on_platform(self, node_id: int):
        found = [binding == node_id for binding in self.layer.values()]
        return found
    
    def get_layer_names(self):
        return np.array(list(self.layer.keys()))
    
    # Swap the binding of two layers that are on different platforms
    def swap_layers(self, layers: list[str]):
        bindings = np.array([self.layer[layer] for layer in layers])
        first_l, second_l = layers
        first_b, second_b = bindings
        self.layer[first_l] = second_b
        self.layer[second_l] = first_b
    
    # Move one layer to another platform
    def move_layer(self, layer: str, to: int):
        self.layer[layer] = to
    
    # Swap ALL layers assigned to two different platforms
    def swap_platforms(self, platform_a: int, platform_b: int):
        for name, binding in self.layer.items():
            if binding == platform_a:
                self.layer[name] = platform_b
            elif binding == platform_b:
                self.layer[name] = platform_a


class MultiTenancyProblem():
    def __init__(self, work_dir: str, node_stats: SystemResult, ga: GraphAnalyzer):
        self.work_dir = work_dir
        self.plot_dir = os.path.join(work_dir, "figures")
        self.node_stats = node_stats
        self.node_ids = node_stats.get_platform_ids()
        self.ga = ga
        self.execution_graphs = copy.deepcopy(ga.exec_graphs)
        self.rng = np.random.default_rng()

        # Parse execution graphs into LayerTracker
        networks = []
        layers = defaultdict(int)
        for network, graph in ga.exec_graphs.items():
            for node, data in graph.reduced_graph.nodes(data=True):
                networks.append(network)
                layers[node] = data["binding"]

        self.layer_tracker = LayerTracker(networks, layers)
        self.num_nodes = len(layers)
        _, self.initial_schedule = list_scheduling(copy.deepcopy(ga.exec_graphs), self.node_stats, writeback_layer_info=False)

        # SA configuration
        self.initial_cost = self._calc_cost(self.layer_tracker, self.initial_schedule)
        self.num_iter = self.num_nodes * 100
        self.t_start = self.initial_cost*8
        self.convergence_counter = 0
        self.cost_history = []
    
    def _simulated_annealing(self) -> tuple[int, LayerTracker]:
        old_cost = self.initial_cost
        best_cost = np.inf
        best_state = None
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
                
                _, node_schedules = list_scheduling(new_exec_graphs, self.node_stats, writeback_layer_info=False)
                
                new_cost = self._calc_cost(new_state, node_schedules)
                
                if self.rng.random() <= np.exp( -(new_cost - old_cost) / t ):
                    current_state = new_state
                    old_cost = new_cost
                    
                    # Check for new optimum
                    self.cost_history.append(new_cost)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_state = copy.deepcopy(new_state)
                        solution_improved = True
            
            #Check convergence: No solution in three cooling cycles
            if solution_improved:
                improvement_counter = self.convergence_counter
            else:
                improvement_counter -= 1

            if improvement_counter <= 0:
                convergence = True
            
        print(self.node_stats[0].designs[0].networks["alexnet"].layers['/features/features.0/Conv'])
        fig = px.line(self.cost_history, title="EDP History for SA")
        fig.update_yaxes(title_text="EDP")
        fig.update_xaxes(title_text="Iteration")
        fig.write_html(os.path.join(self.plot_dir, "edp_history_sa.html"))

        return best_cost, best_state

    
    def _random_modify(self, current_state: LayerTracker):
        layer_tracker = copy.deepcopy(current_state)
        action_space = [layer_tracker.swap_layers, layer_tracker.move_layer, layer_tracker.swap_platforms]
        action = self.rng.choice(a=action_space, p=[0.45, 0.45, 0.1])

        match action:
            case layer_tracker.swap_layers:
                platforms_to_swap = self.rng.choice(self.node_ids, size=2, replace=False)
                layers_to_swap = []
                for platform in platforms_to_swap:
                    found = np.array(layer_tracker.get_all_on_platform(platform))
                    layers = layer_tracker.get_layer_names()[found] # These are all layers on this platform
                    if layers.size == 0:
                        return layer_tracker
                    layers_to_swap.append(self.rng.choice(layers))
                action(layers_to_swap)

            case layer_tracker.move_layer:
                # Move one layer to other platform
                platform_from, platform_to = self.rng.choice(self.node_ids, size=2, replace=False)
                layer_to_move_sel = np.array(layer_tracker.get_all_on_platform(platform_from)) 
                if layer_to_move_sel.size == 0:
                    return layer_tracker
                layer_names = layer_tracker.get_layer_names()
                layer_candidates = layer_names[layer_to_move_sel]
                if layer_candidates.size == 0:
                    return layer_tracker
                layer_to_move = self.rng.choice(layer_names[layer_to_move_sel])
                action(layer_to_move, platform_to)

            case layer_tracker.swap_platforms:
                platform_a, platform_b = self.rng.choice(self.node_ids, size=2, replace=False)
                action(platform_a, platform_b)

        return layer_tracker


    def minimize(self):
        # Distribute workload between platforms with SA

        best_cost, best_solution =  self._simulated_annealing()

        # Move binding to graphs
        for network, layer, binding in best_solution:
            self.execution_graphs[network].reduced_graph.nodes(data=True)[layer]["binding"] = binding
            print(self.execution_graphs[network].reduced_graph.nodes(data=True)[layer]["binding"])
        
        # For node_stats: when a layer was not bound to a specific platform, reset scheduling information
        for platform, data in self.node_stats.platforms.items():
            for nw_name, nw_data in data.designs[0].networks.items():
                for layer_name, layer_data in nw_data.layers.items():
                    binding = best_solution.layer[layer_name]
                    if binding != platform:
                        layer_data.reset_scheduling()

        #TODO: Between binding and scheduling, the design must be fixed and a new node_stats must be created with the single design

        # Perform the scheduling
        scheduled_node_stats, _ = list_scheduling(self.execution_graphs, self.node_stats, writeback_layer_info=True)
        return scheduled_node_stats

    def _score_binding(self, tracker: LayerTracker) -> float:
        total_edp = 0.0    
        for nw, layer, bind in tracker:
            # TODO Currently assumes that there is only one design
            layer_data = self.node_stats[bind].designs[0].networks[nw].layers[layer]
            total_edp += layer_data.edp
        
        return total_edp
    
    def _score_utilization(self, schedules: dict[int, PlatformSchedule]) -> float:
        idle_times = [sched.get_idle_time() for sched in schedules.values()]
        return sum(idle_times)
    
    def _score_scheduling(self, schedules: dict[int, PlatformSchedule]) -> float:
        end_times = [sched.cur_time for sched in schedules.values()]
        end_times_pairs = it.combinations(end_times, r=2)
        end_times_diffs = [abs(pair[0]-pair[1]) for pair in end_times_pairs]
        return sum(end_times_diffs)

    def _calc_cost(self, tracker: LayerTracker, schedules: dict[int, PlatformSchedule]) -> float:
        edp_cost = self._score_binding(tracker)
        lat_cost = self._score_scheduling(schedules)
        util_cost = self._score_utilization(schedules)

        return edp_cost**1.0 + lat_cost**1.0 + util_cost**2.8
    
    def _write_binding_to_graph(self, tracker: LayerTracker, exec_graphs: dict[str, ExecutionGraph]):
        for nw, layer, binding in tracker:
            exec_graphs[nw].reduced_graph.nodes[layer]["binding"] = binding

