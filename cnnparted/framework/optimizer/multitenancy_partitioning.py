import os
import copy
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng
import tqdm

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from joblib import Parallel, delayed

from framework.optimizer.optimizer import Optimizer
from framework.optimizer.partitioning_problem import PartitioningProblem
from framework.optimizer.config.partitioning_opt_config import PartitioningOptConfig
from framework.graph_analyzer import GraphAnalyzer
from framework.helpers.config_helper import ConfigHelper
from framework.node.node_evaluator import SystemResult
from framework.model.execution_graph import ExecutionGraph


# What this should do:
#   - given be a random, initial partitioning of the networks
#   - try to optimize EDP by moving partitions between accelerators

class MultiTenancyPartitioning(Optimizer):
    def __init__(self, node_stats: SystemResult, ga: GraphAnalyzer):
        self.ga = ga
        self.system_result = node_stats

    def optimize(self):
        self._gen_initial_binding()

        algorithm = MultiTenancyProblem(self.system_result, self.ga)
        algorithm.minimize()

    def _swap_random_layer(self):
        ...
    
    def _move_random_layer(self):
        ...
    
    def _eval_design_layer_set(self):
        ...
    
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
    
    def swap_layers(self, layers: list[str]):
        bindings = np.array([self.layer[layer] for layer in layers])
        first_l, second_l = layers
        first_b, second_b = bindings
        self.layer[first_l] = second_b
        self.layer[second_l] = first_b


class MultiTenancyProblem():
    def __init__(self, node_stats: SystemResult, ga: GraphAnalyzer):
        self.node_stats = node_stats
        self.node_ids = node_stats.get_platform_ids()
        self.ga = ga

        # Parse execution graphs into LayerTracker
        networks = []
        layers = defaultdict(int)
        for network, graph in ga.exec_graphs.items():
            for node, data in graph.reduced_graph.nodes(data=True):
                networks.append(network)
                layers[node] = data["binding"]

        self.layer_tracker = LayerTracker(networks, layers)

        # SA configuration
        self.num_iter = 100_000
        self.t_start = 1000
        self.cost_history = []
    
    def _simulated_annealing(self):
        old_cost, _ = self._score_binding_graph(self.ga.exec_graphs)

        for i in range(self.num_iter):
            t = self.t_start / (i+1)
            new_state = self._random_modify()
            new_cost, _ = self._score_binding_tracker(new_state)
            
            if np.random.random() <= np.exp( -(new_cost - old_cost) / t ):
                self.layer_tracker = new_state
                old_cost = new_cost
                self.cost_history.append(new_cost)
                print(new_cost)

    
    def _random_modify(self):
        # For testing: randomly swap two bindings
        platforms_to_swap = np.random.choice(self.node_ids, size=2, replace=False)
        layers_to_swap = []
        layer_tracker = copy.deepcopy(self.layer_tracker)
        for platform in platforms_to_swap:
            found = np.array(self.layer_tracker.get_all_on_platform(platform))
            layers = np.array(list(self.layer_tracker.layer.keys()))[found] # These are all layers on this platform
            layers_to_swap.append(np.random.choice(layers))
        layer_tracker.swap_layers(layers_to_swap)
        return layer_tracker

    def minimize(self):
        self._simulated_annealing()
    
    def _score_binding_graph(self, graphs: dict[str, ExecutionGraph]) -> tuple[int, dict[int, int]]:
        total_edp = 0        
        edp_per_platform = defaultdict(int)
        for nw, graph in graphs.items():
            for node, data in graph.reduced_graph.nodes(data=True):
                binding = data["binding"]
                platform = self.node_stats[binding]

                # TODO Currently assumes that there is only one design
                layer_data = platform.designs[0].get_network_dict(nw)[node]
                energy = layer_data["energy"]
                latency = layer_data["latency"]
                total_edp += energy * latency
                edp_per_platform[binding] += energy * latency
        
        return total_edp, edp_per_platform
    
    def _score_binding_tracker(self, tracker: LayerTracker) -> tuple[int, dict[int, int]]:
        total_edp = 0        
        edp_per_platform = defaultdict(int)
        for nw, layer, bind in tracker:
            # TODO Currently assumes that there is only one design
            layer_data = self.node_stats[bind].designs[0].get_network_dict(nw)[layer] #layers_dict[nw]
            total_edp += layer_data["energy"] * layer_data["latency"]
            edp_per_platform[bind] += layer_data["energy"] * layer_data["latency"]
        
        return total_edp, edp_per_platform

