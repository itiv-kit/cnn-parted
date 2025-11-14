##
# Taken from: https://github.com/akshay-k2/m5
##

from dataclasses import dataclass
import itertools as it
import random
from joblib import Parallel, delayed
import tqdm
import networkx as nx
import numpy as np

from typing import Literal, Optional, Union
from copy import deepcopy

from framework.noc.noc import NocTopology
from framework.model.execution_graph import ExecutionGraph
from framework.node.node_evaluator import SystemResult, LayerResult
from framework.helpers.dtypes import ComputeTask, TransferTask

def topo_sort_starting_node(
    G: nx.DiGraph, strating_node: int, seed: Optional[int] = None, as_ndarray: bool = False
) -> Union[list[int], np.ndarray]:

    if seed is not None:
        random.seed(seed)

    G_unique = G.copy()
    L: list[int] = []

    # set all node tags to "unvisited"
    for node in G_unique.nodes:
        G_unique.nodes[node]["tag"] = "unvisited"

    # set starting node tag to "center"
    G_unique.nodes[strating_node]["tag"] = "center"

    def tag_connected_nodes(G, node):
        # compute intersection of successors and predecessors
        successors = set(G.successors(node))
        predecessors = set(G.predecessors(node))
        connected_nodes_intersection = successors.intersection(predecessors)
        if len(connected_nodes_intersection) > 0:
            raise ValueError("Found nodes that are both successors and predecessors, this is not allowed in a DAG")

        for child in G.successors(node):
            if G.nodes[child]["tag"] == "top":
                continue
            else:
                G.nodes[child]["tag"] = "bottom"
        for parent in G.predecessors(node):
            if G.nodes[parent]["tag"] == "bottom":
                continue
            else:
                G.nodes[parent]["tag"] = "top"

    def remove_and_add_to_list(G_unique, L, node):
        # get node data
        node_tag = G_unique.nodes[node]["tag"]

        if node_tag == "unvisited":
            raise ValueError("Node is not tagged as a node that can be removed")

        if node_tag == "center":
            # append to empty list
            if L != []:
                raise ValueError("Center node is not the first node in the list")
            L.append(node)
            # tag connected nodes
            tag_connected_nodes(G_unique, node)
            # remove node from graph
            G_unique.remove_node(node)

        elif node_tag == "top":
            # append to the top of the list
            L.insert(0, node)
            # tag connected nodes
            tag_connected_nodes(G_unique, node)
            # remove node
            G_unique.remove_node(node)

        elif node_tag == "bottom":
            # append to the bottom of the list
            L.append(node)
            # tag connected nodes
            tag_connected_nodes(G_unique, node)
            # remove node
            G_unique.remove_node(node)
        else:
            raise ValueError("Node is not tagged as a node that can be removed")

    def filter_top_nodes(G: nx.DiGraph, possible_nodes: list[int]) -> list[int]:
        top_nodes = []
        for node in possible_nodes:
            node_descendants = set(nx.descendants(G, node))
            node_descendants_tags = [G.nodes[n]["tag"] for n in node_descendants]
            if "top" in node_descendants_tags:
                continue
            else:
                top_nodes.append(node)
        return top_nodes

    def filter_bottom_nodes(G: nx.DiGraph, possible_nodes: list[int]) -> list[int]:
        bottom_nodes = []
        for node in possible_nodes:
            node_ancestors = set(nx.ancestors(G, node))
            node_ancestors_tags = [G.nodes[n]["tag"] for n in node_ancestors]
            if "bottom" in node_ancestors_tags:
                continue
            else:
                bottom_nodes.append(node)
        return bottom_nodes

    count = 0

    # inital_pos = layout_dot(G_unique)
    # os.makedirs("./figures/random_graph_gen/", exist_ok=True)
    # plot_tagged_graph(G_unique, inital_pos, f"./figures/random_graph_gen/{count}.png")

    remove_and_add_to_list(G_unique, L, strating_node)
    while len(L) < len(G.nodes):
        possible_top_nodes = [node for node in G_unique.nodes if G_unique.nodes[node]["tag"] == "top"]
        possible_bottom_nodes = [node for node in G_unique.nodes if G_unique.nodes[node]["tag"] == "bottom"]

        possible_top_nodes_filtered = filter_top_nodes(G_unique, possible_top_nodes)
        possible_bottom_nodes_filtered = filter_bottom_nodes(G_unique, possible_bottom_nodes)

        possible_node = possible_top_nodes_filtered + possible_bottom_nodes_filtered
        # randomly pick a node
        random_node = random.choice(possible_node)
        # remove node from graph and add to list
        remove_and_add_to_list(G_unique, L, random_node)
        count += 1
        # print(f"Progress: {round(count/len(G.nodes)*100, 2)}%")
        # plot_tagged_graph(G_unique, inital_pos, f"./figures/random_graph_gen/{count}.png")

    if not check_sort(G, L):
        raise ValueError("Sort is not a valid topological sort")

    if as_ndarray:
        L_np = np.array(L)
        return L_np
    else:
        return L

def topo_sort_random_start_node(G, n: int = 1, seed: int = None, as_ndarray: bool = False, n_jobs: int = 1, progress : bool = False):

    if seed is not None:
        random.seed(seed)

    starting_points = random.choices(list(G.nodes), k=n)
    sorts = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(topo_sort_starting_node)(deepcopy(G), starting_point, seed=None, as_ndarray=as_ndarray)
        for starting_point in tqdm.tqdm(starting_points, "Schedules", disable=(not progress))
    )

    if as_ndarray:
        sorts_np = [np.array(s) for s in sorts]
        return list(sorts_np)
    else:
        return list(sorts)


class PlatformSchedule:
    def __init__(self, id: int):
        self.id = id
        self.layers: list[LayerResult] = []
        self.cur_time = 0.0

        self.tasks: list[ComputeTask | TransferTask] = []
    
    def get_time(self):
        return self.cur_time
    
    def assign_layer(self, network: str, layer: LayerResult, earliest_start: float):
        start_time = max(earliest_start, self.cur_time)
        end_time = max(earliest_start, self.cur_time) + layer.latency

        self.cur_time = start_time + layer.latency
        
        task = ComputeTask(network, layer.name, start_time, end_time)
        self.tasks.append(task)
        return task
    
    def assign_tensor(self, network: str, tensor):
        ...
        n_bytes = 1000
    
    def get_idle_time(self) -> float:
        idle_time = [(nxt.end_time-cur.start_time)  for cur, nxt in it.pairwise(self.tasks)]
        return sum(idle_time)
    

class SystemSchedule:
    def __init__(self, platform_ids: list[int], networks: list[str]):
        self.node_schedules = {id: PlatformSchedule(id) for id in platform_ids}
        self.networks = networks
        self.layers: dict[int, dict[str, dict[str, ComputeTask]]] = {}
        for platform_id in platform_ids: 
            self.layers[platform_id] = {nw: {} for nw in networks}

    def assign_layer(self, platform_id: int, network: str, layer: LayerResult, earliest_start: float):
        task = self.node_schedules[platform_id].assign_layer(network, layer, earliest_start)
        
        # Keep a dict to make searching for a specifc layer easier
        self.layers[platform_id][network][task.layer] = task 

    @property
    def system_latency(self):
        latencies = [sched.cur_time for sched in self.node_schedules.values()]
        return max(latencies)
    
    def get_system_energy(self, node_stats: SystemResult):
        total_energy = 0.0
        for platform, sched in self.node_schedules.items():
            for task in sched.tasks:
                layer_result = node_stats.platforms[platform].designs[0].networks[task.network].layers[task.layer]
                total_energy += layer_result.energy
        return total_energy


def list_scheduling(exec_graphs: dict[str, ExecutionGraph], node_stats: SystemResult, 
                    priority: Literal["earliest_start", "shortest_job", "earliest_finish"]) -> SystemSchedule:
    """Used for MultiTenancy partitioning. After the layers have been assigned to cores,
    we must determine a start time for each layer"""

    from framework.model.onnx_model import OnnxModel
    dummy = OnnxModel()

    # Define helper functions to find layers
    def find_layer_info(id: int, nw_name: str, layer_name: str):
        network = node_stats[id].designs[0].networks[nw_name]
        if layer_name in network:
            return network[layer_name]
        return False

    schedule = SystemSchedule(node_stats.get_platform_ids(), list(exec_graphs.keys()) )

    # Need to keep track of earliest valid starting time for each node here
    for network, exec_graph in exec_graphs.items():
        for node_name in exec_graph.reduced_graph.nodes:
            node = exec_graph.reduced_graph.nodes[node_name]
            node["earliest_start"] = 0.0

    # While not all nodes have been assigned
    all_assigned = False
    while not all_assigned:
        #Determine all nodes that can be scheduled (all that have no predecessor)
        V_ready: list[tuple] = []
        for network, exec_graph in exec_graphs.items():
            for node_name in exec_graph.reduced_graph.nodes:
                if exec_graph.reduced_graph.in_degree(node_name) == 0:
                    node = exec_graph.reduced_graph.nodes[node_name]
                    platform = node["binding"]
                    node["eval_data"] = find_layer_info(platform, network, node_name)
                    introduced_delay = abs(schedule.node_schedules[platform].cur_time - node["earliest_start"])
                    V_ready.append( (network, node_name, node, introduced_delay) )
        
        match priority:
            case "earliest_start":
                V_ready = sorted(V_ready, key=lambda data: data[2]["earliest_start"])
            case "shortest_job":
                V_ready = sorted(V_ready, key=lambda data: data[2]["eval_data"].latency)
            case "earliest_finish":
                V_ready = sorted(V_ready, key=lambda data: data[2]["earliest_start"]+data[2]["eval_data"].latency)
            case _:
                raise RuntimeError(f"Found an unknown {priority}")


        for network, layer_name, layer_data, introduced_delay in V_ready:
            binding = layer_data["binding"]
            layer_result = find_layer_info(binding, network, layer_name) #should be name
            if layer_result:
                schedule.assign_layer(binding, network, layer_result, layer_data["earliest_start"])

                # Before we remove the node, set the earliest start for all successory
                for succ_name in exec_graphs[network].reduced_graph.successors(layer_name):
                    succ_node = exec_graphs[network].reduced_graph.nodes[succ_name]
                    succ_task = schedule.layers[binding][network][layer_name]
                    succ_node["earliest_start"] = max(succ_node["earliest_start"], succ_task.end_time)

                exec_graphs[network].reduced_graph.remove_node(layer_name)

        num_remaining_nodes = sum([graph.reduced_graph.number_of_nodes() for graph in exec_graphs.values()])
        all_assigned = (num_remaining_nodes == 0)

    return schedule


def noc_scheduling(noc: NocTopology, execution_graph: dict[str, ExecutionGraph]):
    # If the noc is given and we have the relations between tensors and computes, perform the scheduling
    ...


def check_sort(G, L):
    for i, n in enumerate(L):
        ancestors = list(nx.ancestors(G, n))
        for a in ancestors:
            if a not in L[:i]:
                return False
    return True
