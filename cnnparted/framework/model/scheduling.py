##
# Taken from: https://github.com/akshay-k2/m5
##

import random
from joblib import Parallel, delayed
import tqdm
import networkx as nx
import numpy as np

from typing import List, Optional, Union


def topo_sort_starting_node(
    G: nx.DiGraph, strating_node: int, seed: Optional[int] = None, as_ndarray: bool = False
) -> Union[List[int], np.ndarray]:

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

def topo_sort_random_start_node(G, n: int = 1, seed: int = None, as_ndarray: bool = False, n_jobs: int = 1):

    if seed is not None:
        random.seed(seed)

    starting_points = random.choices(list(G.nodes), k=n)
    sorts = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(topo_sort_starting_node)(G, starting_point, seed=None, as_ndarray=as_ndarray)
        for starting_point in tqdm.tqdm(starting_points)
    )

    if as_ndarray:
        sorts_np = [np.array(s) for s in sorts]
        return list(sorts_np)
    else:
        return list(sorts)


def check_sort(G, L):
    for i, n in enumerate(L):
        ancestors = list(nx.ancestors(G, n))
        for a in ancestors:
            if a not in L[:i]:
                return False
    return True
