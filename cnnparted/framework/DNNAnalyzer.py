import torch
from torch import nn, Tensor
from torch.nn import functional
from torchinfo import summary
from torchinfo.layer_info import LayerInfo

# from framework.Memory_Estimator.Estimator import Estimator
import time
import numpy as np
import time
import os
from collections import OrderedDict

from .model.model import TreeModel
from .model.graph import LayersGraph
from .model.memoryHelper import MemoryInfo
from typing import List, Dict
from framework.constants import MODEL_PATH
from copy import deepcopy


class DNNAnalyzer:
    def __init__(self, model_path: str, input_size: tuple, constraints: dict) -> None:
        self.partition_points = []
        self.input_size = input_size
        self.constraints = constraints
        self.memoryInfo = MemoryInfo()

        self.stats = {}
        t0 = time.time()
        # output_name = "model.onnx"
        # model_path = os.path.join(ROOT_DIR, output_name)

        # x = torch.randn(input_size)
        # torch.onnx.export(model, x, MODEL_PATH, verbose=False, input_names=['input'], output_names=['output'])

        self._Tree_Model = TreeModel(model_path)
        self._tree = self._Tree_Model.get_Tree()

        self.graph = LayersGraph(self._tree)
        graph_pp = self.graph.get_graph_partition_points()

        self.partition_points = [
            point for point in self._tree if point["name"] in graph_pp
        ]

        max_size = self.constraints["max_out_size"]
        if max_size != 0:
            self.partpoints_filtered = [
                layer
                for layer in self.partition_points
                if np.prod(layer.get("output_size")) < max_size
            ]

        else:
            self.partpoints_filtered = (self.partition_points.copy(),)

        t1 = time.time()
        self.stats["sim_time"] = t1 - t0

        print("Found", len(self.partpoints_filtered), "partition points.")

        self.max_conv_layer = self.get_max_conv2d_layer()
        self.max_part_point = self.graph.find_the_nearest_ancestor(
            source=self.max_conv_layer, node_list=self.partition_points
        )

    def get_conv2d_layers(self):
        output = [layer for layer in self._tree if layer.get("op_type") == "Conv"]
        return output
        # return [layer for layer in self.layers if isinstance(layer.module, nn.Conv2d)]

    def get_max_conv2d_layer(self):
        max_mem_allowed = self.constraints["max_memory_size"]
        width_bytes = self.constraints["word_width"] / 8
        conv_layers = self.get_conv2d_layers()
        ofms, ifms, weights = self.memoryInfo.get_convs_memory(conv_layers)

        convs_subgraphs, root = self.graph.get_all_conv_subgraphs()

        if (ifms[root] + ofms[root] + weights[root]) * width_bytes > max_mem_allowed:
            return root

        memory = weights[root]
        for subgraph in convs_subgraphs:
            orders = self.graph.get_all_topological_orders(subgraph)
            (
                subgraph_max_memory,
                orders_memory,
            ) = self.memoryInfo.calculate_max_memory_usage(
                subgraph, weights, ifms, ofms, orders
            )
            subgraph_min_memory_necessary = min(subgraph_max_memory.values())

            max_memory = subgraph_min_memory_necessary + memory
            print("subghraph ", subgraph, "max_memory ", max_memory * width_bytes)
            if max_memory * width_bytes > max_mem_allowed:
                return orders[0][0]

            weights_to_sum = [weights[key] for key in orders[0][1:]]
            weights_sum = sum(weights_to_sum)
            memory += weights_sum

    def search_partition_point(self, layer_name):
        # print(layer_name)
        match = next(
            (
                layer
                for layer in self._tree
                if layer.get("name") == layer_name and layer in self.partition_points
            ),
            None,
        )
        if match != None:
            return layer_name
        else:
            found_current_layer = False
            layer_graph = self.graph.get_Graph()
            nodes = layer_graph.nodes()
            sorted_nodes = sorted(
                nodes, key=lambda node: list(layer_graph.nodes()).index(node)
            )

            for node in sorted_nodes:
                if node == layer_name:
                    found_current_layer = True
                    continue

                if found_current_layer:
                    match = next(
                        (
                            layer
                            for layer in self._tree
                            if layer.get("name") == node
                            and layer in self.partition_points
                        ),
                        None,
                    )
                    if match == None:
                        continue
                    return match.get("name")

            raise Exception("Not able to find parent partition point")
