import time
import numpy as np
import torch
from torch import nn, Tensor
from .model.model import TreeModel
from .model.graph import LayersGraph
from .model.memoryHelper import MemoryInfo
from framework.constants import MODEL_PATH
from collections import OrderedDict
from typing import List, Dict
from copy import deepcopy
from torchinfo.layer_info import LayerInfo
from .Filter import Filter

class _DictToTensorModel(nn.Module):
    def __init__(self, d : Dict[str, Tensor]) -> None:
        super(_DictToTensorModel, self).__init__()

        self.key = list(d.keys())[0]

    def forward(self, x: Tensor) -> Tensor:
        return x[self.key]


class DNNAnalyzer:
    def __init__(self, model_path: str, input_size: tuple,conf_helper) -> None:
        self.partition_points = []
        self.input_size = input_size
        self.config_helper = conf_helper
        self.constraints = conf_helper.get_constraints()
        self.node_components,self.link_components =  conf_helper.get_system_components()
        self.memoryInfo = MemoryInfo()

        self.num_bytes = int(self.constraints["word_width"] / 8)
        if self.constraints["word_width"] % 8 > 0:
            self.num_bytes += 1

        self.stats = {}
        t0 = time.time()

        self._Tree_Model = TreeModel(model_path,self.input_size)
        self._tree = self._Tree_Model.get_Tree()
        self.torchModels= self._Tree_Model.get_torchModels()

        self.graph = LayersGraph(self._tree)
        graph_pp = self.graph.get_graph_partition_points()

        self.partition_points = [
            point for point in self._tree if point["name"] in graph_pp
        ]

        conv_layers = self.get_conv2d_layers()
        mems = self.memoryInfo.get_max_conv2d_layer(self.graph,conv_layers ,self.input_size)

        self.Filter = Filter(self.memoryInfo,self.config_helper,self.partition_points,mems)
        self.partpoints_filtered, self.part_max_layer,self.nodes_memory  = self.Filter.apply_filter()

        t1 = time.time()
        self.stats["sim_time"] = t1 - t0

        print("Found", len(self.partpoints_filtered), "partition points.")

        t0 = time.time()

        t1 = time.time()
        self.stats["mem_estimation_time"] = t1 - t0

    def get_conv2d_layers(self):
        output = [layer for layer in self._tree if layer.get("op_type") == "Conv"]
        return output

    def search_partition_point(self, layer_name):
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
