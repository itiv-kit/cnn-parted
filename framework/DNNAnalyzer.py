import torch
from torch import nn, Tensor
from torch.nn import functional
from torchinfo import summary
from torchinfo.layer_info import LayerInfo
from framework.Memory_Estimator.Estimator import Estimator
import time

import numpy as np
import time

from typing import List, Dict


class _DictToTensorModel(nn.Module):
    def __init__(self, d : Dict[str, Tensor]) -> None:
        super(_DictToTensorModel, self).__init__()

        self.key = list(d.keys())[0]

    def forward(self, x: Tensor) -> Tensor:
        return x[self.key]


def buildSequential(layers : List[LayerInfo], input_size : list, device : str) -> nn.Sequential:
    modules = []
    output_size = list(input_size)
    for layer in layers:
        if len(layer.input_size) != len(output_size):
            modules.append(nn.Flatten(1))

        output_size = layer.output_size
        modules.append(layer.module)

        rand_tensor = torch.randn(layer.input_size, device=device)
        layer.module.to(device)
        out = layer.module(rand_tensor)
        if isinstance(out, dict):
            modules.append(_DictToTensorModel(out))

    return nn.Sequential(*modules)


class DNNAnalyzer:
    def _find_partitions(self, layers : List[LayerInfo], root_layer : LayerInfo) -> bool:
        if isinstance(root_layer.module, nn.ModuleDict): # skip nn.ModuleDict
            if root_layer in self.partition_points:
                root_idx = self.partition_points.index(root_layer)
                self.partition_points[root_idx:root_idx] = layers
                self.partition_points.remove(root_layer)
            return True

        if root_layer.input_size != []:
            seq = buildSequential(layers, root_layer.input_size, self.device)
            rand_tensor = torch.randn(root_layer.input_size, device=self.device)
        else:
            seq = buildSequential(layers, self.input_size, self.device)
            rand_tensor = torch.randn(self.input_size, device=self.device)

        try:
            root_layer.module.eval()
            res1 = root_layer.module(rand_tensor)
            seq.eval()
            res2 = seq(rand_tensor)

            if isinstance(res1, dict):          # hotfix for FCN_ResNet architecture
                res1 = _DictToTensorModel(res1)(res1)
                input_shape = rand_tensor.shape[-2:]
                res2 = functional.interpolate(res2, size=input_shape, mode='bilinear', align_corners=False)

            if res1.size() != res2.size():      # hotfix for flattening in SqueezeNet
                res2 = torch.flatten(res2, 1)

            if not False in torch.unique(res1 == res2):
                if root_layer in self.partition_points:
                    root_idx = self.partition_points.index(root_layer)
                    self.partition_points[root_idx:root_idx] = layers
                    self.partition_points.remove(root_layer)
                else:
                    for layer in layers:
                        self.partition_points.append(layer)

                return True

        except RuntimeError as rte:
            if not 'expected input' in rte.args[0]:
                raise Exception

        return False


    def _scan(self, graph : dict, node : LayerInfo) -> None:
        found = True
        if not node.is_leaf_layer:
            found = self._find_partitions(graph[node.layer_id], node)
        elif node.is_leaf_layer and (not node in self.partition_points):
            self.partition_points.append(node)

        if found:
            for neighbor in graph[node.layer_id]:
                self._scan(graph, neighbor)


    def _add_dummy_input(self) -> None:
        input_data = LayerInfo('Identity', nn.Identity(), 0)
        input_data.input_size = self.input_size
        input_data.output_size = self.input_size
        self.partition_points.insert(0, input_data)
        self.partpoints_filtered.insert(0, input_data)


    def __init__(self, model : nn.Module, input_size : tuple, constraints : dict) -> None:
        self.partition_points : list[LayerInfo] = []
        modsum = summary(model, input_size, depth=100, verbose=0)
        self.layers = [layer for layer in modsum.summary_list if layer.class_name != "Dropout" ]
        self.input_size = input_size
        self.mem_estimator = Estimator()
        self.constraints = constraints

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._topLayer = None
        self._layerTree = {}

        self.stats = {}
        t0 = time.time()

        for layer in self.layers:
            self._layerTree[layer.layer_id] = []
            if layer.parent_info == None:
                self._topLayer = layer
            else:
                self._layerTree[layer.parent_info.layer_id].append(layer)

        self._scan(self._layerTree, self._topLayer)

        #self.max_conv2d_layer = self.get_max_conv2d_layer()

        max_size = self.constraints["max_out_size"]
        if max_size != 0:
            self.partpoints_filtered = [layer for layer in self.partition_points if np.prod(layer.output_size) < max_size]
        else:
            self.partpoints_filtered = self.partition_points.copy()
        self._add_dummy_input()

        t1 = time.time()
        self.stats['sim_time'] = t1 - t0

        print("Found", len(self.partpoints_filtered), "partition points.")

    def get_conv2d_layers(self) -> List[LayerInfo]:
        return [layer for layer in self.layers if isinstance(layer.module, nn.Conv2d)]
    
    #Mahdi:
    def get_linear_layers(self) -> List[LayerInfo]:
        for layer in self.layers:
            print(layer.module)

    def get_max_conv2d_layer(self):

        max_mem_allowed = self.constraints["max_memory_size"]
        width_bytes =self.constraints["word_width"]/8
        weights=0

        layers = self.get_conv2d_layers()

        # return all layers when max_mem_allowed not defined
        if max_mem_allowed == 0 :
            return layers
        
        output= []
        for layer in layers:
            start_time = time.time()
            memory_paramers_count,weights_local = self.mem_estimator.compute(layer) 
            end_time = time.time()
            print("time in sec : ", end_time - start_time) 
            print("memory ",(memory_paramers_count + weights)*width_bytes)  
            if (memory_paramers_count + weights)*width_bytes >= max_mem_allowed: # weights from old layers must be included in memory??
                break
            
            weights += weights_local
            output.append(layer)
        
        return output

    def search_partition_point(self, layer : LayerInfo) -> LayerInfo:
        for pp in self.partition_points:
            if layer.layer_id == pp.layer_id:
                return pp

        if layer.parent_info:
            return self.search_partition_point(layer.parent_info)

        raise Exception("Not able to find parent partition point")