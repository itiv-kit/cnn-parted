import torch
from torch import nn, Tensor
from torch.nn import functional
from torchinfo import summary
from torchinfo.layer_info import LayerInfo
#from framework.Memory_Estimator.Estimator import Estimator
import time
import numpy as np
import time
import os
from collections import OrderedDict

from .model.model import TreeModel
from .model.graph import LayersGraph
from typing import List, Dict
from framework.constants import MODEL_PATH
from copy import deepcopy


class _DictToTensorModel(nn.Module):
    def __init__(self, d : Dict[str, Tensor]) -> None:
        super(_DictToTensorModel, self).__init__()

        self.key = list(d.keys())[0]

    def forward(self, x: Tensor) -> Tensor:
        return x[self.key]


def buildSequential(layers : List[LayerInfo], input_size : list, device : str) -> nn.Sequential:
    modules = OrderedDict()
    output_size = list(input_size)
    for layer in layers:
        if len(layer.input_size) != len(output_size):
            modules[str(layer.layer_id) + 'f'] = nn.Flatten(1)

        modules[str(layer.layer_id)] = layer.module

        output_size = layer.output_size

        rand_tensor = torch.randn(layer.input_size, device=device)
        m = deepcopy(layer.module)
        m.to(device)
        out = m(rand_tensor)
        if isinstance(out, dict):
            modules[str(layer.layer_id)] = _DictToTensorModel(out)

    return nn.Sequential(modules)


class DNNAnalyzer:


    def _add_dummy_input(self) -> None:
        input_data = LayerInfo('Identity', nn.Identity(), 0)
        input_data.input_size = self.input_size
        input_data.output_size = self.input_size
        self.partition_points.insert(0, input_data)
        self.partpoints_filtered.insert(0, input_data)


    def __init__(self, model : nn.Module, input_size : tuple, constraints : dict) -> None:
        self.partition_points = []
        self.input_size = input_size
        #self.mem_estimator = Estimator()
        self.constraints = constraints

        self.stats = {}
        t0 = time.time()
        # output_name = "model.onnx"
        # model_path = os.path.join(ROOT_DIR, output_name)

        x = torch.randn(input_size)
        torch.onnx.export(model, x, MODEL_PATH, verbose=False, input_names=['input'], output_names=['output'])

        self._Tree_Model = TreeModel(MODEL_PATH)
        self._tree = self._Tree_Model.get_Tree()

        self.graph = LayersGraph(self._tree)
        graph_pp = self.graph.get_graph_partition_points()
        
        self.partition_points = [point for point in self._tree if point['name'] in graph_pp]
        max_size = self.constraints["max_out_size"]
        if max_size != 0:
            self.partpoints_filtered = [layer for layer in self.partition_points if np.prod(layer.get('output_size')) < max_size]
        
        else:
            self.partpoints_filtered = self.partition_points.copy(),
        
        t1 = time.time()
        self.stats['sim_time'] = t1 - t0

        print("Found", len(self.partpoints_filtered), "partition points.")
        


        ################


    def get_conv2d_layers(self):
        output = [layer for layer in self._tree if layer.get('op_type') == 'Conv']
        return output
        #return [layer for layer in self.layers if isinstance(layer.module, nn.Conv2d)]
    
    #Mahdi:
    def get_linear_layers(self) -> List[LayerInfo]:
        for layer in self.layers:
            print(layer.module)

    #def get_max_conv2d_layer(self):

        # max_mem_allowed = self.constraints["max_memory_size"]
        # width_bytes =self.constraints["word_width"]/8
        # weights=0

        # layers = self.get_conv2d_layers()

        # # return all layers when max_mem_allowed not defined
        # if max_mem_allowed == 0 :
        #     return layers
        
        # output= []
        # for layer in layers:
        #     start_time = time.time()
        #     #memory_paramers_count,weights_local = self.mem_estimator.compute(layer) 
        #     end_time = time.time()
        #     print("time in sec : ", end_time - start_time) 
        #     print("memory ",(memory_paramers_count + weights)*width_bytes)  
        #     #if (memory_paramers_count + weights)*width_bytes >= max_mem_allowed: # weights from old layers must be included in memory??
        #         break
            
        #     weights += weights_local
        #     output.append(layer)
        
        #return output

    def search_partition_point(self, layer_name):
        print(layer_name)
        match = next((layer for layer in self._tree if layer.get('name') == layer_name and layer in self.partition_points),None)
        if match != None:
            return layer_name         
        else:
            found_current_layer = False
            layer_graph = self.graph.get_Graph()
            nodes = layer_graph.nodes()
            sorted_nodes = sorted(nodes, key=lambda node: list(layer_graph.nodes()).index(node))

            for node in sorted_nodes:

                if node == layer_name: 
                    found_current_layer = True
                    continue  

                if found_current_layer:
                    match = next((layer for layer in self._tree if layer.get('name') == node and layer in self.partition_points),None)
                    if match == None:
                        continue
                    return match.get('name')

            raise Exception("Not able to find parent partition point")

        