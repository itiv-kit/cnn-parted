import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy

class ExecutionGraph:
    def __init__(self, model_tree):
        self.model_tree = model_tree

        self.evaluated_layer_types = ["Conv", "Gemm", "MatMul"] #if op_type == 1 it is an input/output layer
        self.pooling_types = ["MaxPool", "GlobalAveragePool"]

        self.full_graph = nx.DiGraph()
        self.reduced_graph = nx.DiGraph()
        self._create_layer_relationships()
        self._delete_not_simulated_nodes()
        self.output_sizes = self._set_output_sizes()
    
    def get_graph(self):
        return self.reduced_graph

    def _set_output_sizes(self) -> dict:
        layer_names = [layer['name'] for layer in self.model_tree]
        layer_outputs = [layer['output_size'] for layer in self.model_tree]
        return {key: value for key, value in zip(layer_names, layer_outputs)}

    def _create_layer_relationships(self):
        for layer in self.model_tree:
            if layer['op_type']!= 'Constant':
                tmp = copy.deepcopy(layer)
                del tmp["children"]
                self.full_graph.add_node(layer["name"], 
                                         data=tmp, 
                                         pooling={"fused_pooling": False}, 
                                         children=[], 
                                         binding=None)
                for child in layer["children"]:
                    self.full_graph.add_edge(layer["name"], child["name"], weight=np.prod(layer["output_size"]) )
        self._fuse_conv_maxpool()
    
    def _fuse_conv_maxpool(self):
        node_keys = list(self.full_graph.nodes)
        for node_key in node_keys:
            node = self.full_graph.nodes[node_key]
            successors = self.full_graph.succ[node_key]
            if successors is not None:
                if node["data"]["op_type"] in self.evaluated_layer_types and len(successors) != 1:
                    raise RuntimeError("Conv with multiple outputs is not supported")

                for succ in successors:
                    succ_node = self.full_graph.nodes[succ]
                    
                    # Store information about pooling as pooling nodes will later be deleted
                    if succ_node["data"]["op_type"] in self.pooling_types:
                        node["pooling"]["fused_pooling"] = True
                        node["pooling"]["pooling_output_size"] = succ_node["data"]["output_size"]

    def _delete_not_simulated_nodes(self):
        graph = copy.deepcopy(self.full_graph)
        node_keys = list(graph)
        for node_key in node_keys:
            node = graph.nodes[node_key]
            if node["data"]["op_type"] not in self.evaluated_layer_types:
                for in_src, _ in graph.in_edges(node_key):
                    for _, out_dst in graph.out_edges(node_key):
                        in_node = graph.nodes[in_src]
                        weight = np.prod(in_node["data"]["output_size"]) if not in_node["pooling"]["fused_pooling"] else np.prod(in_node["data"]["pooling"]["pooling_output_size"])
                        graph.add_edge(in_src, out_dst, weight=weight)
                graph.remove_node(node_key)
        self.reduced_graph = graph

    def get_successors(self, node):
        return list(self._graph.successors(node))
