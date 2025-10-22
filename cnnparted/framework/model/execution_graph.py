import itertools as it

#import framework.stages.inputs.workload_parser as wlparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy

from framework.model.model import LayerParameters 

class ExecutionGraph:
    def __init__(self, model_tree: list[LayerParameters], workload_info):
        self.model_tree = model_tree
        self.workload_info = workload_info

        self.evaluated_layer_types = ["Conv", "Gemm", "MatMul"] #if op_type == 1 it is an input/output layer
        self.pooling_types = ["MaxPool", "GlobalAveragePool"]

        self.full_graph = self._create_full_graph()
        self.reduced_graph = self._create_reduced_graph()

        self.output_sizes = self._set_output_sizes()
    
    def get_graph(self):
        return self.reduced_graph

    def _set_output_sizes(self) -> dict:
        layer_names = [layer.name for layer in self.model_tree]
        layer_outputs = [layer.output_size for layer in self.model_tree]
        return {key: value for key, value in zip(layer_names, layer_outputs)}

    def _create_full_graph(self):
        graph = self._create_layer_relationships()
        graph = self._propagate_shape_info(graph)
        graph = self._fuse_conv_maxpool(graph)
        return graph
    
    def _create_reduced_graph(self):
        reduced_graph = self._delete_not_simulated_nodes()
        return reduced_graph

    def _create_layer_relationships(self):
        graph = nx.DiGraph()
        for layer in self.model_tree:
            if layer.op_type != 'Constant':
                tmp = copy.deepcopy(layer)
                tmp.children = None
                graph.add_node(layer.name, 
                                         data=tmp, 
                                         pooling={"fused_pooling": False}, 
                                         binding=None)
                for child in layer.children:
                    graph.add_edge(layer.name, child.name, weight=np.prod(layer.output_size) )
        return graph
    
    def _propagate_shape_info(self, graph: nx.DiGraph):
        # First, we set the shape for all input layers
        for node_name in graph.nodes:
            if graph.in_degree(node_name) == 0:
                node = graph.nodes[node_name]
                node["data"].input_size = self.workload_info.input_shape
        
        # Next, propagate info about output shape to all successors
        for node_name in graph.nodes:
            node = graph.nodes[node_name]
            node_out_size = node["data"].output_size

            for succ_name  in graph.successors(node_name):
                succ_node = graph.nodes[succ_name]
                succ_node["data"].input_size = node_out_size

        return graph
        

    def _fuse_conv_maxpool(self, graph: nx.DiGraph):
        node_keys = list(graph.nodes)
        for node_key in node_keys:
            node = graph.nodes[node_key]
            successors = graph.succ[node_key]
            if successors is not None:

                # Check if there are multiple successors that produce different output shapes
                # If that is the case, we can't currently process this
                if node["data"].op_type in self.evaluated_layer_types and len(successors) != 1:
                    tmp_out_shapes = []
                    for succ in successors:
                        succ_node = graph.nodes[succ]
                        tmp_out_shapes.append(succ_node["data"].output_size)
                    shape_pairs = np.array(list(it.combinations(tmp_out_shapes, r=2)))
                    pairs_equal = all([np.array_equiv(pair[0], pair[1]) for pair in shape_pairs])

                    if not pairs_equal:
                        raise RuntimeError("Conv with multiple outputs is not supported")

                for succ in successors:
                    succ_node = graph.nodes[succ]
                    
                    # Store information about pooling as pooling nodes will later be deleted
                    if succ_node["data"].op_type in self.pooling_types:
                        node["pooling"]["fused_pooling"] = True
                        node["pooling"]["pooling_output_size"] = succ_node["data"].output_size
        return graph

    def _delete_not_simulated_nodes(self):
        graph = copy.deepcopy(self.full_graph)
        node_keys = list(graph)
        for node_key in node_keys:
            node = graph.nodes[node_key]
            if node["data"].op_type not in self.evaluated_layer_types:
                for in_src, _ in graph.in_edges(node_key):
                    for _, out_dst in graph.out_edges(node_key):
                        in_node = graph.nodes[in_src]
                        weight = np.prod(in_node["data"].output_size) if not in_node["pooling"]["fused_pooling"] else np.prod(in_node["pooling"]["pooling_output_size"])
                        graph.add_edge(in_src, out_dst, weight=weight)
                graph.remove_node(node_key)
        return graph
