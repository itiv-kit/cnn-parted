import networkx as nx

class LayersGraph:
    def __init__(self, model_tree):
        self.model_tree = model_tree
        self._graph = nx.DiGraph()
        self._create_layer_relationships()
        self.output_sizes = self._set_output_sizes()

    def get_Graph(self):
        return self._graph

    def _set_output_sizes(self) -> dict:
        layer_names = [layer['name'] for layer in self.model_tree]
        layer_outputs = [layer['output_size'] for layer in self.model_tree]
        return {key: value for key, value in zip(layer_names, layer_outputs)}

    def _create_layer_relationships(self):
        for layer in self.model_tree:
            if layer['op_type']!= 'Constant':
                self._graph.add_node(layer["name"], data=layer, children=[])
                for child in layer["children"]:
                    self._graph.add_edge(layer["name"], child["name"])

    def get_successors(self, node):
        return list(self._graph.successors(node))

    def get_conv2d_layers(self):
        output = [
            layer["name"] for layer in self.model_tree if layer.get("op_type") == "Conv"
        ]
        return output
