import networkx as nx

class LayersGraph:

    def __init__(self, model_tree):
        self._graph = nx.DiGraph()
        self._create_layer_relationships(model_tree)
        self._sorted_partition_points= self._sorted_common_nodes()

    def get_Graph(self):
        return self._graph

    def _create_layer_relationships(self, model_tree):
        for layer in model_tree:
            self._graph.add_node(layer['name'], data=layer, children=[])
            for child in layer['children']:
                self._graph.add_edge(layer['name'], child['name'])

    def get_graph_partition_points(self):
        return self._sorted_partition_points

    def _all_paths(self,Graph, start='input', end='output'):
        return list(nx.all_simple_paths(Graph, start, end))
    
    def _sorted_common_nodes(self):
        paths= self._all_paths(self._graph)
        common_nodes = set.intersection(*map(set, paths))       
        # Sort the common nodes to ensure subsequent order
        output = sorted(common_nodes, key=lambda node: list(self._graph.nodes()).index(node))
        return output
    
    def _create_subgraphs(self, Graph, sorted_common_nodes):    
        subgraphs = []

        for i in range(len(sorted_common_nodes) - 1):
            source = sorted_common_nodes[i]
            target = sorted_common_nodes[i + 1]

            # Find all simple paths between source and target
            all_paths = list(nx.all_simple_paths(Graph, source=source, target=target))

            # Create subgraphs from all paths
            subgraph = nx.DiGraph()
            for path in all_paths:
                nx.add_path(subgraph, path)

            subgraphs.append(subgraph)

        return subgraphs
    
    def _get_all_topological_sorts(_graph):
        return list(nx.all_topological_sorts(_graph))