import networkx as nx
import copy

class LayersGraph:
    def __init__(self, model_tree):
        self.model_tree = model_tree
        self._graph = nx.DiGraph()
        self._create_layer_relationships()
        self.paths =  list(nx.all_simple_paths(self._graph, "input", "output"))
        self._sorted_partition_points =  self._sorted_common_nodes(
             self._graph, self.paths
         )

    def get_Graph(self):
        return self._graph

    def get_all_simple_paths(self, graph):
        src, end = self._get_start_end_graph(graph)

        paths = list(nx.all_simple_paths(graph, src, end))

        return paths

    # def get_all_simple_paths_eff(self, graph):
    #     sorted_common_nodes=["input","Mul_3","Mul_5","Mul_7","Add_0","Mul_13","Mul_15",
    #     "Add_1","Mul_23","Add_2","Add_3","Mul_33","Mul_35","Add_4","Mul_45","Add_6","Add_7"
    #     ,"Add_8","output"]
    #     _,_,paths = self._create_subgraphs(graph, sorted_common_nodes)
    #     all_paths = self.stitch_paths(paths, "input", "output",sorted_common_nodes)
    #     return all_paths


    def stitch_paths(self, subgraph_paths, start_node, end_node, sorted_common_nodes):
    # The initial stitched paths are the paths in the first subgraph that start with the start_node

        stitched_paths = {start_node: []}
        for path in subgraph_paths[0]:
            if path[0] == start_node:
                stitched_paths[start_node].append(path)

        # Iterate over pairs of subgraph paths and stitch them together
        for i in range(len(sorted_common_nodes) - 2):  # -2 because we look ahead by one, and the last node has no out-paths
            new_stitched = []
            # Get the end node of the current segment to find the next starting paths
            current_end_node = sorted_common_nodes[i+1]
            next_node_paths = subgraph_paths[i + 1]  # Paths to stitch next
            for path in stitched_paths.get(current_end_node, []):
                # Check all the paths in the next subgraph and see if we can stitch them to the current path
                for next_path in next_node_paths:
                    if next_path[0] == path[-1]:  # Check if paths can be stitched
                        # Avoid including the last node of the current path twice
                        new_stitched.append(path + next_path[1:])
            # The new stitched paths become the current stitched paths for the next iteration
            stitched_paths[current_end_node] = new_stitched

        # At the end, we're interested in paths that reach the end_node
        # Return all paths that end with the end_node
        return [path for path in stitched_paths.get(sorted_common_nodes[-2], []) if path[-1] == end_node]

    def _create_layer_relationships(self):
        for layer in self.model_tree:
            if layer['op_type']!= 'Constant':
                self._graph.add_node(layer["name"], data=layer, children=[])
                for child in layer["children"]:
                    self._graph.add_edge(layer["name"], child["name"])

    def get_graph_partition_points(self):
        return self._sorted_partition_points

    def _all_paths(self, Graph, start=None, end=None):
        return list(nx.all_simple_paths(Graph, start, end))

    def _sorted_common_nodes(self, graph, paths):
        common_nodes = set.intersection(*map(set, paths))
        # Sort the common nodes to ensure subsequent order
        output = sorted(common_nodes, key=lambda node: list(graph.nodes()).index(node))
        return output

    def _create_subgraphs(self, Graph, sorted_common_nodes):
        subgraphs = []
        subgraph_ids = []
        paths=[]

        for i in range(len(sorted_common_nodes) - 1):
            source = sorted_common_nodes[i]
            target = sorted_common_nodes[i + 1]
            unique_id = f"{source}->{target}"
            subgraph_ids.append(unique_id)

            # Find all simple paths between source and target
            all_paths = list(nx.all_simple_paths(Graph, source=source, target=target))
            paths.append(all_paths)
            # Create subgraphs from all paths
            subgraph = nx.DiGraph()
            for path in all_paths:
                nx.add_path(subgraph, path)

            subgraphs.append(subgraph)

        return subgraphs,subgraph_ids,paths

    # optimized _create subgraphs for long chains: needs to be tested first
    def opt_create_subgraphs(self, Graph, sorted_common_nodes):
        subgraphs = []

        i = 0
        while i < (len(sorted_common_nodes) - 1):
            source = sorted_common_nodes[i]
            if len(list(Graph.successors(source))) > 1:
                target = sorted_common_nodes[i + 1]
                i = i + 1
            else:
                for j in range(i + 1, (len(sorted_common_nodes) - 1)):
                    node = sorted_common_nodes[j]
                    if len(list(Graph.successors(node))) > 1:
                        target = sorted_common_nodes[j]
                        i = j
                        break
                    i = j

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

    def get_successors(self, graph, node):
        return list(graph.successors(node))

    def get_conv_graph(self):
        dummy_convs=[]
        conv_nodes = self.get_conv2d_layers()
        graph = copy.deepcopy(self._graph)
        for node in self._graph.nodes:
            if node not in conv_nodes:
                predecessors = list(graph.predecessors(node))
                successors = list(graph.successors(node))

                if len(predecessors) + len(successors) > 2:
                    dummy= node+"_dummy_conv"
                    graph = nx.relabel_nodes(graph, {node: dummy})
                    dummy_convs.append(dummy)

                else:
                    graph.remove_node(node)
                    for pred in predecessors:
                        for succ in successors:
                            graph.add_edge(pred, succ)
        return graph,dummy_convs

    def get_conv2d_layers(self):
        output = [
            layer["name"] for layer in self.model_tree if layer.get("op_type") == "Conv"
        ]
        return output

    def _get_start_end_graph(self, graph):
        source = next(
            (node for node, in_degree in graph.in_degree() if in_degree == 0), None
        )
        target = next(
            (node for node, out_degree in graph.out_degree() if out_degree == 0), None
        )
        return source, target

    def get_all_conv_subgraphs(self):
        graph,dummy_convs = self.get_conv_graph()
        source = next(
            (node for node, in_degree in graph.in_degree() if in_degree == 0), None
        )
        target = next(
            (node for node, out_degree in graph.out_degree() if out_degree == 0), None
        )

        paths = list(nx.all_simple_paths(graph, source, target))
        nodes_c_nodes = self._sorted_common_nodes(graph, paths)
        subgraphs,ids,_ = self._create_subgraphs(graph, nodes_c_nodes)

        return subgraphs,ids, source,dummy_convs

    def get_all_topological_orders(self, graph):
        return list(nx.all_topological_sorts(graph))

    def find_the_nearest_ancestor(self,source,node_list):
        _source=source

        if _source == None:
            return None
        if 'dummy_conv' in  _source:
                _source = source.replace('_dummy_conv','')



        graph_reversed= nx.reverse(self._graph)
        reverse_simple_paths = list(nx.all_simple_paths(graph_reversed, source=_source, target=node_list[0]))

        if reverse_simple_paths:
            reverse_simple_path = reverse_simple_paths[0]
        else:
            reverse_simple_path = []

        nearest_node = None
        nodes = [d["name"] for d in node_list]
        for path_node in reverse_simple_path:
            if path_node in nodes:
                nearest_node = path_node
                break

        return nearest_node

    def find_the_nearest_descendant(self,source,node_list):

        _source=source
        if _source == None:
            return None
        if 'dummy_conv' in  _source:
                _source = source.replace('_dummy_conv','')
        graph = self._graph
        simple_paths = list(nx.all_simple_paths(graph, source=_source, target=node_list[0]))

        if simple_paths:
            simple_path = simple_paths[0]
        else:
            simple_path = []

        nearest_node = None
        nodes = [d["name"] for d in node_list]
        for path_node in simple_path:
            if path_node in nodes:
                nearest_node = path_node
                break

        return nearest_node
