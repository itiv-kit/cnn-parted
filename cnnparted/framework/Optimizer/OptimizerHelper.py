class OptimizerHelper:
    def __init__(self):
        pass
    
    def find_best_node(self, nodes_list, optimization_objectives):
        
        # # Define the key function to extract the desired metric for the specified device
        # def key_function(node):
        #     if device == "sensor" and metric == "latency":
        #         return node['sensor_latency']
        #     elif device == "sensor" and metric == "energy":
        #         return node['sensor_energy']
        #     elif device == "link" and metric == "latency":
        #         return node['link_latency']
        #     elif device == "link" and metric == "energy":
        #         return node['link_energy']
        #     elif device == "edge" and metric == "latency":
        #         return node['edge_latency']
        #     elif device == "edge" and metric == "energy":
        #         return node['edge_energy']
        #     elif metric == 'throughput':
        #         return node['throughput']
        #     elif metric == "energy":
        #         return node['energy']
        #     elif metric == "latency":
        #         return node['latency']
        #     else:
        #         raise ValueError("Invalid device or metric.")
        
        best_node =  min(nodes_list, key=lambda node: node.get(optimization_objectives, float('inf')))
        return best_node
