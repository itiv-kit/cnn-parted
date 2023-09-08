from traitlets.traitlets import All
import networkx as nx

class MemoryInfo:
    def __init__(self) -> None:  
       pass
    
    def get_convs_memory(self,conv_layers):
        all_ifms={}
        all_ofms={}
        all_weights={}
        for layer  in conv_layers:
          name    =layer['name']
          ifms    =layer['conv_params']['ifms']
          ofms    =layer['conv_params']['ofms']
          weights =layer['conv_params']['weights']
          all_ifms[name] = ifms
          all_ofms[name] = ofms
          all_weights[name] = weights

        return all_ofms, all_ifms,all_weights

    def calculate_max_memory_usage(self,graph, weights, ifms, ofms, topological_orders):
        max_memory = {}
        orders_memory = {}

        for order_id, order in enumerate(topological_orders):
          orders_memory[order_id] = []
          ofms_in_memory_list=[]
          visited_node = []
          memory_nodes = {}
          max_memory[order_id] = 0
          memory = ofms[order[0]] 
          ofms_in_memory_list.append(order[0])

          for node_id, node in enumerate(order[1:]):
              memory_nodes[node]=0
              visited_node.append(node)
              parents =  list(graph.predecessors(node))

              # If Last node not a parent we should store its ofms 
              if order[node_id] not in parents:
                 memory += ofms[order[node_id]]
                 ofms_in_memory_list.append(order[node_id])

              for parent in parents:
                suc_list = list(graph.successors(parent))
                all_successors_visited = all(node in visited_node for node in suc_list)

                if parent in ofms_in_memory_list:
                  # If all children are visited ofms of the parent should be removed from the memory
                  if all_successors_visited:
                    memory -= ofms[parent]
                    ofms_in_memory_list.remove(parent)
                elif not all_successors_visited:
                  ofms_in_memory_list.append(parent)
                  memory += ofms[parent]

              if len(parents) == 1:
                  if parents[0] in ofms_in_memory_list:
                    memory += ofms[node]+ weights[node]
                  else:
                     memory += ofms[node]+ ifms[node]+ weights[node]
              else: # if node has multiple inputs 
                 memory += ofms[node]+ifms[node]+ weights[node]

              max_memory[order_id] = max(max_memory[order_id],memory)
              memory_nodes[node] = memory 

              if len(parents) == 1:
                  if parents[0] in ofms_in_memory_list:
                    memory -= ofms[node]
                  else:
                     memory -= ofms[node]+ifms[node] 

              else:
                 memory -= ofms[node]+ifms[node]                
       
          orders_memory[order_id] = memory_nodes

        return max_memory,orders_memory