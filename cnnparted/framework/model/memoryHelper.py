from traitlets.traitlets import All
import numpy as np

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
        all_ifms['dummy_conv']= 0
        all_ofms['dummy_conv']= 0
        all_weights['dummy_conv']= 0
        return all_ofms, all_ifms,all_weights

    def get_memory_for_2_partitions(self,partition_points,mems,num_bytes):
      mem_1 = {}
      mem_2 = {}
      last_mem_1 = {}
      last_mem_2 = {}
      for point in partition_points:
          # For mem_1
          max_fmaps_1 = 0
          total_weights_1 = 0
          if point["name"] in mems:
              for key in mems:
                  max_fmaps_1 = max(max_fmaps_1, mems[key]['fmaps'])
                  total_weights_1 += mems[key]['weights']
                  if key == point["name"]:
                      break
              mem_1[point["name"]] = (max_fmaps_1+ total_weights_1)*num_bytes
              last_mem_1 = mem_1[point["name"]]
          else:
              mem_1[point["name"]] = last_mem_1
          # For mem_2

          max_fmaps_2 = 0
          total_weights_2 = 0
          passed_point = False
          if point["name"] in mems:
              for key in mems:
                  if key == point["name"]:
                      passed_point = True
                  if passed_point:
                      max_fmaps_2 = max(max_fmaps_2, mems[key]['fmaps'])
                      total_weights_2 += mems[key]['weights']
              mem_2[point["name"]] = (max_fmaps_2+ total_weights_2)*num_bytes #{'fmaps': max_fmaps_2, 'weights': total_weights_2}
              last_mem_2= mem_2[point["name"]]
          else:
              if point["name"]== "output":
                 mem_2[point["name"]] = 0 # need output size of the nnet
              else:
                mem_2[point["name"]] = last_mem_2

      return mem_1, mem_2

    def _calculate_max_memory_usage(self,graph, weights, ifms, ofms, topological_orders):
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

        weights_mem=0
        for node in topological_orders[0]:
           weights_mem +=  weights[node]


        return max_memory,weights_mem,orders_memory


    def get_max_conv2d_layer(self, graph, conv_layers,input_size):

      ofms, ifms, weights = self.get_convs_memory(conv_layers)
      convs_subgraphs, subgraphs_ids, root, dummy_convs = graph.get_all_conv_subgraphs()
      for dummy in dummy_convs:
          ofms[dummy] = 0
          ifms[dummy] = 0
          weights[dummy] = 0
      subgraph_mem = {}
      inp_mem = {"fmaps": np.prod(input_size), "weights": 0}
      subgraph_mem.update({"input": inp_mem})
      memory_dict = {"fmaps": ifms[root] + ofms[root], "weights": weights[root]}
      subgraph_mem.update({root: memory_dict})
      for i, subgraph in enumerate(convs_subgraphs):
          orders = graph.get_all_topological_orders(subgraph)
          subgraph_max_memory, weight_mem, orders_memory = self._calculate_max_memory_usage(
              subgraph, weights, ifms, ofms, orders
          )
          subgraph_min_memory_necessary = min(subgraph_max_memory.values())
          if orders[0][-1] not in dummy_convs:
              memory_dict = {"fmaps": subgraph_min_memory_necessary, "weights": weight_mem}
              subgraph_mem.update({orders[0][-1]: memory_dict})
      return subgraph_mem
