import copy
import json
from pprint import pprint
from collections import defaultdict
import numpy as np

from framework.node.node_evaluator import SystemResult, NetworkResult
from framework.helpers.config import SystemConfig
from framework import ConfigHelper

class MultitenancySolution:
    ...
    def __init__(self, node_stats: SystemResult, config: ConfigHelper):
        ...
        # From node_stats: sort results by network and figure out where they are bound
        self.nw_names = [wl.name for wl in config.workload.workload_spec]
        nw_results_init = {nw_name: NetworkResult() for nw_name in self.nw_names} #initial value to copy layer data over
        self.nw_results = {nw_name: NetworkResult() for nw_name in self.nw_names} #final result, layers sorted by starting time

        for platform, data in node_stats.platforms.items():
            for nw_name_node, nw_data_node in data.designs[0].networks.items():
                for layer_name_node, layer_data_node in nw_data_node:
                    if layer_data_node.binding is not None:
                        nw_results_init[nw_name_node].add_layer(copy.deepcopy(layer_data_node))
                        #self.nw_results[nw_name_node].add_layer(copy.deepcopy(layer_data_node))

        # Sort layers by their starting time
        for nw_name, nw_data in nw_results_init.items():
            sorted_layers = sorted(nw_data.layers.values(), key=lambda layer_data: layer_data.start_time)
            for layer in sorted_layers:
                self.nw_results[nw_name].add_layer(layer)

        #From system_config: get hardware parameters of the resulting architecture
        self.system = defaultdict(str)
        for node in config.system.compute:
            unique_platform_id = str(node.id) + "_" + node.evaluation.accelerator
            self.system[unique_platform_id] = {}
            self.system[unique_platform_id]["design"] = node_stats[node.id].designs[0].architecture

    
    def to_json(self, out_path: str):
        if not out_path.endswith(".json"):
            out_path = out_path + ".json"

        result = {
            "network_data": {},
            "system_hardware": {},
        }

        for nw_name, nw_data in self.nw_results.items():
            result["network_data"][nw_name] = nw_data.to_dict()

        with open(out_path, "w") as f:
           json.dump(result, f, indent=4) 
        
    
    def to_numpy(self):
        ...