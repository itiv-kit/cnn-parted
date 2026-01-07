import copy
import json
from dataclasses import dataclass
from collections import defaultdict

from framework.node.node_evaluator import SystemResult, NetworkResult
from framework import ConfigHelper
from framework.scheduling import SystemSchedule
from framework.optimizer.multitenancy_partitioning import LayerTracker

@dataclass
class MultiTenancySolutionElem:
    network_res: dict[str, NetworkResult]
    total_energy: float
    total_latency: float


class MultitenancySolution:
    def __init__(self, node_stats: SystemResult, mt_solutions:list[tuple[LayerTracker, SystemSchedule]],  config: ConfigHelper):
        self.nw_names = [wl.name for wl in config.workload.workload_spec]

        self.res: list[dict[str, NetworkResult]] = []
        self.res_summary: list[tuple[float, float]] = [] #Energy, Latency

        self.res: list[MultiTenancySolutionElem] = []

        for tracker, schedule in mt_solutions:
            nw_results_init = {nw_name: NetworkResult() for nw_name in self.nw_names} #initial value to copy layer data over
            nw_results_sorted = {nw_name: NetworkResult() for nw_name in self.nw_names} 
            for network in tracker.network_info.keys():
                for layer, binding in tracker.network_info[network].items():
                    # Find the layer data in node_stats and update it with scheduling information
                    layer_data = copy.deepcopy(node_stats[binding].designs[0].networks[network].layers[layer])
                    layer_schedule = schedule.layers[binding][network][layer]
                    layer_data.start_time = layer_schedule.start_time
                    layer_data.end_time = layer_schedule.end_time
                    layer_data.binding = binding
                    nw_results_init[network].add_layer(layer_data)
            # Sort layers by their starting time
            for nw_name, nw_data in nw_results_init.items():
                sorted_layers = sorted(nw_data.layers.values(), key=lambda layer_data: layer_data.start_time)
                for layer in sorted_layers:
                    nw_results_sorted[nw_name].add_layer(layer)
            
            elem = MultiTenancySolutionElem(network_res=nw_results_sorted, 
                                            total_energy=schedule.get_system_energy(node_stats),
                                            total_latency=schedule.system_latency)
            self.res.append(elem)

        #From system_config: get hardware parameters of the resulting architecture
        self.system = defaultdict(str)
        for node in config.system.compute:
            unique_platform_id = str(node.id) + "_" + node.evaluation.accelerator
            self.system[unique_platform_id] = {}
            self.system[unique_platform_id]["design"] = node_stats[node.id].designs[0].architecture

    
    def to_json(self, out_path: str):
        if not out_path.endswith(".json"):
            out_path = out_path + ".json"

        result_json = []
        for res in self.res:
            result_item = {
                "summary": {
                    "total_energy": res.total_energy,
                    "total_latency": res.total_latency,
                },
                "network_data": {},
                "system_hardware": {},
            }
            for nw_name, nw_data in res.network_res.items():
                result_item["network_data"][nw_name] = nw_data.to_dict()
            result_json.append(result_item)

        with open(out_path, "w") as f:
           json.dump(result_json, f, indent=4) 
        
    
    def to_numpy(self):
        ...