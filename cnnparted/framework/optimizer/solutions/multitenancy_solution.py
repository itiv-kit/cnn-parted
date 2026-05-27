import operator
from functools import reduce
import json
from dataclasses import dataclass
from collections import defaultdict
import statistics

from framework.helpers.dtypes import ComputeTask
from framework.node.node_evaluator import SystemResult
from framework import ConfigHelper
from framework.scheduling.schedules import SystemSchedule
from framework.model.layer_tracker import LayerBindings

@dataclass
class MultiTenancySolutionElem:
    network_res: dict[str, list[ComputeTask]]
    network_end_times: dict[str, list[int]]
    total_energy: float
    total_latency: float
    total_edp: float
    max_min_span: float


class MultitenancySolution:
    def __init__(self, node_stats: SystemResult, mt_solutions:list[SystemSchedule],  config: ConfigHelper):
        self.nw_names = [wl.name for wl in config.workload.workload_spec]

        self.res: list[MultiTenancySolutionElem] = []
        self.res_summary: list[tuple[float, float]] = [] #Energy, Latency

        for schedule in mt_solutions:
            nw_results: dict[str, list[ComputeTask]] = {nw_name: [] for nw_name in self.nw_names}

            schedule_edp = 0.0
            for node_schedule in schedule.node_schedules.values():
                for task in node_schedule.tasks:
                    nw_results[task.network].append(task)
                    schedule_edp += (task.computation_energy * task.computation_latency) + \
                                        (task.communication_energy * task.communication_latency)

            for nw_tasks in nw_results.values():
                nw_tasks.sort(key=lambda task: task.start_time)

            elem = MultiTenancySolutionElem(network_res=nw_results, 
                                            total_energy=schedule.get_system_energy(node_stats),
                                            total_latency=schedule.system_latency,
                                            network_end_times=schedule.network_finish_times,
                                            total_edp = schedule_edp,
                                            max_min_span=schedule.max_min_span)
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
        
        total_energies = [res.total_energy for res in self.res]
        total_latencies = [res.total_latency for res in self.res]
        min_energy_sol_idx = total_energies.index(min(total_energies))
        min_latency_sol_idx = total_latencies.index(min(total_latencies))
        min_energy_sol = self.res[min_energy_sol_idx]
        min_latency_sol = self.res[min_latency_sol_idx]
        edp_values = [res.total_edp for res in self.res]

        result_json = {
            "other_stats":{
                "median_latency": statistics.median(total_latencies),
                "median_energy": statistics.median(total_energies),
                "median_edp": statistics.median(edp_values),
            },
            "min_energy": {
                "index": min_energy_sol_idx,
                "total_energy": min_energy_sol.total_energy,
                "total_latency": min_energy_sol.total_latency
            },
            "min_latency": {
                "index": min_latency_sol_idx,
                "total_energy": min_latency_sol.total_energy,
                "total_latency": min_latency_sol.total_latency
            },
            "all": []
            }
        for idx, res in enumerate(self.res):
            result_item = {
                "summary": {
                    "index": idx,
                    "total_energy": res.total_energy,
                    "total_latency": res.total_latency,
                    "total_edp": res.total_edp,
                    "max_min_span": res.max_min_span,
                    "network_endtimes": res.network_end_times,
                },
                "network_data": {},
                "system_hardware": {},
            }
            for nw_name, nw_data in res.network_res.items():
                compute_task_list_of_dicts = [task.to_dict() for task in nw_data]
                compute_task_dict_of_dicts = reduce(operator.ior, compute_task_list_of_dicts, {})
                result_item["network_data"][nw_name] = compute_task_dict_of_dicts
            result_json["all"].append(result_item)

        with open(out_path, "w") as f:
           json.dump(result_json, f, indent=4) 
        
    
    def to_numpy(self):
        ...