import itertools as it

from framework.node.node_evaluator import SystemResult, LayerResult
from framework.helpers.dtypes import ComputeTask, TransferTask

class PlatformSchedule:
    def __init__(self, id: int):
        self.id = id
        self.layers: list[LayerResult] = []
        self.cur_time = 0.0

        self.tasks: list[ComputeTask | TransferTask] = []
    
    def get_time(self):
        return self.cur_time
    
    def assign_layer(self, network: str, layer: LayerResult, earliest_start: float):
        start_time = max(earliest_start, self.cur_time)
        end_time = start_time + layer.latency

        self.cur_time = start_time + layer.latency
        
        task = ComputeTask(network, layer.name, start_time, end_time)
        self.tasks.append(task)
        return task
    
    def assign_tensor(self, network: str, tensor):
        ...
    
    def get_idle_time(self) -> float:
        idle_time = [(nxt.start_time-cur.end_time)  for cur, nxt in it.pairwise(self.tasks)]
        if self.tasks[0].start_time > 0:
            idle_time = [self.tasks[0].start_time] + idle_time
        return sum(idle_time)
    

class SystemSchedule:
    def __init__(self, platform_ids: list[int], networks: list[str]):
        self.node_schedules = {id: PlatformSchedule(id) for id in platform_ids}
        self.networks = networks
        self.layers: dict[int, dict[str, dict[str, ComputeTask]]] = {}
        for platform_id in platform_ids: 
            self.layers[platform_id] = {nw: {} for nw in networks}

    def assign_layer(self, platform_id: int, network: str, layer: LayerResult, earliest_start: float):
        task = self.node_schedules[platform_id].assign_layer(network, layer, earliest_start)
        
        # Keep a dict to make searching for a specifc layer easier
        self.layers[platform_id][network][task.layer] = task 

    @property
    def system_latency(self):
        latencies = [sched.cur_time for sched in self.node_schedules.values()]
        return max(latencies)
    
    def get_system_energy(self, node_stats: SystemResult):
        total_energy = 0.0
        for platform, sched in self.node_schedules.items():
            for task in sched.tasks:
                layer_result = node_stats.platforms[platform].designs[0].networks[task.network].layers[task.layer]
                total_energy += layer_result.energy
        return total_energy