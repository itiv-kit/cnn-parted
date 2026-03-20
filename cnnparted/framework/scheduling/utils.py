import math
import itertools as it
import numpy as np
from sortedcontainers import SortedDict

from framework.noc.noc import NocTopology, NocLink
from framework.link.link import Link
from framework.model.tensor_wrapper import TensorWrapper
from framework.model.onnx_model import OnnxModel
from framework.model.layers.compute_layer import ComputationLayer
from framework.node.node_evaluator import SystemResult, LayerResult
from framework.helpers.dtypes import ComputeTask, TransferTask

class PlatformSchedule:
    def __init__(self, id: int):
        self.id = id
        self.layers: list[LayerResult] = []
        self.cur_time = 0.0

        self.tasks: list[ComputeTask] = []
    
    def schedule_layer(self, network: str, layer_result: LayerResult, compute_layer: ComputationLayer):
        # Check if the layer can be backfilled between two earlier tasks
        successive_tasks =[(pred, succ, succ.start_time-pred.end_time)  for pred, succ in it.pairwise(self.tasks)]
        earliest_start = compute_layer.scheduling.earliest_start

        for pred, succ, slack in successive_tasks:
            if slack >= layer_result.latency:
                if earliest_start <= pred.end_time:
                    # Check if layer can be scheduled immediately after pred
                    start_time = pred.end_time
                    end_time = start_time + layer_result.latency
                    task = ComputeTask(network, layer_result.name, self.id, start_time, end_time)
                    insert_idx = self.tasks.index(succ)
                    self.tasks.insert(insert_idx, task)
                    return task
                elif earliest_start <= succ.start_time and (succ.start_time - earliest_start) <= layer_result.latency:
                    # Check if layer can be scheduled immediately before succ
                    start_time = succ.start_time - earliest_start
                    end_time = start_time + layer_result.latency
                    task = ComputeTask(network, layer_result.name, self.id, start_time, end_time)
                    insert_idx = self.tasks.index(succ)
                    self.tasks.insert(insert_idx, task)
                    return task

        # No backfill possible, append the task to the end and increment cur_time
        start_time = max(earliest_start, self.cur_time)
        end_time = start_time + layer_result.latency

        self.cur_time = start_time + layer_result.latency
        
        task = ComputeTask(network, layer_result.name, self.id, start_time, end_time)
        self.tasks.append(task)
        return task
    
    def get_idle_time(self) -> float:
        idle_time = [(nxt.start_time-cur.end_time)  for cur, nxt in it.pairwise(self.tasks)]
        if self.tasks[0].start_time > 0:
            idle_time = [self.tasks[0].start_time] + idle_time
        return sum(idle_time)

class LinkSchedule:
    def __init__(self, start: int, end: int, link: NocLink):
        self.link = link
        self.start = start
        self.end = end
        self.transfer_events = SortedDict()
        self.tasks: list[TransferTask] = []
    
    def schedule_tensor_transfer(self, link: Link, layer_result: LayerResult, tensor: TensorWrapper):
        ...
    
class SystemSchedule:
    def __init__(self, platform_ids: list[int], networks: list[str], noc: NocTopology):
        self.noc = noc
        self.networks = networks

        # Computation tasks
        self.node_schedules = {id: PlatformSchedule(id) for id in platform_ids}
        self.layers: dict[int, dict[str, dict[str, ComputeTask]]] = {}
        for platform_id in platform_ids: 
            self.layers[platform_id] = {nw: {} for nw in networks}
        
        # Network transfer tasks
        self.links: dict[tuple[int], LinkSchedule] = {}
        for start, end, key in noc.edges:
            noc_link = noc.id_to_link[ (start.coord, end.coord) ]
            self.links[(start.id, end.id)] = LinkSchedule(start.id, end.id, noc_link)
    
    def schedule_input_transfers(self, platform_id: int, network: str, compute_layer: ComputationLayer) -> tuple[TransferTask]:
        allocated_bw = 0.05 #TODO
        num_elems = sum([t.num_elems for t in compute_layer.inputs])

        # Determine which links will be traversed
        dest_compute_node = self.noc.id_to_node[platform_id].coord
        start_dram_node = self.noc.get_nearest_dram_node(dest_compute_node)
        if dest_compute_node == start_dram_node:
            task = TransferTask(network, "inputs", 
                                start_time=compute_layer.scheduling.earliest_start,
                                end_time=compute_layer.scheduling.earliest_start,
                                src_node=start_dram_node, dest_node=dest_compute_node,
                                bandwidth=allocated_bw)
            return tuple([task])

        visited_nodes = self.noc.singlecast_routing(start_dram_node, dest_compute_node)
        traversed_link_ids = it.pairwise([node.id for node in visited_nodes])
        traversed_links = [self.links[link_ids] for link_ids in traversed_link_ids]

        # For the scheduling, we need to determine the latencies
        # Also the point in time at which a link starts transmitting
        link_eval: list[tuple[float, float]] = [ link_sched.link.eval(num_elems, allocated_bw) for link_sched in traversed_links]
        link_latencies = [leval[0] for leval in link_eval ]
        cum_latencies: list[float] = np.cumsum(link_latencies).tolist()
        cum_latencies.reverse()
        link_start_times = [compute_layer.scheduling.earliest_start - cum_latency for cum_latency in cum_latencies]

        link_tasks = []
        for traversed_link, link_start, link_latency in zip(traversed_links, link_start_times, link_latencies):
            link_finish = link_start + link_latency
            traversed_link.transfer_events[link_start] = traversed_link.transfer_events.get(link_start, 0.0) + allocated_bw
            traversed_link.transfer_events[link_finish] = traversed_link.transfer_events.get(link_finish, 0.0) + allocated_bw

            task = TransferTask(network, "inputs", 
                                start_time=link_start, end_time=link_finish, 
                                src_node=start_dram_node, dest_node=dest_compute_node,
                                bandwidth=allocated_bw)
            traversed_link.tasks.append(task)
            link_tasks.append(task)
        return tuple(link_tasks)

    def schedule_output_transfers(self, platform_id: int, network: str, compute_task: ComputeTask, compute_layer: ComputationLayer) -> tuple[TransferTask]:
        allocated_bw = 0.05 #TODO
        num_elems = compute_layer.output.num_elems

        # Determine which links will be traversed
        start_compute_node = self.noc.id_to_node[platform_id].coord
        dest_dram_node = self.noc.get_nearest_dram_node(start_compute_node)
        if start_compute_node == dest_dram_node:
            task = TransferTask(network, "inputs", 
                                start_time=compute_layer.scheduling.end_time,
                                end_time=compute_layer.scheduling.end_time,
                                src_node=start_compute_node, dest_node=dest_dram_node,
                                bandwidth=allocated_bw)
            return tuple([task])
        visited_nodes = self.noc.singlecast_routing(start_compute_node, dest_dram_node)
        traversed_link_ids = it.pairwise([node.id for node in visited_nodes])
        traversed_links = [self.links[link_ids] for link_ids in traversed_link_ids]

        # For the scheduling, we need to determine the latencies
        # Also the point in time at which a link starts transmitting
        link_eval: list[tuple[float, float]] = [ link.link.eval(num_elems, allocated_bw) for link in traversed_links]
        link_latencies = [leval[0] for leval in link_eval]
        cum_latencies: list[float] = np.cumsum(link_latencies).tolist()
        link_start_times = [compute_task.end_time + cum_latency for cum_latency in cum_latencies]

        link_tasks = []
        for traversed_link, link_start, link_latency in zip(traversed_links, link_start_times, link_latencies):
            link_finish = link_start + link_latency
            traversed_link.transfer_events[link_start] = traversed_link.transfer_events.get(link_start, 0.0) + allocated_bw
            traversed_link.transfer_events[link_finish] = traversed_link.transfer_events.get(link_finish, 0.0) + allocated_bw

            task = TransferTask(network, "output", 
                                start_time=link_start, end_time=link_finish, 
                                src_node=start_compute_node, dest_node=dest_dram_node,
                                bandwidth=allocated_bw)
            traversed_link.tasks.append(task)
            link_tasks.append(task)
        return tuple(link_tasks)
        
    def schedule_layer(self, platform_id: int, network: str, layer_result: LayerResult, compute_layer: ComputationLayer) -> ComputeTask:
        load_tasks = self.schedule_input_transfers(platform_id, network, compute_layer)

        compute_task = self.node_schedules[platform_id].schedule_layer(network, layer_result, compute_layer)

        store_tasks = self.schedule_output_transfers(platform_id, network, compute_task, compute_layer)
        store_complete = max([store_task.end_time for store_task in store_tasks])

        #compute_task.end_time = store_complete
        
        # Keep a dict to make searching for a specifc layer easier
        self.layers[platform_id][network][compute_task.layer] = compute_task 
        return compute_task

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
    
    def get_network_finish_times(self) -> dict[str, float]:
        networks_finish: dict[str, float] = {network: 0.0 for network in self.networks}
        for node_schedule in self.node_schedules.values():
            for task in node_schedule.tasks:
                networks_finish[task.network] = max(networks_finish[task.network], task.end_time)
        return networks_finish


def validate_schedule(schedule: SystemSchedule, models: dict[str, OnnxModel]):
    are_constraints_fulfilled = True
    network_end_times = schedule.get_network_finish_times()
    for network, model in models.items():
        for layer in model.nodes:
            if not isinstance(layer, ComputationLayer):
                continue
            layer_task = schedule.layers[layer.scheduling.binding][network][layer.layer_name]
            for succ in model.successors(layer):
                # Check that all succesors actually start after their
                # predecessor is finished
                succ_task = schedule.layers[succ.scheduling.binding][network][succ.layer_name]
                assert succ_task.start_time >= layer_task.end_time, f"Layer {succ.layer_name} of {network} is not scheduled after predecssor {layer.layer_name}"
        
        # Check if this schedule fulfills the constraints
        if model.deadline != math.inf and network_end_times[network] > model.deadline:
            # Scheduling with a given deadline
            are_constraints_fulfilled = False
        elif model.frames > 1 and network_end_times[network] > 1_000:
            # Scheduling a fixed amount of frames per second (here: 1000ms)
            are_constraints_fulfilled = False
    
    return are_constraints_fulfilled


