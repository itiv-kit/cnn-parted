import itertools as it
from sortedcontainers import SortedDict

from framework.helpers.dtypes import ComputeTask, NocCoordinate, TransferTask, LinkEvent
from framework.noc.noc import NocTopology, NocLink, NocNode
from framework.model.layers.compute_layer import ComputationLayer
from framework.node.node_evaluator import SystemResult, LayerResult


class PlatformSchedule:
    """
    Stores the schedule for one specific compute platform.  Every platform has 
    a schedule that is modelled as a list. Each entry stores the network, layer name,
    start and end time of the computation.
    """
    def __init__(self, id: int):
        self.id = id
        self.layers: list[LayerResult] = []
        self.cur_time = 0.0

        self.tasks: list[ComputeTask] = []

    def schedule_layer(self, network: str, current_frame: int, earliest_start: float,
                        layer_result: LayerResult, compute_layer: ComputationLayer):
        # Check if the layer can be backfilled between two earlier tasks
        successive_tasks =[(pred, succ, succ.start_time-pred.end_time)  for pred, succ in it.pairwise(self.tasks)]

        # Ideally, we are able to fill the task in between two tasks:
        # |pred_task| - |current_task| - |succ_task|
        # Here, pred and succ refer to a pair of two consecutive, already scheduled tasks
        for pred, succ, slack in successive_tasks:
            if slack >= layer_result.latency:
                if earliest_start <= pred.end_time:
                    # Check if layer can be scheduled immediately after pred
                    start_time = pred.end_time
                    end_time = start_time + layer_result.latency
                    task = ComputeTask(network, current_frame, compute_layer.layer_name,
                                        self.id, start_time, end_time, layer_result.latency,
                                        layer_result.energy)
                    insert_idx = self.tasks.index(succ)
                    self.tasks.insert(insert_idx, task)
                    return task
                elif earliest_start <= succ.start_time and (earliest_start + layer_result.latency < succ.start_time):
                    # Check if layer can be scheduled immediately before succ
                    start_time = earliest_start
                    end_time = start_time + layer_result.latency
                    task = ComputeTask(network, current_frame, compute_layer.layer_name,
                                        self.id, start_time, end_time, layer_result.latency,
                                        layer_result.energy)
                    insert_idx = self.tasks.index(succ)
                    self.tasks.insert(insert_idx, task)
                    return task

        # No backfill possible, append the task to the end and increment cur_time
        start_time = max(earliest_start, self.cur_time)
        end_time = start_time + layer_result.latency

        self.cur_time = start_time + layer_result.latency

        task = ComputeTask(network, current_frame, compute_layer.layer_name, 
                           self.id, start_time, end_time, layer_result.latency,
                           layer_result.energy)
        self.tasks.append(task)
        return task

    def append_layer(self, network: str, current_frame: int, layer_name: str, 
                     layer_result: LayerResult,  start_time: float, 
                     end_time: float = None, latency: float = None):

        if end_time is not None and latency is None:
            latency = end_time - start_time
        elif latency is not None and end_time is None:
            end_time = start_time + latency
        elif latency is not None and end_time is None:
            assert start_time + latency == end_time, "Invalid spec"
        else:
            raise RuntimeError("Invalid usage of append layer: must specify either end_time or latency")

        task = ComputeTask(network, current_frame, layer_name, 
                           self.id, start_time, end_time, latency, 
                           layer_result.energy)
        self.cur_time = max(self.cur_time, end_time)
        self.tasks.append(task)
        return task

    def get_idle_time(self) -> float:
        idle_time = [(nxt.start_time-cur.end_time)  for cur, nxt in it.pairwise(self.tasks)]
        if self.tasks[0].start_time > 0:
            idle_time = [self.tasks[0].start_time] + idle_time
        return sum(idle_time)


class LinkSchedule:
    """
    Manages the schedule of single link between two nodes. It is stored as
    a SortedDict to enable easy access to peak bandwidth utilization
    """
    def __init__(self, start_node: NocNode, end_node: NocNode, link: NocLink):
        self.link = link
        self.start = start_node.id
        self.end = end_node.id
        self.start_coord = start_node.coord
        self.end_coord = end_node.coord
        self.transfer_events = SortedDict()
        self.tasks: list[LinkEvent] = []
    
    @property
    def max_utiliaztion(self):
        max_util = 0.0
        current_util = 0.0
        for time in self.transfer_events.irange(minimum=0.0, maximum=None, inclusive=(False, True)):
            bw_delta = self.transfer_events[time]
            current_util += bw_delta
            if current_util > max_util:
                max_util = current_util
        return max_util

    def schedule_tensor_transfer(self, network: str, tensor_name: str, 
                                link_start_time: float, link_latency: float, link_energy: float, allocated_bw: float):
        link_finish_time = link_start_time + link_latency
        self.transfer_events[link_start_time] =  self.transfer_events.get(link_start_time, 0.0)  + allocated_bw
        self.transfer_events[link_finish_time] = self.transfer_events.get(link_finish_time, 0.0) - allocated_bw

        event = LinkEvent(network, tensor_name,
                          self.start, self.end,
                          link_start_time, link_finish_time,
                          allocated_bw, link_energy)
        self.tasks.append(event)
        return event

class SystemSchedule:
    """
    This class stores start and end time for every compute platform in the system.
    """
    def __init__(self, platform_ids: list[int], networks: list[str], noc: NocTopology):
        self.noc = noc
        self.networks = list(networks)
        self.network_finish_times: dict[str, list[float]] = {nw: [] for nw in networks} #track when each frame of a network is completed
        self.is_feasible = True # solution obtained from CP-SAT solver might not find a feasible schedule

        self.transfer_tasks: list[TransferTask] = []

        # Computation tasks
        self.node_schedules = {id: PlatformSchedule(id) for id in platform_ids}
        self.layers: dict[int, dict[str, dict[str, ComputeTask]]] = {}
        for platform_id in platform_ids:
            self.layers[platform_id] = {nw: {} for nw in networks}

        # Network transfer tasks
        self.links: dict[tuple[int], LinkSchedule] = {}
        for start_node, end_node, key in noc.edges:
            noc_link = noc.id_to_link[ (start_node.coord, end_node.coord) ]
            self.links[(start_node.id, end_node.id)] = LinkSchedule(start_node, end_node, noc_link)

    def schedule_input_transfers(self, platform_id: int, network: str, compute_task: ComputeTask, compute_layer: ComputationLayer) -> TransferTask:
        # Conservative estimation for bandwidth: 
        # all compute platforms must be fed data in parallel
        # all tiles should be able to be executed in parallel
        allocated_bw = min(1/len(self.node_schedules),
                           1/compute_layer.num_tiles,
                           )
        num_elems = sum([t.num_elems for t in compute_layer.inputs])

        # Determine which links will be traversed
        dest_compute_node = self.noc._resolve_node(platform_id)
        start_dram_node = self.noc.get_nearest_dram_node(dest_compute_node)

        visited_nodes = self.noc.singlecast_routing(start_dram_node, dest_compute_node)
        traversed_link_ids = it.pairwise([node.id for node in visited_nodes])
        traversed_links = [self.links[link_ids] for link_ids in traversed_link_ids]

        # For the scheduling, we need to determine the latencies
        # Also the point in time at which a link starts transmitting
        # list of (latency, energy)
        link_eval: list[tuple[float, float]] = [ link_sched.link.eval(num_elems, allocated_bw) for link_sched in traversed_links]
        link_latencies = [leval[0] for leval in link_eval]
        link_energies = [leval[1] for leval in link_eval]
        transfer_start_time = compute_task.start_time - sum(link_latencies)

        link_events = []
        link_start_time = transfer_start_time
        for traversed_link, link_latency, link_energy in zip(traversed_links, link_latencies, link_energies):
            task = traversed_link.schedule_tensor_transfer(network, f"inputs_{compute_layer.layer_name}", 
                                                           link_start_time, link_latency, link_energy,
                                                           allocated_bw)
            link_events.append(task)
            link_start_time += link_latency
        
        # Generate one TransferTask that contains all link events
        task = TransferTask(network, f"inputs_{compute_layer.layer_name}",
                            transfer_start_time, transfer_start_time+sum(link_latencies), start_dram_node, dest_compute_node,
                            allocated_bw, link_events)
        return task

    def schedule_output_transfers(self, platform_id: int, network: str, compute_task: ComputeTask, compute_layer: ComputationLayer) -> TransferTask:
        allocated_bw = min(1/len(self.node_schedules),
                           1/compute_layer.num_tiles,
                           )
        num_elems = compute_layer.output.num_elems

        # Determine which links will be traversed
        start_compute_node = self.noc._resolve_node(platform_id)
        dest_dram_node = self.noc.get_nearest_dram_node(start_compute_node)

        visited_nodes = self.noc.singlecast_routing(start_compute_node, dest_dram_node)
        traversed_link_ids = it.pairwise([node.id for node in visited_nodes])
        traversed_links = [self.links[link_ids] for link_ids in traversed_link_ids]

        # For the scheduling, we need to determine the latencies
        # Also the point in time at which a link starts transmitting
        # list of (latency, energy)
        link_eval: list[tuple[float, float]] = [ link.link.eval(num_elems, allocated_bw) for link in traversed_links]
        link_latencies = [leval[0] for leval in link_eval]
        link_energies = [leval[1] for leval in link_eval]
        transfer_start_time = compute_task.end_time

        link_events = []
        link_start_time = transfer_start_time
        for traversed_link, link_latency, link_energy in zip(traversed_links, link_latencies, link_energies):
            task = traversed_link.schedule_tensor_transfer(network, f"output_{compute_layer.layer_name}", 
                                                           link_start_time, link_latency, link_energy,
                                                           allocated_bw)
            link_events.append(task)
            link_start_time += link_latency

        # Generate one TransferTask that contains all link events
        task = TransferTask(network, f"output_{compute_layer.layer_name}",
                            transfer_start_time, transfer_start_time + sum(link_latencies), 
                            start_compute_node, dest_dram_node,
                            allocated_bw, link_events)
        return task

    def schedule_layer(self, platform_id: int, network: str, current_frame: int, earliest_start: float,
                        layer_result: LayerResult, compute_layer: ComputationLayer) -> ComputeTask:

        # We first determine when the computation should take place, as this not necessarily exactly at earliest_start
        compute_task = self.node_schedules[platform_id].schedule_layer(network, current_frame, earliest_start, layer_result, compute_layer)

        # Next, determine when load and store opertions should begin
        load_task = self.schedule_input_transfers(platform_id, network, compute_task, compute_layer)
        store_task = self.schedule_output_transfers(platform_id, network, compute_task, compute_layer)

        self.transfer_tasks.append(load_task)
        self.transfer_tasks.append(store_task)

        compute_task.loads.append(load_task)
        compute_task.stores.append(store_task)

        # TODO
        #compute_task.end_time = store_task.end_time

        # Keep a dict to make searching for a specifc layer easier
        self.layers[platform_id][network][compute_task.layer] = compute_task
        return compute_task
    
    def append_layer(self, platform_id: int, network: str, current_frame: int, layer_name: str, 
                     layer_result: LayerResult,  start_time: float, 
                     end_time: float = None, latency: float = None):

        compute_task = self.node_schedules[platform_id].append_layer(network, current_frame, layer_name, 
                                                      layer_result, start_time, end_time, latency)
        self.layers[platform_id][network][compute_task.layer] = compute_task
        return compute_task

    def find_task(self, platform_id: int, network: str, frame: int, layer_name: str):
        sched = self.node_schedules[platform_id]
        for task in sched.tasks:
            match = (task.network == network and task.layer == layer_name and task.frame == frame) 
            if match:
                return task
        raise RuntimeError(f"Could not find ComputeTask(network={network}, layer={layer_name}, frame={frame})")

    @property
    def system_latency(self):
        latencies = [sched.cur_time for sched in self.node_schedules.values()]
        return max(latencies)
    
    @property
    def max_min_span(self):
        latencies = [sched.cur_time for sched in self.node_schedules.values()]
        return max(latencies) - min(latencies)


    def get_system_energy(self, node_stats: SystemResult):
        total_energy = 0.0
        for platform, sched in self.node_schedules.items():
            for task in sched.tasks:
                layer_result = node_stats.platforms[platform].designs[0].networks[task.network].layers[task.layer]
                total_energy += layer_result.energy + task.communication_energy
        return total_energy

    def get_network_finish_times(self) -> dict[str, float]:
        networks_finish: dict[str, float] = {network: 0.0 for network in self.networks}
        for node_schedule in self.node_schedules.values():
            for task in node_schedule.tasks:
                networks_finish[task.network] = max(networks_finish[task.network], task.end_time)
        return networks_finish
