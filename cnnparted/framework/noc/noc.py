from abc import ABC, abstractmethod
from typing import Literal

from framework.model.model import TensorParameters
from framework.helpers.config_helper import ConfigHelper
from framework.helpers.dtypes import ComputeTask, LinkId, NodeId, NocCoordinate, TransferTask
from framework.link.link import Link
from framework.link.ethernet_link import EthernetLink
from framework.link.network_on_interposer import NoILink
from framework.link.ble_link import BLELink 
from framework.node.node_evaluator import LayerResult, NodeResult

type NocComponent = NocLink | NocNode
type NocLinkKind = EthernetLink | NoILink | BLELink
type NocTopologyCfg = Literal["mesh", "fully_connected"]
type NocPath = list[NocLink]

class NocLink: 
    def __init__(self, start: NocCoordinate, end: NocCoordinate, link: Link):
        self.start = start
        self.end = end
        self.cur_time = 0.0
        self.link = link
        self.schedule: list[TransferTask] = []
    
    def eval(self, n_bytes: int,):
        latency, energy, slice_size = self.link.eval(n_bytes)
        return latency, energy


class NocNode:
    def __init__(self, coord: NocCoordinate,  result: NodeResult = None):
        # Need a more meaningful description here, or just remove it
        # This really should just be a way to access the corresponding NodeResult
        self.coord = coord
        self.result = result
        self.cur_time = 0.0
        self.schedule: list[ComputeTask] = []

    def attach_result(self, result: NodeResult):
        #TODO Probably need some validation here
        self.result = result
    

class NocTopology(ABC):
    # Need to fetch links by start and end
    # need to fetch nodes by node_id
    links: dict[LinkId, NocLink]
    nodes: dict[NodeId, NocNode]
    
    @abstractmethod
    def determine_routing(self, start: NocCoordinate, ends: list[NocCoordinate]) -> list[NocPath]:
        # ends is a list as a tensor maybe can be broadcast to multiple platforms
        ...

    def assign_layer(self, node: NodeId, network: str, layer: LayerResult, earliest_start: float):
        node_cur_time = self.nodes[node].cur_time
        start_time = max(earliest_start, node_cur_time)
        end_time = max(earliest_start, node_cur_time) + layer.latency
        task = ComputeTask(network, layer.name, start_time, end_time)

        self.nodes[node].schedule.append(task)
    
    def assign_tensor(self, link: LinkId, network: str, tensor: TensorParameters):
        #TODO Start time calculation can't be this simple,
        #     need to take into account time at which the layer start computation
        start_time = self.links[link].cur_time

        latency, energy  = self.links[link].eval(n_bytes=tensor.num_elems)
        end_time = start_time + latency
        task = TransferTask(network, tensor.name, start_time, end_time)
        self.links[link].schedule.append(task)