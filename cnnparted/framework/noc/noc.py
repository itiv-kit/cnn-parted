from abc import abstractmethod
from typing import Literal, Any, TYPE_CHECKING
from dataclasses import dataclass

import networkx as nx

from framework.helpers.dtypes import LinkId, NodeId, NocCoordinate
from framework.link.link import Link
from framework.link.ethernet_link import EthernetLink
from framework.link.network_on_interposer import NoILink
from framework.link.ble_link import BLELink 

type NocComponent = NocLink | NocNode
type NocLinkKind = EthernetLink | NoILink | BLELink
type NocTopologyCfg = Literal["mesh", "fully_connected"]
type NocPath = list[NocLink]

class NocLink: 
    def __init__(self, start: NocCoordinate, end: NocCoordinate, link: Link):
        self.start = start
        self.end = end
        self.link = link
        #self.cur_time = 0.0
        #self.schedule: list[TransferTask] = []
    
    def eval(self, n_bytes: int,):
        latency, energy, slice_size = self.link.eval(n_bytes)
        return latency, energy


class NocNode:
    def __init__(self, id: int, coord: NocCoordinate):
        # Need a more meaningful description here, or just remove it
        self.id = id
        self.coord = coord
        #self.cur_time = 0.0
        #self.schedule = PlatformSchedule(id)
        #self.allocated_tensors: list[AllocatedTensor] = []
    
    def __repr__(self):
        return f"{self.__class__.__name__}(\nid={self.id},\ncoord={self.coord})"


class NocTopology(nx.MultiDiGraph[NocNode] if TYPE_CHECKING else nx.MultiDiGraph):
    # Need to fetch links by start and end
    # need to fetch nodes by node_id
    def __init__(self, **attr: Any):
        super().__init__(**attr)
        self.id_to_link: dict[LinkId, NocLink] = {}
        self.id_to_node: dict[NodeId, NocNode] = {}
        self.dram_neighbors: list[NocCoordinate] = [] #nodes directly next to DRAM
        self.default_link: Link = None

    @abstractmethod
    def singlecast_routing(self, start: NocCoordinate, end: NocCoordinate) -> NocPath:
        ...

    @abstractmethod
    def multicast_routing(self, start: NocCoordinate, ends: list[NocCoordinate]) -> list[NocPath]:
        # ends is a list as a tensor maybe can be broadcast to multiple platforms
        ...
    
    @abstractmethod
    def get_num_hops(self, start: NocCoordinate, end: NocCoordinate):
        ...

    @abstractmethod
    def get_min_hops_to_dram(self, start: NocCoordinate):
        ...
