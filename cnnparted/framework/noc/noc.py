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

# A node can be identified by the node itsef, its coordinate, or the ID of the accelerator
type NodeIdentifier = NocNode | NocCoordinate | int

class NocLink: 
    def __init__(self, start: NocCoordinate, end: NocCoordinate, link: Link):
        self.start = start
        self.end = end
        self.link = link
        #self.cur_time = 0.0
        #self.schedule: list[TransferTask] = []
    
    def eval(self, n_bytes: int, allocated_bw=1.0) -> tuple[float, float]:
        latency, energy, slice_size = self.link.eval(n_bytes, allocated_bw)
        return latency, energy


class NocNode:
    def __init__(self, id: int, coord: NocCoordinate, accel_name: str):
        self.id = id
        self.coord = coord
        self.accelerator_name = accel_name
        #self.cur_time = 0.0
        #self.schedule = PlatformSchedule(id)
        #self.allocated_tensors: list[AllocatedTensor] = []
    
    def __repr__(self):
        return f"NocNode(cord={self.coord}, accel={self.id}_{self.accelerator_name})"

class DramNode(NocNode):
    def __init__(self, coord: NocCoordinate):
        super().__init__(-1, coord, "DRAM")


class NocTopology(nx.MultiDiGraph[NocNode] if TYPE_CHECKING else nx.MultiDiGraph):
    # Need to fetch links by start and end
    # need to fetch nodes by node_id
    def __init__(self, **attr: Any):
        super().__init__(**attr)

        self.id_to_link: dict[LinkId, NocLink] = {}
        self.id_to_node: dict[NodeId, NocNode] = {}
        self.coord_to_id: dict[NocCoordinate, int] = {}

        self.dram_neighbors: list[NocCoordinate] = [] #nodes directly next to DRAM
        self.default_link: Link = None
        self.dram_node: DramNode = None

    @abstractmethod
    def singlecast_routing(self, start: NodeIdentifier, end: NodeIdentifier) -> list[NocNode]:
        ...

    @abstractmethod
    def multicast_routing(self, start: NodeIdentifier, ends: list[NodeIdentifier]) -> list[list[NocNode]]:
        # ends is a list as a tensor maybe can be broadcast to multiple platforms
        ...
    
    @abstractmethod
    def get_num_hops(self, start: NodeIdentifier, end: NodeIdentifier):
        ...

    @abstractmethod
    def get_nearest_dram_node(self, start: NodeIdentifier) -> NocCoordinate:
        ...

    @abstractmethod
    def get_min_hops_to_dram(self, start: NodeIdentifier):
        ...
    
    @property
    def link_ids(self) -> list[LinkId]:
        return list(self.id_to_link.keys())

    def _resolve_node(self, key: NocNode | int | NocCoordinate) -> NocNode:
        if isinstance(key, NocNode):
            return key
        if isinstance(key, int):
            return self.id_to_node[key]
        
        coord: NocCoordinate = key
        node_id = self.coord_to_id[coord]
        return self.id_to_node[node_id]

    
    def setup_dram_nodes(self, dram_coord: NocCoordinate):
        self.dram_node = DramNode(dram_coord)
        self.add_node(self.dram_node)

        for dram_neighbor in self.dram_neighbors:
            neighbor_id = self.coord_to_id[dram_neighbor]
            neighbor_node = self.id_to_node[neighbor_id]

            self.add_edge(self.dram_node, neighbor_node)
            self.add_edge(neighbor_node, self.dram_node)

            link = NocLink(dram_neighbor, self.dram_node.coord, self.default_link)
            self.id_to_link[(neighbor_node.coord, self.dram_node.coord)] = link
            self.id_to_link[(self.dram_node.coord, neighbor_node.coord)] = link