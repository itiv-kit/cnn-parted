import itertools as it
import networkx as nx

from framework.link.link import Link
from framework.helpers.config_helper import ConfigHelper
from framework.helpers.dtypes import NocCoordinate
from .noc import NocLink, NocNode, NocTopology, NodeIdentifier


class FullyConnectedNoc(NocTopology):
    """A network where each node is directly connected with all other nodes"""
    def __init__(self, cfg: ConfigHelper, link_kind: Link):
        super().__init__()

        self.default_link = link_kind
        
        # As all nodes are directly connected, we use the node id as the corrdinate
        for node_id, info in cfg.system.platform_id_mapping.items():
            coord = NocCoordinate((node_id,))
            node = NocNode(node_id, coord, info.accelerator_name)
            self.id_to_node[node_id] = node
            self.coord_to_id[(node_id,)] = node_id
            self.add_node(node)
        
        # Create a connection between all nodes
        all_node_pairs = it.permutations(cfg.system.platform_ids, r=2)
        for (start, end) in all_node_pairs:
            start_coord = NocCoordinate((start,))
            end_coord = NocCoordinate((end,))
            link = NocLink(start_coord, end_coord, link_kind)
            self.id_to_link[(start, end)] = link

        input_nodes = [node for node in self.id_to_node.values()]
        self.add_edges_from(it.permutations(input_nodes, 2))
    
    # TODO Fix routing here
    def singlecast_routing(self, start: NodeIdentifier, end: NodeIdentifier):
        start = self._resolve_node(start)
        end = self._resolve_node(end)
        visited_nodes = nx.shortest_path(self, start, end)
        return visited_nodes
    
    def mulitcast_routing(self, start: NodeIdentifier, ends: list[NodeIdentifier]) -> dict[NocNode, list[NocNode]]:
        # all connections are just one hop
        ends = [self._resolve_node(end) for end in ends]
        paths: dict[NocNode, list[list[NocNode]]] = {}
        for end in ends:
            paths[end] = self.singlecast_routing(start, end)
        return paths
    
    def get_num_hops(self, start: NocCoordinate, end: NocCoordinate) -> int:
        return 1 #all nodes are directly connected with each other
    
    def get_min_hops_to_dram(self, start: NocCoordinate):
        return 1
