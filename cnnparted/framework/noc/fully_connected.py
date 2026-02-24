import itertools as it

from framework.link.link import Link
from framework.helpers.config_helper import ConfigHelper
from framework.helpers.dtypes import NocCoordinate
from .noc import NocLink, NocNode, NocTopology, NocPath


class FullyConnectedNoc(NocTopology):
    """A network where each node is directly connected with all other nodes"""
    def __init__(self, cfg: ConfigHelper, link_kind: Link):
        super().__init__()

        self.default_link = link_kind
        
        # As all nodes are directly connected, we use the node id as the corrdinate
        for node_id in cfg.system.platform_ids:
            coord = NocCoordinate((node_id,))
            node = NocNode(node_id, coord)
            self.id_to_node[node_id] = node
            self.add_node(node)
        
        # Create a connection between all nodes
        #TODO: Link mapping
        all_node_pairs = it.permutations(cfg.system.platform_ids, r=2)
        for (start, end) in all_node_pairs:
            start_coord = NocCoordinate((start,))
            end_coord = NocCoordinate((end,))
            link = NocLink(start_coord, end_coord, link_kind)
            self.id_to_link[(start, end)] = link

        input_nodes = [node for node in self.id_to_node.values()]
        self.add_edges_from(it.permutations(input_nodes, 2))
    
    def determine_routing(self, start: NocCoordinate, ends: list[NocCoordinate]) -> dict[NocCoordinate, NocPath]:
        # all connections are just one hop
        paths: dict[NocCoordinate, NocPath] = {}
        for end in ends:
            paths[end] = self.links[(start, end)]
        return paths
    
    def get_num_hops(self, start: NocCoordinate, end: NocCoordinate) -> int:
        return 1 #all nodes are directly connected with each other
    
    def get_min_hops_to_dram(self, start: NocCoordinate):
        return 1
