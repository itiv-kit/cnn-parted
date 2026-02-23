import networkx as nx
import numpy as np
import math
import matplotlib as plt

from framework.helpers.dtypes import NocCoordinate
from framework.helpers.config_helper import ConfigHelper
from framework.link.link_model_interface import LinkModelInterface
from .noc import NocLink, NocNode, NocTopology

class MeshNoc(NocTopology):
    ...
    def __init__(self, cfg: ConfigHelper, link: LinkModelInterface):
        super().__init__()

        noc_cfg = cfg.system.noc
        noc_shape = noc_cfg.shape

        assert math.prod(noc_shape) == len(cfg.system.platform_id_mapping), f"For Mesh NoC, number of nodes (is: {cfg.system.num_platforms}) must be exactly shape of NoC (is: {noc_shape})"

        noc_layout = np.empty(shape=noc_shape, dtype=NocNode)
        self.noc_ids = np.empty(shape=noc_shape, dtype=int)

        for node_id, info in cfg.system.platform_id_mapping.items():
            coord = NocCoordinate(info.noc_coord)
            node = NocNode(node_id, coord)
            noc_layout[info.noc_coord] = node
            self.noc_ids[info.noc_coord] = node_id
            self.id_to_node[node_id] = node
        
        init_graph:nx.Graph = nx.grid_graph(dim=noc_layout.tolist())
        init_graph = init_graph.to_directed()
        self.add_edges_from(init_graph)
        nx.from_pandas_edgelist

        for source, sink, key in self.edges:
            source_coord = NocCoordinate(source.coord)
            sink_coord = NocCoordinate(sink.coord)
            link = NocLink(source_coord, sink_coord, link)
            self.id_to_link[(source.coord, sink.coord)] = link
        
        for u, v, k in self.edges:
            self.add_edge(v, u)
        
        fig = plt.pyplot.figure()
        nx.draw_circular(self, ax=fig.add_subplot(), connectionstyle='arc3, rad = 0.1')
        fig.savefig("mesh.png")
    
    def get_num_hops(self, start: NocCoordinate, end: NocCoordinate):
        hops_in_dim = [abs(start_dim - end_dim ) for start_dim, end_dim in zip(start, end)]
        return sum(hops_in_dim)
    
    def get_min_hops_to_dram(self, start):
        return start[1] #TODO: This assumes DRAM is below NoC
    
    def determine_routing(self, start, ends):
        paths: dict[NocCoordinate, list[NocLink]] = {}
        start_node_id = self.noc_ids[start]
        start_node  = self.id_to_node[start_node_id]
        for end in ends:
            end_node_id = self.noc_ids[end]
            end_node = self.id_to_node[end_node_id]
            path = nx.shortest_path(self, start_node, end_node)
            paths[start] = path

        return paths