import networkx as nx
import numpy as np
import math
import matplotlib as plt
from pathlib import Path

from framework.link.link import Link
from framework.helpers.dtypes import NocCoordinate
from framework.helpers.config_helper import ConfigHelper
from .noc import NocLink, NocNode, NocTopology

class MeshNoc(NocTopology):
    """
    Represents an N-D Mesh topology
    """
    def __init__(self, cfg: ConfigHelper, link: Link):
        super().__init__()

        noc_cfg = cfg.system.noc
        noc_shape = noc_cfg.shape

        self.default_link = link

        assert math.prod(noc_shape) == len(cfg.system.platform_id_mapping), f"For Mesh NoC, every nodes (is: {cfg.system.num_platforms}) must exactly fit into one slot of the NoC (shape is: {noc_shape})"

        noc_layout = np.empty(shape=noc_shape, dtype=NocNode)
        self.noc_ids = np.empty(shape=noc_shape, dtype=int)

        for node_id, info in cfg.system.platform_id_mapping.items():
            coord = NocCoordinate(info.noc_coord)
            node = NocNode(node_id, coord, info.accelerator_name)
            noc_layout[info.noc_coord] = node
            self.noc_ids[info.noc_coord] = node_id
            self.id_to_node[node_id] = node
            self.add_node(node)
        
        node_neigbors = self._get_nd_neighbors()
        for node, neighbors in node_neigbors.items():
            for neighbor in neighbors:
                self.add_edge(node, neighbor)

        for source, sink, key in self.edges:
            source_coord = NocCoordinate(source.coord)
            sink_coord = NocCoordinate(sink.coord)
            link = NocLink(source_coord, sink_coord, link)
            self.id_to_link[(source.coord, sink.coord)] = link
        
        # Add reverse edge to model bidirectional data transfer
        for u, v, k in self.edges:
            self.add_edge(v, u)
        
        # Configure DRAM neighbor nodes
        if noc_cfg.dram.location is not None:
            assert len(noc_shape) == 2, "Can only specify DRAM location key for 2D Mesh"
            match noc_cfg.dram.location:
                case "above":
                    max_y = noc_shape[1]-1
                    self.dram_neighbors = [(x, max_y) for x in range(noc_shape[0])]
                case "below":
                    self.dram_neighbors = [(x, 0) for x in range(noc_shape[0])]
                case "left":
                    self.dram_neighbors = [(0, y) for y in range(noc_shape[1])]
                case "right":
                    max_x = noc_shape[0]-1
                    self.dram_neighbors = [(max_x, y) for y in range(noc_shape[1])]
        else:
            self.dram_neighbors = noc_cfg.dram.neighbors
        
        if len(noc_shape) == 2:
            fig = plt.pyplot.figure()
            pos = {node:node.coord for node in self.nodes}
            nx.draw(self, 
                    pos=pos, 
                    node_color='lightgreen', 
                    with_labels=True,
                    node_size=600)
            fig_dir = Path(cfg.general.work_dir) / "figures" / "mesh.png"
            fig.savefig(fig_dir)
    
    def get_num_hops(self, start: NocCoordinate, end: NocCoordinate):
        hops_by_dim = [abs(start_dim - end_dim ) for start_dim, end_dim in zip(start, end)]
        return sum(hops_by_dim)
    
    def get_min_hops_to_dram(self, start) -> int:
        hops_to_dram_neighbor = [self.get_num_hops(start, dram_neighbor) for dram_neighbor in self.dram_neighbors]
        return min(hops_to_dram_neighbor)

    def singlecast_routing(self, start, end) -> list[NocNode]:
        start_node_id = self.noc_ids[start]
        start_node  = self.id_to_node[start_node_id]
        end_node_id = self.noc_ids[end]
        end_node = self.id_to_node[end_node_id]
        nodes_on_path = nx.shortest_path(self, start_node, end_node)
        return nodes_on_path
    
    def multicast_routing(self, start, ends):
        paths: dict[NocCoordinate, list[list[NocNode]]] = {}
        for end in ends:
            paths[end] = self.singlecast_routing(start, end)
        return paths
    
    def _get_nd_neighbors(self):
        """
        Determines neighbors for objects in an n-dimensional mesh.
        Returns a dict: {element: [list of neighbor objects]}
        """
        # 1. Map coordinates to objects
        # Assumes obj.coords is a tuple, e.g., (x, y, z, w...)
        spatial_map = {tuple(obj.coord): obj for obj in self.nodes}
        
        neighbors_registry:dict[NocNode, list[NocNode]] = {}

        for obj in self.nodes:
            coords = tuple(obj.coord)
            n_dims = len(coords)
            obj_neighbors = []

            # 2. Iterate through each dimension
            for i in range(n_dims):
                # Check both directions (+1 and -1) for the current dimension
                for delta in [-1, 1]:
                    neighbor_coords = list(coords)
                    neighbor_coords[i] += delta
                    
                    # Look up the potential neighbor
                    neighbor = spatial_map.get(tuple(neighbor_coords))
                    if neighbor:
                        obj_neighbors.append(neighbor)
            
            neighbors_registry[obj] = obj_neighbors
            
        return neighbors_registry