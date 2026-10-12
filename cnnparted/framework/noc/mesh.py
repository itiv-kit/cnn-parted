import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path

from framework.link.link import Link
from framework.helpers.dtypes import NocCoordinate
from framework.helpers.config_helper import ConfigHelper
from .noc import NocLink, NocNode, NocTopology, NodeIdentifier


class MeshNoc(NocTopology):
    """
    Represents an N-D Mesh topology
    """
    def __init__(self, cfg: ConfigHelper, link_model: Link):
        super().__init__()

        noc_cfg = cfg.system.noc
        noc_shape = noc_cfg.shape

        self.default_link = link_model

        assert math.prod(noc_shape) == len(cfg.system.platform_id_mapping), f"For Mesh NoC, every nodes (is: {cfg.system.num_platforms}) must exactly fit into one slot of the NoC (shape is: {noc_shape})"

        noc_layout = np.empty(shape=noc_shape, dtype=NocNode)
        self.noc_ids = np.empty(shape=noc_shape, dtype=int)

        for node_id, info in cfg.system.platform_id_mapping.items():
            coord = NocCoordinate(info.noc_coord)
            node = NocNode(node_id, coord, info.accelerator_name)
            noc_layout[info.noc_coord] = node
            self.noc_ids[info.noc_coord] = node_id
            self.id_to_node[node_id] = node
            self.coord_to_id[info.noc_coord] = node_id
            self.add_node(node)
        
        node_neigbors = self._get_nd_neighbors()
        for node, neighbors in node_neigbors.items():
            for neighbor in neighbors:
                self.add_edge(node, neighbor)

        for source, sink, key in self.edges:
            source_coord = NocCoordinate(source.coord)
            sink_coord = NocCoordinate(sink.coord)
            link = NocLink(source_coord, sink_coord, link_model)
            self.id_to_link[(source.coord, sink.coord)] = link
        
        # Add reverse edge to model bidirectional data transfer
        for u, v, k in self.edges:
            self.add_edge(v, u) #TODO Why no id_to_link?

        
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
            self.dram_neighbors = [tuple(nghbr) for nghbr in noc_cfg.dram.neighbors]
        
        # Setup the DRAM as its own node that is connected to all neighbors
        self.setup_dram_nodes( (-1,)*len(noc_shape) ) #dummy coord
        
        if len(noc_shape) == 2:
            fig = plt.figure()
            pos = {node:node.coord for node in self.nodes}
            nx.draw(self, 
                    pos=pos, 
                    node_color='lightgreen', 
                    with_labels=True,
                    node_size=600)
            fig_dir = Path(cfg.general.work_dir) / "figures" / "mesh.png"
            fig.savefig(fig_dir)
    
    def get_num_hops(self, start: NodeIdentifier, end: NodeIdentifier):
        start = self._resolve_node(start)
        end = self._resolve_node(end)
        visited_nodes = nx.shortest_path(self, start, end)
        return len(visited_nodes) - 1
    
    def get_min_hops_to_dram(self, start) -> int:
        start_node = self._resolve_node(start)
        path_length = nx.shortest_path_length(self, start_node, self.dram_node)
        return int(path_length)
    
    def get_nearest_dram_node(self, start) -> NocCoordinate:
        start_node = self._resolve_node(start)
        path = nx.shortest_path(self, start_node, self.dram_node)
        return path[-1]

    def singlecast_routing(self, start, end) -> list[NocNode]:
        start_node = self._resolve_node(start)
        end_node = self._resolve_node(end)
        nodes_on_path = nx.shortest_path(self, start_node, end_node)
        return nodes_on_path
    
    def multicast_routing(self, start, ends):
        ends = [self._resolve_node(end) for end in ends]
        paths: dict[NocNode, list[list[NocNode]]] = {}
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