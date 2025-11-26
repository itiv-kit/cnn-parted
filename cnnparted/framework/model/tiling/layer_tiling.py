import itertools as it

#import framework.stages.inputs.workload_parser as wlparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy

from framework.model.execution_graph import ExecutionGraph
from framework.helpers.config_helper import ConfigHelper

class LayerTiling:
    def __init__(self, exec: dict[str,ExecutionGraph], config: ConfigHelper):
        self.exec = copy.deepcopy(exec)
        #TODO: Specify num of tiles for M, P, Q dimensions for now
        #      no support for tiling of H/W is planned currently, C prob doesn't make a lot of sense
        # Tiling will insert new compute nodes into the graph and remove old ones with smaller size
        # Or maybe it should just update the layer information? And make a note about tiling?

        # Notes on shapes:
        #   batch_size: N
        #   out_channels: M
        #   out_height: P
        #   out_width: Q
        self.num_tiles_n: dict[str, int] = None
        self.num_tiles_m: dict[str, int] = None
        self.num_tiles_p: dict[str, int] = None
        self.num_tiles_q: dict[str, int] = None

        # Dependencies between the shapes (what must be tiles)
        # If N is tiled -> no dependencies
        # If P is tiled -> depends on H (input height inferred automatically by Timeloop)
        # If Q is tiled -> depends on W (input width inferred automatically by Timeloop)
        # If M is tiled -> depends on M
    
    def _tile_reduced_graph(self):

        for nw, graph in self.exec.items():
            num_tiles_n = self.num_tiles_n[nw]
            num_tiles_m = self.num_tiles_m[nw]
            num_tiles_p = self.num_tiles_p[nw]
            num_tiles_q = self.num_tiles_q[nw]
            
            for layer_name, layer_data in graph.reduced_graph.nodes(data=True):
                breakpoint()
                if num_tiles_n > 1:
                    ...
                if num_tiles_m > 1:
                    ...
                
                #TODO Implement these
                #if num_tiles_p > 1:
                #    ...
                #if num_tiles_q > 1:
                #    ...



