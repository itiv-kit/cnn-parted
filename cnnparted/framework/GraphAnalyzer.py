import os
import numpy as np
import pandas as pd

from .model.model import TreeModel
from .model.graph import LayersGraph

from .model.scheduling import topo_sort_random_start_node

class GraphAnalyzer:
    def __init__(self, run_name : str, input_size : tuple, progress : bool) -> None:
        self.run_name = run_name
        self.input_size = input_size
        self.progress = progress
        self._Tree_Model = TreeModel(self.run_name, self.input_size)
        self._tree = self._Tree_Model.get_Tree()
        self.torchmodel = self._Tree_Model.get_torchModel()

        self.graph = LayersGraph(self._tree)
        self.conv_layers = self.get_conv2d_layers()

    def find_schedules(self, num_topos : int) -> list:
        fname_csv = self.run_name + "_" + "schedules.csv"
        if os.path.isfile(fname_csv):
            df = pd.read_csv(fname_csv, header=None, index_col=0)
            self.schedules = df.values.tolist()
        else:
            self.schedules = topo_sort_random_start_node(G=self.graph.get_Graph(), n=num_topos, seed=0, as_ndarray=True, progress=self.progress)
            self.schedules = np.unique(self.schedules, axis=0)
            df = pd.DataFrame(self.schedules)
            df.to_csv(fname_csv, header=False)
        return self.schedules

    def get_timeloop_layers(self):
        output = [layer for layer in self._tree if layer.get("op_type") == "Conv" or layer.get("op_type") == "Gemm"]
        return output

    def get_mnsim_layers(self):
        return self._tree

    def get_conv2d_layers(self):
        output = [layer for layer in self._tree if layer.get("op_type") == "Conv"]
        return output

    def get_gemm_layers(self):
        output = [layer for layer in self._tree if layer.get("op_type") == "Gemm"]
        return output
