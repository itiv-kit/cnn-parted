import os
import random
import numpy as np
import pandas as pd
import networkx as nx

from framework.model.model import TreeModel
from framework.model.graph import LayersGraph

from framework.model.scheduling import topo_sort_random_start_node

class GraphAnalyzer:
    def __init__(self, work_dir: str, run_name : str, input_size : tuple, progress : bool) -> None:
        self.work_dir = work_dir
        self.run_name = run_name
        self.input_size = input_size
        self.progress = progress
        self._Tree_Model = TreeModel(self.run_name, self.input_size)
        self._tree = self._Tree_Model.get_Tree()
        self.torchmodel = self._Tree_Model.get_torchModel()

        self.graph = LayersGraph(self._tree)
        self.conv_layers = self.get_conv2d_layers()

    def find_schedules(self, num_topos : int) -> list:
        #fname_csv = os.path.join(self.work_dir, self.run_name + "_" + "schedules.csv")
        fname_csv = os.path.join(self.work_dir, "schedules.csv")
        if os.path.isfile(fname_csv):
            df = pd.read_csv(fname_csv, header=None, index_col=0)
            self.schedules = df.values.tolist()
        else:
            #try:
            #    self.schedules = topo_sort_random_start_node(G=self.graph.get_Graph(), n=num_topos, seed=0, as_ndarray=True, progress=self.progress)
            #except:
            #    topo_sorts = nx.all_topological_sorts(self.graph.get_Graph())
            #    self.schedules = self._iter_sample_fast(topo_sorts, num_topos)

            self.schedules = []
            self.schedules.append(list(nx.topological_sort(self.graph.get_Graph())))

            self.schedules = np.unique(self.schedules, axis=0)
            df = pd.DataFrame(self.schedules)
            df.to_csv(fname_csv, header=False)
        return self.schedules

    # https://stackoverflow.com/questions/12581437/python-random-sample-with-a-generator-iterable-iterator
    def _iter_sample_fast(self, iterable, samplesize):
        results = []
        iterator = iter(iterable)
        # Fill in the first samplesize elements:
        for _ in range(samplesize): results.append(next(iterator))
        random.shuffle(results)  # Randomize their positions
        for i, v in enumerate(iterator, samplesize):
            r = random.randint(0, i)
            if r < samplesize:
                results[r] = v  # at a decreasing rate, replace random items

        if len(results) < samplesize:
            raise ValueError("Sample larger than population.")
        return results

    def get_timeloop_layers(self):
        output = [layer for layer in self._tree if layer.get("op_type") == "Conv" or layer.get("op_type") == "Gemm" or layer.get("op_type") == "MatMul"]
        return output

    def get_mnsim_layers(self):
        return self._tree

    def get_conv2d_layers(self):
        output = [layer for layer in self._tree if layer.get("op_type") == "Conv"]
        return output

    def get_gemm_layers(self):
        output = [layer for layer in self._tree if layer.get("op_type") == "Gemm" or layer.get("op_type") == "MatMul"]
        return output
