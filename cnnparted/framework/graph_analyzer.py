import os
import random
import numpy as np
import pandas as pd
import networkx as nx

#from framework.stages.inputs.workload_parser import WorkloadInfo
from framework.model.model import TreeModel
from framework.model.graph import LayersGraph

from framework.model.scheduling import topo_sort_random_start_node

class GraphAnalyzer:
    def __init__(self, work_dir: str, run_name : str, 
                 workloads: dict, progress : bool) -> None:
        self.work_dir = work_dir
        self.run_name = run_name
        self.workloads = workloads
        self.networks = [nw for nw in workloads.keys()]
        self.progress = progress

        self._tree_models: dict[str, TreeModel] = {}
        self._trees: dict[str, list] = {}
        self.torchmodels  = {}
        self.graphs: dict[str, LayersGraph] = {}
        for network, workload_info in self.workloads.items():
            self._tree_models[network] = TreeModel(self.run_name, network, tuple(workload_info.input_shape))
            self._trees[network] = self._tree_models[network].get_tree()
            self.torchmodels[network] = self._tree_models[network].get_torch_model()
            self.graphs[network] = LayersGraph(self._trees[network])
        self.schedules: dict[str, list] = {}


    def find_schedules(self, num_topos : int) -> list:
        #fname_csv = os.path.join(self.work_dir, self.run_name + "_" + "schedules.csv")
        for network in self.networks:
            fname_csv = os.path.join(self.work_dir, network + "_schedules.csv")
            if os.path.isfile(fname_csv):
                df = pd.read_csv(fname_csv, header=None, index_col=0)
                self.schedules[network] = df.values.tolist()
            else:
                #try:
                #    self.schedules = topo_sort_random_start_node(G=self.graphs[network].get_graph(), n=num_topos, seed=0, as_ndarray=True, progress=self.progress)
                #except:
                #    topo_sorts = nx.all_topological_sorts(self.graphs[network].get_graph())
                #    self.schedules = self._iter_sample_fast(topo_sorts, num_topos)

                self.schedules[network] = []
                self.schedules[network].append(list(nx.topological_sort(self.graphs[network].get_graph())))

                self.schedules[network] = np.unique(self.schedules[network], axis=0)
                df = pd.DataFrame(self.schedules[network])
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

    def get_timeloop_layers(self, network: str):
        output = [layer for layer in self._trees[network] if layer.get("op_type") == "Conv" or layer.get("op_type") == "Gemm" or layer.get("op_type") == "MatMul"]
        return output

    def get_mnsim_layers(self, network: str):
        return self._trees[network]

    def get_conv2d_layers(self, network: str):
        output = [layer for layer in self._trees[network] if layer.get("op_type") == "Conv"]
        return output

    def get_gemm_layers(self, network: str):
        output = [layer for layer in self._trees[network] if layer.get("op_type") == "Gemm" or layer.get("op_type") == "MatMul"]
        return output
