import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from loguru import logger

from framework.helpers.dtypes import WorkloadInfo
from framework.helpers.config_helper import ConfigHelper
from framework.model.onnx_model import OnnxModel, OnnxParser 
from framework.model.tiling.layer_tiling import LayerTiling

from framework.scheduling import topo_sort_random_start_node  # noqa: F401

class GraphAnalyzer:
    def __init__(self, work_dir: str, run_name : str, 
                 workloads: dict[str, WorkloadInfo], progress : bool,
                 config: ConfigHelper) -> None:
        self.work_dir = work_dir
        self.run_name = run_name
        self.workloads = workloads
        self.networks = [nw for nw in workloads.keys()]
        self.progress = progress
        self.graph_dot_dir = os.path.join(self.work_dir, "figures", "graphs")
        os.makedirs(self.graph_dot_dir, exist_ok=True)

        self.models: dict[str, OnnxModel] = {}
        self.torchmodels  = {}
        for network, workload_info in self.workloads.items():
            logger.debug(f"Parsing Model for network {network}")
            onnx_parser = OnnxParser(self.run_name, network)
            onnx_model = onnx_parser.parse_graph()
            onnx_model.deadline = workload_info.deadline
            onnx_model.frames = workload_info.frames

            self.models[network]  = onnx_model
            self.torchmodels[network] = onnx_parser.get_torch_model() 
            nx.nx_pydot.write_dot(onnx_model, os.path.join(self.graph_dot_dir, f"{network}.dot") )
        
        # Perform layer tiling and store results as a DOT file for debugging/visualization
        layer_tiling = LayerTiling(config)
        layer_tiling.tile_layers_inplace(self.models)
        for network, model in self.models.items():
            nx.nx_pydot.write_dot(model, os.path.join(self.graph_dot_dir, f"{network}_tiled.dot") )

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
                #self.schedules[network].append(list(nx.topological_sort(self.graphs[network].get_graph())))
                topo_sort_graph = [layer.layer_name for layer in nx.topological_sort(self.models[network])]
                self.schedules[network].append(topo_sort_graph)

                self.schedules[network] = np.unique(self.schedules[network], axis=0)
                df = pd.DataFrame(self.schedules[network])
                df.to_csv(fname_csv, header=False)
        return self.schedules

    # https://stackoverflow.com/questions/12581437/python-random-sample-with-a-generator-iterable-iterator
    def _iter_sample_fast(self, iterable, samplesize):
        results = []
        iterator = iter(iterable)
        # Fill in the first samplesize elements:
        for _ in range(samplesize):
            results.append(next(iterator))
        random.shuffle(results)  # Randomize their positions
        for i, v in enumerate(iterator, samplesize):
            r = random.randint(0, i)
            if r < samplesize:
                results[r] = v  # at a decreasing rate, replace random items

        if len(results) < samplesize:
            raise ValueError("Sample larger than population.")
        return results

    def get_timeloop_layers(self, network: str):
        output = [layer for layer in self.models[network].nodes if layer.op_type == "Gemm" or layer.op_type == "MatMul" or layer.op_type == "Conv"]
        return output

    def get_mnsim_layers(self, network: str):
        return self.models[network], self._trees[network]

    def get_conv2d_layers(self, network: str):
        output = [layer for layer in self.models[network].nodes if layer.op_type == "Conv"]
        return output

    def get_gemm_layers(self, network: str):
        output = [layer for layer in self.models[network].nodes if layer.op_type == "Gemm" or layer.op_type == "MatMul"]
        return output
