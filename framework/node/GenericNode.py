import torch
from torch import nn
import torch.utils.benchmark as benchmark
from typing import List

from framework.DNNAnalyzer import buildSequential

def run_model(model : nn.Module, tensor : torch.tensor) -> None:
        model.eval()
        model(tensor)

class GenericNode:
    def __init__(self, config : dict) -> None:
        self.num_threads = torch.get_num_threads()  if not config.get('max_threads') else config['max_threads']
        self.num_runs    = 1000                     if not config.get('num_runs') else config['num_runs']
        self.device      = "cpu"                    if not config.get('device') else config['device']

    def run(self, layers : list, input_size : List[int]) -> float:
        model = buildSequential(layers, input_size, self.device)
        rand_tensor = torch.randn(input_size, device=self.device)

        tim = benchmark.Timer(
            stmt='run_model(m, x)',
            setup='from framework.node.GenericNode import run_model',
            globals={'m' : model, 'x': rand_tensor},
            num_threads=self.num_threads)

        try:
            meas = tim.timeit(self.num_runs)
            output = {
                'latency_ms' : meas.median * 1e3,
                'energy_mJ' : 0
            }
        except RuntimeError as rte:
            print("Warning:", rte.args[0])
            output = {}

        return output
