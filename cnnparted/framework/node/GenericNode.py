import onnxruntime
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from typing import List

def run_model(ort_session, ort_inputs):
    ort_session.run(None, ort_inputs)


class GenericNode:
    def __init__(self, config: dict) -> None:
        self.num_threads = torch.get_num_threads() if not config.get('max_threads') else config['max_threads']
        self.num_runs = 1000 if not config.get('num_runs') else config['num_runs']
        self.device = 'CPUExecutionProvider' if not config.get('device') or config.get('device')=='cpu' else config['device']

    def run(self, model_path: str, input_size: List[int]) -> float:
        ort_session = onnxruntime.InferenceSession(model_path,providers=[self.device])
        ort_inputs = {ort_session.get_inputs()[0].name: np.random.randn(*input_size).astype(np.float32)}

        tim = benchmark.Timer(
            stmt='run_model(ort_session, ort_inputs)',
            setup='from framework.node.GenericNode import run_model',
            globals={'run_model': run_model, 'ort_session': ort_session, 'ort_inputs': ort_inputs},
            num_threads=self.num_threads)
        try:
            meas = tim.timeit(self.num_runs)
            output = {
                'latency_ms': meas.median * 1e3,
                'latency_iqr': meas.iqr * 1e3,
                'energy_mJ': 0
            }
        except RuntimeError as rte:
            print("Warning:", rte.args[0])
            output = {}

        return output