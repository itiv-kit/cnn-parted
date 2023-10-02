import onnxruntime
import numpy as np
import torch
import time
from typing import List
from .GenericProviders import providers

class GenericNode:
    def __init__(self, config: dict) -> None:
        self.num_threads = config.get('max_threads', torch.get_num_threads())
        self.num_runs = config.get('num_runs', 1000)
        provider = config.get('device', 'cpu')
        
        if provider in providers:
            self.device = providers[provider]
        else:
            supported_keys = list(providers.keys())
            raise ValueError(f"device '{provider}' is not supported. Supported providers: {supported_keys}")

    def run(self, model_path: str, input_size: List[int]) -> dict:
        ort_session = onnxruntime.InferenceSession(model_path, providers=[self.device])
        ort_inputs = {ort_session.get_inputs()[0].name: np.random.randn(*input_size).astype(np.float32)}

        # Warm-up runs
        for _ in range(10):
            ort_session.run(None, ort_inputs)

        run_times = []
        for _ in range(self.num_runs):
            start_time = time.time()
            ort_session.run(None, ort_inputs)
            end_time = time.time()
            run_times.append(end_time - start_time)

        run_times.sort()

        median_time_per_run = run_times[self.num_runs // 2] if self.num_runs % 2 == 1 else \
            (run_times[(self.num_runs - 1) // 2] + run_times[self.num_runs // 2]) / 2.0

        Q1 = run_times[self.num_runs // 4]
        Q3 = run_times[(3 * self.num_runs) // 4]
        IQR = Q3 - Q1

        output = {
            'latency_ms': median_time_per_run * 1e3,
            'latency_iqr': IQR * 1e3,
            'energy_mJ': 1  
        }

        return output
