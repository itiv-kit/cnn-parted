import os
import subprocess
import importlib
import shutil
from dataclasses import dataclass
from collections.abc import Callable
import torch
from loguru import logger

from framework.helpers.dtypes import WorkloadInfo
from framework.stages.stage_base import Stage
from framework.stages.artifacts import Artifacts
from framework.constants import MODEL_PATH, WORKLOAD_FOLDER


class WorkloadParser(Stage):
    def __init___(self):
        super().__init__()

    def run(self, artifacts):
        logger.info("Parsing workload configuration")
        self._take_artifacts(artifacts)

        workloads: dict[str, WorkloadInfo] = {}
        networks = [wl.name for wl in self.workload_array.workload_spec]
        for workload in self.workload_array.workload_spec:
            try:
                wl_name = workload.name

                filename = os.path.join(MODEL_PATH, self.run_name, workload.dest_model_fname)
                os.makedirs(os.path.join(MODEL_PATH, self.run_name), exist_ok=True)
        
                # Check if the model is specified as a locally available ONNX file
                if os.path.isfile(workload.src_model_fname) and workload.src_model_fname.endswith(".onnx"):
                    logger.debug(f"Instantiate workload {wl_name} from ONNX file and input-size {workload.input_size}")
                    shutil.copy(workload.src_model_fname, filename)

                    #TODO Accuracy evaluation is currently not supported, if possible integrate a way to specify it
                    accuracy_function = None
        
                else:
                    logger.debug(f"Instantiate workload {wl_name} from PyTorch repository and input-size {workload.input_size}")
                    model = importlib.import_module(
                        f"{WORKLOAD_FOLDER}.{workload.name}", package=__package__
                    ).model
            
                    accuracy_function = importlib.import_module(
                        f"{WORKLOAD_FOLDER}.{workload.name}", package=__package__
                    ).accuracy_function
        
                    input_size = workload.input_size
                    x = torch.randn(input_size)
                    torch.onnx.export(model, x, filename, verbose=False, input_names=['input'], output_names=['output'])

                workloads[wl_name] = WorkloadInfo(accuracy_function, workload, filename)
    
            except KeyError:
                print()
                logger.error(f"Workload {workload.name} not found. Make sure to specify a valid workload name or path to an ONNX file in the configuration.")
                print()
                quit(1)

        self._update_artifacts(artifacts, accuracy_function, networks, workloads)

    def _take_artifacts(self, artifacts: Artifacts):
        self.run_name = artifacts.config.args.run_name
        self.workload_array = artifacts.config.workload #["workload"]
        
    def _update_artifacts(self, artifacts: Artifacts, accuracy_func, networks: list[str], workloads: dict[str, WorkloadInfo]):
        artifacts.workload_parser.accuracy_function = accuracy_func
        artifacts.workload_parser.networks = networks
        artifacts.workload_parser.workloads = workloads


@dataclass
class WorkloadParserArtifacts:
    accuracy_function: callable = None
    networks: list[str] = None
    workloads: dict[str, WorkloadInfo] = None