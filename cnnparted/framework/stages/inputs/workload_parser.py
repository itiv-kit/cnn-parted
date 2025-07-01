import os
import subprocess
import importlib
import shutil
from dataclasses import dataclass
from collections.abc import Callable
import torch

from framework.stages.stage_base import Stage
from framework.stages.artifacts import Artifacts
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

@dataclass
class WorkloadInfo:
    accuracy_function: Callable
    input_shape: list[int]
    onnx_filepath: str


class WorkloadParser(Stage):
    def __init___(self):
        super().__init__()

    def run(self, artifacts):

        self._take_artifacts(artifacts)

        workloads: dict[str, WorkloadInfo] = {}
        networks = [wl["name"] for wl in self.workload_array]
        for workload in self.workload_array:
            try:
                wl_name = workload["name"]

                filename = os.path.join(MODEL_PATH, self.run_name, wl_name + "_model.onnx")
                subprocess.check_call(['mkdir', '-p', os.path.join(MODEL_PATH, self.run_name)])
        
                # Check if the model is specified as a locally available ONNX file
                if os.path.isfile(workload['name']) and workload['name'].endswith(".onnx"):
                    shutil.copy(workload['name'], filename)

                    #TODO Accuracy evaluation is currently not supported, if possible integrate a way to specify it
                    accuracy_function = None
                    workloads[wl_name] = WorkloadInfo(accuracy_function, workload["input-size"], filename)
        
                else:
                    model = importlib.import_module(
                        f"{WORKLOAD_FOLDER}.{workload['name']}", package=__package__
                    ).model
            
                    accuracy_function = importlib.import_module(
                        f"{WORKLOAD_FOLDER}.{workload['name']}", package=__package__
                    ).accuracy_function
        
                    input_size = workload['input-size']
                    x = torch.randn(input_size)
                    torch.onnx.export(model, x, filename, verbose=False, input_names=['input'], output_names=['output'])

                workloads[wl_name] = WorkloadInfo(accuracy_function, workload["input-size"], filename)
    
    
            except KeyError:
                print()
                print('\033[1m' + 'Workload not available' + '\033[0m')
                print()
                quit(1)

            self._update_artifacts(artifacts, accuracy_function, networks, workloads)

    def _take_artifacts(self, artifacts: Artifacts):
        self.run_name = artifacts.args["run_name"]
        self.workload_array = artifacts.config["workload"]
        
    def _update_artifacts(self, artifacts: Artifacts, accuracy_func, networks, workloads):
        artifacts.set_stage_result(WorkloadParser, "accuracy_function", accuracy_func)
        artifacts.set_stage_result(WorkloadParser, "networks", networks)
        artifacts.set_stage_result(WorkloadParser, "workloads", workloads)

