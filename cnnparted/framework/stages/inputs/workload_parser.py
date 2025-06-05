import os
import subprocess
import importlib
import shutil
import torch

from framework.stages.stage_base import Stage
from framework.stages.artifacts import Artifacts
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

class WorkloadParser(Stage):
    def __init___(self):
        super().__init__()

    def run(self, artifacts):

        self._take_artifacts(artifacts)
        try:
            filename = os.path.join(MODEL_PATH, self.run_name, "model.onnx")
            subprocess.check_call(['mkdir', '-p', os.path.join(MODEL_PATH, self.run_name)])
    
            if os.path.isfile(self.model_settings['name']) and self.model_settings['name'].endswith(".onnx"):
                shutil.copy(self.model_settings['name'], filename)
    
                #TODO This is just a temporary workaround, accuracy eval currently not used
                accuracy_function = importlib.import_module(
                    f"{WORKLOAD_FOLDER}.alexnet", package=__package__
                ).accuracy_function
                return accuracy_function
    
    
            model = importlib.import_module(
                f"{WORKLOAD_FOLDER}.{self.model_settings['name']}", package=__package__
            ).model
    
            accuracy_function = importlib.import_module(
                f"{WORKLOAD_FOLDER}.{self.model_settings['name']}", package=__package__
            ).accuracy_function
    
            input_size= self.model_settings['input-size']
            x = torch.randn(input_size)
            torch.onnx.export(model, x, filename, verbose=False, input_names=['input'], output_names=['output'])
    
            self._update_artifacts(artifacts, accuracy_function)
    
        except KeyError:
            print()
            print('\033[1m' + 'Workload not available' + '\033[0m')
            print()
            quit(1)

    def _take_artifacts(self, artifacts: Artifacts):
        self.run_name = artifacts.args["run_name"]
        self.model_settings = artifacts.config["workload"][0] #TODO Currently only considers one workload
        
    def _update_artifacts(self, artifacts: Artifacts, accuracy_func):
        artifacts.set_stage_result(WorkloadParser, "accuracy_function", accuracy_func)
