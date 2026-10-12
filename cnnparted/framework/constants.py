import inspect
import os
from pathlib import Path

import torchvision.models as tvmodels
import torchvision.models.segmentation as segmodels

this_file_path = str(Path(__file__).resolve())
#this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(this_file_path), '..', '..'))
MODEL_PATH = os.path.join(ROOT_DIR, "onnx_models")

WORKLOAD_FOLDER = "workloads"


DNN_DICT = {**tvmodels.__dict__, **segmodels.__dict__}
