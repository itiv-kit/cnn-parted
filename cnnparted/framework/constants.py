import os
import inspect

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(this_file_path), '..', '..'))
MODEL_PATH = os.path.join(ROOT_DIR, "onnx_models")

WORKLOAD_FOLDER = "workloads"


import torchvision.models as tvmodels
import torchvision.models.segmentation as segmodels
DNN_DICT = {**tvmodels.__dict__, **segmodels.__dict__}


NUM_JOBS = 64
NUM_TOPOS = 64
