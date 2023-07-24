import os
import inspect

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(this_file_path), '..', '..'))


import torchvision.models as tvmodels
import torchvision.models.segmentation as segmodels
DNN_DICT = {**tvmodels.__dict__, **segmodels.__dict__} # move to constants.py
