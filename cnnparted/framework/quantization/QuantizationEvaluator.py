import os

import torch
import torch.nn as nn

from copy import deepcopy
from tqdm import tqdm
from typing import Callable

from model_explorer.utils.data_loader_generator import DataLoaderGenerator
from model_explorer.utils.setup import build_dataloader_generators

from .quantizer import QuantizedModel
from .generate_calibration import generate_calibration


class QuantizationEvaluator():
    def __init__(self, model : nn.Module, input_size : tuple, config : dict, progress : bool) -> None:
        self.bits = config.get('bits')
        self.calib_conf = config.get('calibration')
        self.gpu_device = torch.device(config.get('device'))

        m = deepcopy(model)
        self.qmodel = QuantizedModel(m, self.gpu_device)

        dataloaders = build_dataloader_generators(config['datasets'])
        self.calib_dataloader = dataloaders['calibrate']
        self.train_dataloader = dataloaders['train']
        self.val_dataloader = dataloaders['validation']

        self.param_path = self.calib_conf.get('file')
        if not os.path.exists(self.param_path):
            generate_calibration(model, self.calib_dataloader, True, self.param_path)

        self.train_epochs = config['retraining'].get('epochs')


    def eval(self, partitions : list, schedules : list, accuracy_function : Callable) -> list:
        pass
