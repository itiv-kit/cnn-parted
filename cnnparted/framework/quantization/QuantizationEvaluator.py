import os
import numpy as np

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


    def eval(self, partitions : list, n_var : int, schedules : list, accuracy_function : Callable) -> list:
        quants = self._gen_quant_list(partitions, n_var, schedules)

        print(quants)

        return quants

    def _gen_quant_list(self, partitions : list, n_var : int, schedules : list) -> list:
        num_pp = n_var / 2

        layer_dict = {}
        quant_list = []
        for base_layer, _ in self.qmodel.base_model.named_modules():
            if 'quantizer' in base_layer:
                quant_list.append(self.bits[0])
                for l in schedules[0]:
                    if l in base_layer and l != 'input':
                        layer_dict[l] = len(quant_list) - 1
                        break

        quants = []
        for part in partitions:
            partition = 0
            for layer in schedules[int(part[0])]:
                if layer in layer_dict.keys():
                    quant_list[layer_dict[layer]-1] = self.bits[partition] # input quantizer
                    quant_list[layer_dict[layer]] = self.bits[partition]   # weight quantizer
                if partition < num_pp and layer == schedules[int(part[0])][int(part[partition+1])-1]:
                    partition += 1
            quants.append(deepcopy(quant_list))

        return np.unique(quants, axis=0)
