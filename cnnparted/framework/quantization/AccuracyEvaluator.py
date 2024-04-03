import os
import torch
import torch.nn as nn

import numpy as np

from copy import deepcopy
from tqdm import tqdm
from typing import Callable

from model_explorer.utils.setup import build_dataloader_generators

from .FaultyQuantizedModel import FaultyQuantizedModel
from .generate_calibration import generate_calibration


class AccuracyEvaluator():
    def __init__(self, model : nn.Module, nodeStats : dict, config : dict, progress : bool) -> None:
        self.bits = [nodeStats[acc].get("bits") for acc in nodeStats]
        self.fault_rates = [nodeStats[acc].get("fault_rates") for acc in nodeStats]
        self.calib_conf = config.get('calibration')
        self.gpu_device = torch.device(config.get('device'))
        self.progress = progress

        m = deepcopy(model)
        self.qmodel = FaultyQuantizedModel(m, self.gpu_device)

        dataloaders = build_dataloader_generators(config['datasets'])
        self.calib_dataloadergen = dataloaders['calibrate']
        self.train_dataloadergen = dataloaders['train']
        self.val_dataloadergen = dataloaders['validation']

        self.param_path = self.calib_conf.get('file')
        if not os.path.exists(self.param_path):
            generate_calibration(model, self.calib_dataloadergen, True, self.param_path)

        self.train_epochs = config['retraining'].get('epochs')


    def eval(self, sols : list, n_constr : int, n_var : int, schedules : list, accuracy_function : Callable) -> list:
        quants = self._gen_quant_list(sols, n_constr, n_var, schedules)
        fault_rates = self._gen_fault_rate_list(sols, n_constr, n_var, schedules)

        # Training
        self.qmodel.bit_widths = np.ones(len(quants[0])) * max(self.bits)
        self.qmodel.load_parameters_file(self.param_path)

        self.qmodel.base_model.train()
        for epoch_idx in range(self.train_epochs):
            if self.progress:
                pbar = tqdm(total=len(self.train_dataloadergen), ascii=True,
                            desc="Epoch {} / {}".format(epoch_idx + 1, self.train_epochs),
                            position=1)

            running_loss = 0.0
            train_dataloader = self.train_dataloadergen.get_dataloader()

            for image, target, *_ in train_dataloader:
                image, target = image.to(self.gpu_device), target.to(self.gpu_device)

                self.qmodel.optimizer.zero_grad()

                with torch.set_grad_enabled(mode=True):
                    self.qmodel.base_model.to(self.gpu_device)
                    output = self.qmodel.base_model(image)
                    loss = self.qmodel.criterion(output, target)
                    loss.backward()
                    self.qmodel.optimizer.step()

                    running_loss += loss.item() * output.size(0)

                    if self.progress:
                        pbar.update(output.size(0))

            self.qmodel.lr_scheduler.step()

            if self.progress:
                pbar.close()

        # Evaluation
        for i, q in enumerate(quants):
            self.qmodel.bit_widths = q
            self.qmodel.fault_rates = fault_rates[i]
            self.qmodel.base_model.eval()
            acc = accuracy_function(self.qmodel.base_model, self.val_dataloadergen, progress=self.progress, title=f"Infere")
            sols[i] = np.append(sols[i], acc.cpu().detach().numpy())

        return quants

    def _gen_quant_list(self, sols : list, n_constr : int, n_var : int, schedules : list) -> list:
        layer_dict = {}
        quant_list = []
        for base_layer in self.qmodel.explorable_module_names:
            quant_list.append(self.bits[0])
            for l in schedules[0]:
                if l in base_layer and l != 'input':
                    layer_dict[l] = len(quant_list) - 1
                    break

        quants = []
        num_pp = n_var/2 - 1
        for sol in sols:
            mapping = sol[n_constr+1:n_var+n_constr+1]
            partition = 0
            for layer in schedules[int(sol[0])]:
                acc = int(mapping[int(n_var/2)+partition])
                if layer in layer_dict.keys():
                    quant_list[layer_dict[layer]-1] = self.bits[acc-1] # input quantizer
                    quant_list[layer_dict[layer]] = self.bits[acc-1]   # weight quantizer
                if partition < num_pp and layer == schedules[int(sol[0])][int(mapping[partition])-1]:
                    partition += 1
            quants.append(deepcopy(quant_list))

        return quants

    def _gen_fault_rate_list(self, sols : list, n_constr : int, n_var : int, schedules : list) -> list:
        layer_dict = {}
        fault_rate_list = []
        for base_layer in self.qmodel.faulty_module_names:
            fault_rate_list.append(self.fault_rates[0])
            for l in schedules[0]:
                if l in base_layer:
                    layer_dict[l] = len(fault_rate_list) - 1
                    break

        fault_rates = []
        num_pp = n_var/2 - 1
        for sol in sols:
            mapping = sol[n_constr+1:n_var+n_constr+1]
            partition = 0
            for layer in schedules[int(sol[0])]:
                acc = int(mapping[int(n_var/2)+partition])
                if layer in layer_dict.keys():
                    fault_rate_list[layer_dict[layer]] = self.fault_rates[acc-1]
                if partition < num_pp and layer == schedules[int(sol[0])][int(mapping[partition])-1]:
                    partition += 1
            fault_rates.append(deepcopy(fault_rate_list))

        return fault_rates
