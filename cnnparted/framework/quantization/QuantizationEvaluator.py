import os
import torch
import torch.nn as nn

import numpy as np

from copy import deepcopy
from tqdm import tqdm
from typing import Callable

from model_explorer.utils.setup import build_dataloader_generators

from .quantizer import QuantizedModel
from .generate_calibration import generate_calibration


class QuantizationEvaluator():
    def __init__(self, model : nn.Module, input_size : tuple, config : dict, progress : bool) -> None:
        self.bits = config.get('bits')
        self.calib_conf = config.get('calibration')
        self.gpu_device = torch.device(config.get('device'))
        self.progress = progress

        m = deepcopy(model)
        self.qmodel = QuantizedModel(m, self.gpu_device)

        dataloaders = build_dataloader_generators(config['datasets'])
        self.calib_dataloadergen = dataloaders['calibrate']
        self.train_dataloadergen = dataloaders['train']
        self.val_dataloadergen = dataloaders['validation']

        self.param_path = self.calib_conf.get('file')
        if not os.path.exists(self.param_path):
            generate_calibration(model, self.calib_dataloadergen, True, self.param_path)

        self.train_epochs = config['retraining'].get('epochs')


    def eval(self, partitions : list, n_var : int, schedules : list, accuracy_function : Callable) -> list:
        quants = self._gen_quant_list(partitions, n_var, schedules)

        # Training
        self.qmodel.bit_widths = np.ones(len(quants[0])) * max(self.bits)
        self.qmodel.load_parameters_file(self.param_path)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.qmodel.base_model.parameters(), lr=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                            step_size=1,
                                                            gamma=0.1)

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

                optimizer.zero_grad()

                with torch.set_grad_enabled(mode=True):
                    self.qmodel.base_model.to(self.gpu_device)
                    output = self.qmodel.base_model(image)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * output.size(0)

                    if self.progress:
                        pbar.update(output.size(0))

            lr_scheduler.step()

            if self.progress:
                pbar.close()

        # Evaluation
        for i, q in enumerate(quants):
            self.qmodel.bit_widths = q
            self.qmodel.base_model.eval()
            acc = accuracy_function(self.qmodel.base_model, self.val_dataloadergen, progress=self.progress, title=f"Infere")
            partitions[i] = np.append(partitions[i], acc.cpu().detach().numpy())

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

        return quants
