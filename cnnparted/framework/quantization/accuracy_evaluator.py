import os
import torch
import torch.nn as nn

import numpy as np

from copy import deepcopy
from tqdm import tqdm
from typing import Callable, Tuple

from model_explorer.utils.setup import build_dataloader_generators

from framework.quantization.faulty_quantized_model import FaultyQuantizedModel
from framework.quantization.generate_calibration import generate_calibration
from framework.node.node_evaluator import SystemResult


class AccuracyEvaluator():
    def __init__(self, models : dict[str, nn.Module], node_stats : SystemResult, config : dict, device : str, progress : bool) -> None:
        node_stats = node_stats.to_dict()
        self.bits = [node_stats[acc].get("bits") for acc in node_stats]
        self.fault_rates = [node_stats[acc].get("fault_rates") for acc in node_stats]
        self.faulty_bits = [node_stats[acc].get("faulty_bits") for acc in node_stats]
        self.gpu_device = torch.device(device)
        self.progress = progress

        assert len(models) == 1, "AccuracyEvaluator currently only supports a single specified network"
        model = list(models.values())[0]

        m = deepcopy(model)
        self.qmodel = FaultyQuantizedModel(m, self.gpu_device)

        dataloaders = build_dataloader_generators(config['datasets'])
        self.calib_dataloadergen = dataloaders['calibrate']
        self.train_dataloadergen = dataloaders['train']
        self.val_dataloadergen = dataloaders['validation']

        self.param_path = config.get('datasets').get('calibrate').get('file')
        if not os.path.exists(self.param_path):
            generate_calibration(model, self.calib_dataloadergen, True, self.param_path)

        self.train_epochs = config.get('retraining', {'epochs' : 0})['epochs']


    def eval(self, sols : list, n_constr : int, n_var : int, schedules : dict[str, list], accuracy_function : Callable) -> list:
        if not sols:
            return []

        assert len(schedules) == 1, "AccuracyEvaluator currently only supports a single specified network"
        schedules = list(schedules.values())[0]

        quants = self._gen_quant_list(sols, n_constr, n_var, schedules)
        fault_rates, faulty_bits = self._gen_fault_rate_list(sols, n_constr, n_var, schedules)

        # Training
        # self.qmodel.bit_widths = np.ones(len(quants[0])) * max(self.bits)
        # self.qmodel.load_parameters_file(self.param_path)

        # self.qmodel.base_model.train()
        # for epoch_idx in range(self.train_epochs):
        #     if self.progress:
        #         pbar = tqdm(total=len(self.train_dataloadergen), ascii=True,
        #                     desc="Epoch {} / {}".format(epoch_idx + 1, self.train_epochs),
        #                     position=1)

        #     running_loss = 0.0
        #     train_dataloader = self.train_dataloadergen.get_dataloader()

        #     for image, target, *_ in train_dataloader:
        #         image, target = image.to(self.gpu_device), target.to(self.gpu_device)

        #         self.qmodel.optimizer.zero_grad()

        #         with torch.set_grad_enabled(mode=True):
        #             self.qmodel.base_model.to(self.gpu_device)
        #             output = self.qmodel.base_model(image)
        #             loss = self.qmodel.criterion(output, target)
        #             loss.backward()
        #             self.qmodel.optimizer.step()

        #             running_loss += loss.item() * output.size(0)

        #             if self.progress:
        #                 pbar.update(output.size(0))

        #     self.qmodel.lr_scheduler.step()

        #     if self.progress:
        #         pbar.close()

        # Evaluation
        bits_lut = []
        fault_lut = []
        acc_lut = {}
        for i, q in enumerate(quants):
            self.qmodel.bit_widths = q
            self.qmodel.fault_rates = fault_rates[i]
            self.qmodel.faulty_bits = faulty_bits[i]

            acc = None
            if self.qmodel.bit_widths in bits_lut and self.qmodel.fault_rates in fault_lut:
                idx_b = bits_lut.index(self.qmodel.bit_widths)
                idx_f = fault_lut.index(self.qmodel.fault_rates)

                if (idx_b, idx_f) in [*acc_lut]:
                    acc = acc_lut[(idx_b, idx_f)]
                    print("[AccuracyEvaluator] Skipping inference due to duplicate")
            if acc == None:
                self.qmodel.base_model.eval()
                acc = accuracy_function(self.qmodel.base_model, self.val_dataloadergen, progress=self.progress, title=f"Infere")
                acc = acc.cpu().detach().numpy()

                # store to avoid duplicate evaluations
                if self.qmodel.bit_widths in bits_lut:
                    idx_b = bits_lut.index(self.qmodel.bit_widths)
                else:
                    bits_lut.append(self.qmodel.bit_widths)
                    idx_b = len(bits_lut) - 1

                if self.qmodel.fault_rates in fault_lut:
                    idx_f = fault_lut.index(self.qmodel.fault_rates)
                else:
                    fault_lut.append(self.qmodel.fault_rates)
                    idx_f = len(fault_lut) - 1

                acc_lut[(idx_b, idx_f)] = acc

            sols[i] = np.append(sols[i], acc)

        return quants

    def _gen_quant_list(self, sols : list, n_constr : int, n_var : int, schedules : list) -> list:
        layer_dict = {}
        quant_list = []
        for base_layer in self.qmodel.explorable_module_names:
            quant_list.append(self.bits[0])
            for i, l in enumerate(schedules[0]):
                if l in base_layer and l != 'input' and l != 'output':
                    if l in layer_dict and layer_dict[l] % 2:
                        continue
                    layer_dict[l] = len(quant_list) - 1
                    break
                elif l == 'output': # FIXME? hotfix for last layer being renamed in ONNX file
                    layer_dict[schedules[0][i-1]] = len(quant_list) - 1

        quants = []
        num_pp = n_var/2 - 1
        for sol in sols:
            mapping = sol[n_constr+1:n_var+n_constr+1]
            partition = 0
            for layer in schedules[int(sol[0])]:
                acc = int(mapping[int(n_var/2)+partition])
                if layer in layer_dict.keys():
                    if isinstance(self.bits[acc-1], int):
                        bits = self.bits[acc-1]
                    else:
                        assert(isinstance(self.bits[acc-1], dict))
                        bits = self.bits[acc-1][layer]
                    quant_list[layer_dict[layer]-1] = bits # input quantizer
                    quant_list[layer_dict[layer]] = bits   # weight quantizer
                while partition < num_pp and layer == schedules[int(sol[0])][int(mapping[partition])-1]:
                    partition += 1
            quants.append(deepcopy(quant_list))

        return quants

    def _gen_fault_rate_list(self, sols : list, n_constr : int, n_var : int, schedules : list) -> Tuple[list, list]:
        layer_dict = {}
        fault_rate_list = []
        faulty_bit_list = []
        for base_layer in self.qmodel.faulty_module_names:
            fault_rate_list.append(self.fault_rates[0])
            faulty_bit_list.append(self.faulty_bits[0])
            for l in schedules[0]:
                if l in base_layer:
                    if l in layer_dict:
                        continue
                    layer_dict[l] = len(fault_rate_list) - 1
                    break

        fault_rates = []
        faulty_bits = []
        num_pp = n_var/2 - 1
        for sol in sols:
            mapping = sol[n_constr+1:n_var+n_constr+1]
            partition = 0
            for layer in schedules[int(sol[0])]:
                acc = int(mapping[int(n_var/2)+partition])
                if layer in layer_dict.keys():
                    fault_rate_list[layer_dict[layer]] = self.fault_rates[acc-1]
                    faulty_bit_list[layer_dict[layer]] = self.faulty_bits[acc-1]
                while partition < num_pp and layer == schedules[int(sol[0])][int(mapping[partition])-1]:
                    # breakpoint()
                    partition += 1
            fault_rates.append(deepcopy(fault_rate_list))
            faulty_bits.append(deepcopy(faulty_bit_list))

        return fault_rates, faulty_bits
