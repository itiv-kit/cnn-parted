import os
import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy

from pymoo.core.problem import ElementwiseProblem

from model_explorer.utils.setup import build_dataloader_generators

from ..quantization.FaultyQuantizedModel import FaultyQuantizedModel
from ..quantization.generate_calibration import generate_calibration


class RobustnessProblem(ElementwiseProblem):
    def __init__(
        self,
        model: nn.Module,
        config : dict,
        accuracy_function: callable,
        progress: bool,
        **kwargs,
    ):
        m = deepcopy(model)
        gpu_device = torch.device(config.get('device'))
        self.qmodel = FaultyQuantizedModel(m, gpu_device)

        rob_conf =  config.get('robustness')
        self.min_accuracy = rob_conf.get('min_acc')
        n_constr = len(self.min_accuracy) if isinstance(self.min_accuracy, list) else 1

        self.bits = rob_conf.get('bits')
        assert (
            min(self.bits) > 1
        ), "The lower bound for the bit resolution has to be > 1. 1 bit resolution is not supported and produces NaN."

        super().__init__(
            n_var=self.qmodel.get_explorable_parameter_count(),
            n_constr=n_constr,  # accuracy constraint
            n_obj=2,  # accuracy and low bit num
            xl=0,
            xu=len(self.bits)-1,
            vtype=int,
            kwargs=kwargs
        )

        dataloaders = build_dataloader_generators(config['datasets']) # maybe move to optimizer to enable history?
        calib_dataloadergen = dataloaders['calibrate']
        self.val_dataloadergen = dataloaders['validation']

        calib_conf = config.get('calibration')
        calibration_file = calib_conf.get('file')
        if not os.path.exists(calibration_file):
            generate_calibration(model, calib_dataloadergen, True, calibration_file)

        self.qmodel.load_parameters_file(calibration_file)

        self.accuracy_function = accuracy_function
        self.progress = progress


    def _evaluate(self, x, out, *args, **kwargs):
        layer_bit_nums = []
        for i in x:
            layer_bit_nums.append(self.bits[i])

        self.qmodel.bit_widths = layer_bit_nums
        self.qmodel.base_model.eval()

        accuracy_result = self.accuracy_function(
            self.qmodel.base_model,
            self.val_dataloadergen,
            progress=self.progress,
            title=f"Infere"
        )
        accuracy_result = accuracy_result.cpu().detach().numpy()

        f2_quant_objective = self.qmodel.get_bit_weighted()
        out["F"] = [f2_quant_objective, -accuracy_result]

        g1_accuracy_constraint = 0
        if isinstance(self.min_accuracy, list):
            g1_accuracy_constraint = [(constr - acc_result) for (constr, acc_result) in zip(self.min_accuracy, accuracy_result)]
            out["G"] = g1_accuracy_constraint
        elif isinstance(self.min_accuracy, float):
            g1_accuracy_constraint = self.min_accuracy - accuracy_result
            out["G"] = [g1_accuracy_constraint]
