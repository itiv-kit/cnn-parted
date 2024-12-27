import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


from copy import deepcopy

from model_explorer.utils.setup import build_dataloader_generators

from framework.quantization.faulty_quantized_model import FaultyQuantizedModel
from framework.quantization.generate_calibration import generate_calibration

from scipy.optimize import dual_annealing

from typing import Callable

from framework.optimizer.optimizer import Optimizer



class RobustnessOptimizer(Optimizer):
    def __init__(self, work_dir: str, run_name: str, model: nn.Module, accuracy_function: Callable, config: dict, device : str, progress: bool):
        rob_conf = config.get('robustness')
        self.gpu_device = torch.device(device)

        self.accuracy_function = accuracy_function

        self.qmodel = FaultyQuantizedModel(model, self.gpu_device, same_bit_for_weight_and_input=True)
        self.dataset_config = deepcopy(config['datasets'])

        dataloaders = build_dataloader_generators(self.dataset_config)
        calib_dataloadergen = dataloaders['calibrate']
        self.val_dataloadergen = dataloaders['validation']

        calib_conf = self.dataset_config.get('calibrate')
        calibration_file = calib_conf.get('file')
        if not os.path.exists(calibration_file):
            generate_calibration(model, calib_dataloadergen, True, calibration_file, self.gpu_device)

        self.qmodel.load_parameters_file(calibration_file)

        self.n_var = self.qmodel.get_explorable_parameter_count()
        self.bits = rob_conf.get('bits')
        self.max_bits_idx = [len(self.bits) - 2, len(self.bits) - 1]

        self.delta = rob_conf.get('delta')

        self.fname_csv = os.path.join(work_dir, run_name + "_" + "robustness.csv")
        self.progress = progress


    def optimize(self):
        if not os.path.isfile(self.fname_csv):
            acc_lut = {}
            def f(x, ref):
                layer_bit_nums = []
                x = np.round(x).astype(int)

                if tuple(x) in [*acc_lut]:
                    accuracy_result = acc_lut[tuple(x)]
                    if self.progress:
                        print("[RobustenessOptimizer] Skipping inference due to duplicate: ", accuracy_result)
                else:
                    for i in x:
                        layer_bit_nums.append(self.bits[i])
                    self.qmodel.bit_widths = layer_bit_nums
                    self.qmodel.base_model.eval()

                    accuracy_result = self.accuracy_function(
                        self.qmodel.base_model.to(self.gpu_device),
                        self.val_dataloadergen,
                        progress=self.progress,
                        title=f"Infere"
                    )
                    accuracy_result = float(accuracy_result.cpu().detach().numpy())
                    acc_lut[tuple(x)] = accuracy_result

                if ref == 0:
                    return accuracy_result
                else:
                    return self.qmodel.get_bit_weighted() * abs(accuracy_result - ref)

            init_x = np.full(self.n_var, self.max_bits_idx[-1])
            acc_ref = f(init_x, 0) - self.delta

            lw = np.full(self.n_var, 0)
            up = np.full(self.n_var, self.max_bits_idx[-1])
            _ = dual_annealing(f, args=[acc_ref], bounds=list(zip(lw, up)), maxiter=10, x0=init_x)

            for key in acc_lut:
                acc_lut[key] -= acc_ref
                acc_lut[key] = abs(acc_lut[key])

            res = min(acc_lut, key=acc_lut.get)
            constr = [i for i in res]
            constr.append(None)
            constr.append(acc_lut[res])
            constr.append(1)
            df = pd.DataFrame(constr).replace(to_replace=range(0,len(self.bits)), value=self.bits).transpose()
            df.to_csv(self.fname_csv, header=False)
        else:
            df = pd.read_csv(self.fname_csv, header=None, index_col=0)
            data = df.to_numpy()

        data = df.to_numpy()[0]
        constr_dict = {}
        for i, name in enumerate(self.qmodel.explorable_module_names):
            constr_dict[name] = data[int(i/2)]

        return constr_dict
