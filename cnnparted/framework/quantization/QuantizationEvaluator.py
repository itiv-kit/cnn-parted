import os
import time

import torch
from torchvision import transforms, datasets
#from model_explorer.models.quantized_model import QuantizedModel
#from model_explorer.utils.data_loader_generator import DataLoaderGenerator

from ..DNNAnalyzer import DNNAnalyzer, buildSequential
from .generate_calibration import generate_calibration

from torchinfo import summary
from torchinfo.layer_info import LayerInfo
from copy import deepcopy


class QuantizationEvaluator():
    def __init__(self, model : torch.nn.Module, dnn : DNNAnalyzer, config : dict, accfunc : callable, showProgress : bool) -> None:
        self.fmodel = model
        self.input_size = dnn.input_size
        self.accfunc = accfunc
        self.models : list[list[LayerInfo]] = []
        self.device = config.get('device')
        self.bits = config.get('bits')
        self.part_points_orig = dnn.partition_points
        self.stats = {}

        t0 = time.time()

        for i in range(0,2):
            if self.bits[i] != 32:
                self.models.append(self._create_quantized_model(self.fmodel, self.bits[i], config.get('calibration')))
            else:
                self.models.append(self.part_points_orig)

        self._eval(config['calibration']['datasets']['calibrate'].get('path'), dnn.partpoints_filtered, showProgress)

        t1 = time.time()
        self.stats['sim_time'] = t1 - t0

    def get_stats(self) -> dict:
        return self.stats

    def _cmp_layers_by_name(self, l1 : LayerInfo, l2 : LayerInfo) -> bool:
        if l1.var_name == l2.var_name:
            if l1.parent_info is None and l2.parent_info is None:
                return True
            elif l1.parent_info is not None and l2.parent_info is not None:
                return self._cmp_layers_by_name(l1.parent_info, l2.parent_info)
            else:
                return False

    def _create_quantized_model(self, m : torch.nn.Module, bit : int, calib_conf : dict) -> list[LayerInfo]:
        model = deepcopy(m)

        gpu_device = torch.device(self.device)
        qmodel = QuantizedModel(model, gpu_device)

        param_path = calib_conf.get('file')
        if not os.path.exists(param_path):
            generate_calibration(deepcopy(m), calib_conf, True, param_path)

        qmodel.load_parameters(param_path)

        bits = [bit] * qmodel.get_explorable_parameter_count()
        qmodel.bit_widths = bits

        modsum = summary(qmodel.base_model, self.input_size, depth=100, verbose=0)
        q_layers = [layer for layer in modsum.summary_list]

        q_partition_points : list[LayerInfo] = []
        q_partition_points.append(self.part_points_orig[0])  # Append Identity Layer
        for point in self.part_points_orig:
            for layer in q_layers:
                if self._cmp_layers_by_name(point, layer):
                    q_partition_points.append(layer)
                    break

        return q_partition_points

    def _eval(self, dir_path : str, part_points : list[LayerInfo], showProgress : bool) -> None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        transf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size[-1]),
            transforms.ToTensor(), normalize
        ])

        dataset = datasets.ImageFolder(dir_path, transf)
        dataloadergen = DataLoaderGenerator(dataset, None, batch_size=64)

        for layer in part_points:
            layer_name = layer.get_layer_name(False, True)

            idx = self.part_points_orig.index(layer) + 1
            layers = self.models[0][:idx]
            layers += self.models[1][idx:]

            seqMod = buildSequential(layers, self.input_size, self.device)
            seqMod.append(torch.nn.Flatten(1))

            # inference loop
            acc = self.accfunc(seqMod, dataloadergen, progress=showProgress, title=f"Infere {layer_name}")

            self.stats[layer_name] = acc.cpu().detach().numpy()
