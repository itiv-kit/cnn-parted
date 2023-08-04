import os
import time

import torch
from torchvision import transforms, datasets
from model_explorer.models.quantized_model import QuantizedModel
from model_explorer.utils.data_loader_generator import DataLoaderGenerator

from ..DNNAnalyzer import DNNAnalyzer, buildSequential
from .generate_calibration import generate_calibration

from torchinfo import summary
from torchinfo.layer_info import LayerInfo
from copy import deepcopy


class QuantizationEvaluator():
    def __init__(self, model : torch.nn.Module, dnn : DNNAnalyzer, config : dict, accfunc : callable) -> None:
        self.fmodel = model
        self.dnn = dnn
        self.accfunc = accfunc

        self.stats = {}

        self.qmodel = None

        self.q_partition_points : list[LayerInfo] = []
        self.device = config.get('device')

        self.bits = config.get('bits')


        t0 = time.time()

        self._create_quantized_model(self.fmodel, self.bits[0], config.get('calibration'))
        self._eval(config['calibration']['datasets']['calibrate'].get('path'))

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

    def _create_quantized_model(self, m : torch.nn.Module, bit : int, calib_conf : dict) -> QuantizedModel:
        model = deepcopy(m)

        gpu_device = torch.device(self.device)
        qmodel = QuantizedModel(model, gpu_device)

        param_path = calib_conf.get('file')
        if not os.path.exists(param_path):
            generate_calibration(deepcopy(m), calib_conf, True, param_path)

        qmodel.load_parameters(param_path)

        bits = [bit] * qmodel.get_explorable_parameter_count()
        qmodel.bit_widths = bits

        modsum = summary(qmodel.base_model, self.dnn.input_size, depth=100, verbose=0)
        self.q_layers = [layer for layer in modsum.summary_list]

        self.q_partition_points.append(self.dnn.partition_points[0])
        for point in self.dnn.partition_points:
            for layer in self.q_layers:
                if self._cmp_layers_by_name(point, layer):
                    self.q_partition_points.append(layer)
                    break

        self.qmodel = qmodel

    def _eval(self, dir_path : str) -> None:
        part_points = self.dnn.partition_points
        input_size = self.dnn.input_size

        # summary(self.fmodel, input_size, depth=100)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        transf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size[-1]),
            transforms.ToTensor(), normalize
        ])

        dataset = datasets.ImageFolder(dir_path, transf)
        dataloadergen = DataLoaderGenerator(dataset, None, batch_size=64)

        ## DEBUGGING
        # acc = self.accfunc(self.fmodel, dataloadergen, progress=True, title=f"Infere float")
        # print(f"FLOAT:{acc.cpu().detach().numpy()}")

        # acc = self.accfunc(self.qmodel.base_model, dataloadergen, progress=True, title=f"Infere QModel")
        # print(f"QMODEL:{acc.cpu().detach().numpy()}")
        #####

        for layer in self.dnn.partpoints_filtered:
            layer_name = layer.get_layer_name(False, True)

            idx = part_points.index(layer) + 1
            layers = self.q_partition_points[:idx]

            layers += part_points[idx:]

            seqMod = buildSequential(layers, input_size, self.device)
            seqMod.append(torch.nn.Flatten(1))

            # summary(seqMod, input_size, depth=100)

            # inference loop
            acc = self.accfunc(seqMod, dataloadergen, progress=True, title=f"Infere {layer_name}")

            ## DEBUGGING
            # print(f"{layer_name}:{acc.cpu().detach().numpy()}")

            self.stats[layer_name] = acc.cpu().detach().numpy()
