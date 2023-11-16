import os
import time

import torch
from .quantizer import QuantizedModel

from model_explorer.utils.data_loader_generator import DataLoaderGenerator
from model_explorer.utils.setup import build_dataloader_generators

from ..DNNAnalyzer import DNNAnalyzer
from .generate_calibration import generate_calibration

from ..node.MNSIMInterface import add_weight_noise

from torchinfo import summary
from torchinfo.layer_info import LayerInfo
from copy import deepcopy
from tqdm import tqdm


class QuantizationEvaluator():
    def __init__(self, dnn : DNNAnalyzer, config : dict, nodes : list, accfunc : callable, showProgress : bool) -> None:
        self.fmodel,self.identity_fmodel = dnn.torchModels
        self.input_size = dnn.input_size
        self.accfunc = accfunc
        self.device = config.get('device')
        self.bits = config.get('bits')
        self.part_points_orig = dnn.partition_points
        self.stats = {}
        self.calib_conf = config.get('calibration')
        gpu_device = torch.device(self.device)
        self.fmodel.to(gpu_device)

        self.nodes = nodes

        t0 = time.time()

        dataloaders = build_dataloader_generators(config['datasets'])
        self.calib_dataloader = dataloaders['calibrate']
        self.part_points_orig_model = self._get_part_points(self.identity_fmodel)
        self.part_points_orig_model = self.part_points_orig_model[1:]

        num_epochs = config['retraining'].get('epochs')
        self._eval(dataloaders['train'], num_epochs, dataloaders['validation'], dnn.partpoints_filtered, showProgress)

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

    def _cmp_layers_by_name0(self, l1 : str, l2 : str) -> bool:
        if l1 ==  l2:
            return True
        return False

    def _get_part_points(self,model):
        modsum = summary(model, self.input_size, depth=100, verbose=0)

        q_layers = [layer for layer in modsum.summary_list]

        _identity = '_identity'
        self.qpoints_dict = {}
        self.qpoints_dict['input'] = q_layers[0].var_name

        for point in self.part_points_orig:
            for layer in q_layers:
                if ('_' + point['name'] + _identity) in layer.var_name:
                    modified_name = layer.var_name.replace('_' + point['name'] + _identity, '')
                    layer_type = point['name'].split('_')
                    modified_name = modified_name.replace('Identity_',layer_type[0] + '_')
                    self.qpoints_dict[point['name']] = modified_name

        self.qpoints_dict['output'] = q_layers[-1].var_name


        q_partition_points : list[LayerInfo] = []
        #q_partition_points.append(qpoints_dict['input'] )  # Append Identity Layer
        for point in self.part_points_orig:
            for layer in q_layers:
                if self._cmp_layers_by_name0(self.qpoints_dict[point['name']], layer.var_name):
                    q_partition_points.append(layer)
                    break

        return q_partition_points



    def _eval(self, train_dataloadergen: DataLoaderGenerator, num_epochs : int, eval_dataloadergen : DataLoaderGenerator, part_points : list[LayerInfo], showProgress : bool) -> None:

        model = deepcopy(self.fmodel)
        gpu_device = torch.device(self.device)
        qmodel = QuantizedModel(model, gpu_device)

        param_path = self.calib_conf.get('file')

        if not os.path.exists(param_path):
            generate_calibration(model, self.calib_dataloader, True, param_path)

        for layer in part_points:
            torch_layer_name =  self.qpoints_dict[layer['name']]

            layer_bits = []
            part1_layers = []
            part2_layers = []

            layer_reached = False
            for base_layer, _ in qmodel.base_model.named_modules():
                if 'quantizer' in base_layer:
                    if layer_reached == True:
                        layer_bits.append(self.bits[1])
                        part2_layers.append(base_layer)
                    else:
                        layer_bits.append(self.bits[0])
                        part1_layers.append(base_layer)
                if torch_layer_name in base_layer:
                    layer_reached = True

            qmodel.bit_widths = layer_bits
            qmodel.load_parameters_file(param_path)

            # Training
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(qmodel.base_model.parameters(), lr=0.0001)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                                step_size=1,
                                                                gamma=0.1)

            qmodel.base_model.train()
            for epoch_idx in range(num_epochs):
                if showProgress:
                    pbar = tqdm(total=len(train_dataloadergen), ascii=True,
                                desc="Epoch {} / {}".format(epoch_idx + 1, num_epochs),
                                position=1)

                running_loss = 0.0
                train_dataloader = train_dataloadergen.get_dataloader()

                for image, target, *_ in train_dataloader:
                    image, target = image.to(gpu_device), target.to(gpu_device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(mode=True):
                        qmodel.base_model.to(gpu_device)
                        output = qmodel.base_model(image)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() * output.size(0)

                        if showProgress:
                            pbar.update(output.size(0))

                lr_scheduler.step()

                if showProgress:
                    pbar.close()

            qmodel.base_model.eval()

            # weight update for PIM-based accelerators
            if self.nodes[0].get('mnsim'):
                qmodel.load_parameters(add_weight_noise(self.nodes[0].get('mnsim')['conf_path'], qmodel.base_model.state_dict(), part1_layers, showProgress))
            elif self.nodes[-1].get('mnsim'):
                qmodel.load_parameters(add_weight_noise(self.nodes[-1].get('mnsim')['conf_path'], qmodel.base_model.state_dict(), part2_layers, showProgress))

            # inference loop
            acc = self.accfunc(qmodel.base_model, eval_dataloadergen, progress=showProgress, title=f"Infere {torch_layer_name}")

            if self.nodes[0].get('mnsim') == None and self.nodes[-1].get('mnsim') == None:
                if self.bits[0] == self.bits[1]:
                    for layer in part_points:
                        #layer_name =  self.qpoints_dict[layer['name']]
                        self.stats[layer['name']] = acc.cpu().detach().numpy()
                    break

            self.stats[layer['name']] = acc.cpu().detach().numpy()
