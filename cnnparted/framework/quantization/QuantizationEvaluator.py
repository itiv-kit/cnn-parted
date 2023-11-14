import os
import time

import torch
from .quantizer import QuantizedModel

from model_explorer.utils.data_loader_generator import DataLoaderGenerator
#from .data_loader_generator import DataLoaderGenerator
#from .setup import build_dataloader_generators
from model_explorer.utils.setup import build_dataloader_generators

from ..DNNAnalyzer import DNNAnalyzer
from .generate_calibration import generate_calibration

from torchinfo import summary
from torchinfo.layer_info import LayerInfo
from copy import deepcopy
from tqdm import tqdm


class QuantizationEvaluator():
    def __init__(self, dnn : DNNAnalyzer, config : dict, accfunc : callable, showProgress : bool) -> None:
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


        t0 = time.time()

        dataloaders = build_dataloader_generators(config['datasets'])
        self.calib_dataloader = dataloaders['calibrate']
        self.part_points_orig_model = self._get_part_points(self.identity_fmodel)
        self.part_points_orig_model = self.part_points_orig_model[1:]

        # for i in range(0,2):
        #     if self.bits[i] != 32:
        #         self._create_quantized_model(self.fmodel, self.bits[i], config.get('calibration'), dataloaders['calibrate'])

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

        qmodel.load_parameters(param_path)


        for layer in part_points:
            torch_layer_name =  self.qpoints_dict[layer['name']]

            seqMod = qmodel.create_model(self.fmodel,torch_layer_name,self.bits)

            # Training
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(seqMod.parameters(), lr=0.0001)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                                step_size=1,
                                                                gamma=0.1)

            seqMod.train()
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
                        output = seqMod(image)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() * output.size(0)

                        if showProgress:
                            pbar.update(output.size(0))

                lr_scheduler.step()

                if showProgress:
                    pbar.close()

            # inference loop
            seqMod.eval()
            acc = self.accfunc(seqMod, eval_dataloadergen, progress=showProgress, title=f"Infere {torch_layer_name}")

            if self.bits[0] == self.bits[1]:
                for layer in part_points:
                    #layer_name =  self.qpoints_dict[layer['name']]
                    self.stats[layer['name']] = acc.cpu().detach().numpy()
                break

            self.stats[layer['name']] = acc.cpu().detach().numpy()
