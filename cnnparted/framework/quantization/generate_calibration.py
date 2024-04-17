import torch.nn as nn
import os

from model_explorer.utils.setup import setup_torch_device
from .FaultyQuantizedModel import FaultyQuantizedModel
from model_explorer.utils.data_loader_generator import DataLoaderGenerator

from pytorch_quantization.tensor_quant import QuantDescriptor

def generate_calibration(model : nn.Module, dataloader_gen : DataLoaderGenerator, progress : bool, filename : str, gpu_device : str):
    quant_descriptor = QuantDescriptor(calib_method='histogram')
    qmodel = FaultyQuantizedModel(model, gpu_device, quantization_descriptor=quant_descriptor)

    qmodel.generate_calibration_file(dataloader_gen.get_dataloader(), progress, calib_method='histogram',
                                     method='percentile', percentile=99.99)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    qmodel.save_parameters(filename)
