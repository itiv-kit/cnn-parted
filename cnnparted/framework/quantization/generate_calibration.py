import torch.nn as nn

from model_explorer.utils.setup import build_dataloader_generators, setup_torch_device
from model_explorer.models.quantized_model import QuantizedModel

from pytorch_quantization.tensor_quant import QuantDescriptor

def generate_calibration(model : nn.Module, config : dict, progress : bool, filename : str):
    dataloaders = build_dataloader_generators(config['datasets'])
    device = setup_torch_device()

    dataset_gen = dataloaders['calibrate']

    quant_descriptor = QuantDescriptor(calib_method='histogram')
    qmodel = QuantizedModel(model, device, quantization_descriptor=quant_descriptor)

    qmodel.generate_calibration_file(dataset_gen.get_dataloader(), progress, calib_method='histogram',
                                     method='percentile', percentile=99.99)

    qmodel.save_parameters(filename)
