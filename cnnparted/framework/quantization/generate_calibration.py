import os

import torch.nn as nn

try:
    from model_explorer.utils.data_loader_generator import DataLoaderGenerator
    from model_explorer.utils.setup import setup_torch_device

    MODEL_EXPLORER_AVAILABLE = True
except ImportError:
    MODEL_EXPLORER_AVAILABLE = False

    class DummyDataLoaderGenerator:
        pass

    DataLoaderGenerator = DummyDataLoaderGenerator

    def setup_torch_device(*args, **kwargs):
        return "cpu"


from cnnparted.framework.quantization.faulty_quantized_model import FaultyQuantizedModel

try:
    from pytorch_quantization.tensor_quant import QuantDescriptor

    PYTORCH_QUANTIZATION_AVAILABLE = True
except ImportError:
    PYTORCH_QUANTIZATION_AVAILABLE = False

    class DummyQuantDescriptor:
        def __init__(self, *args, **kwargs):
            pass

    QuantDescriptor = DummyQuantDescriptor


def generate_calibration(
    model: nn.Module,
    dataloader_gen: DataLoaderGenerator,
    progress: bool,
    filename: str,
    gpu_device: str,
):
    quant_descriptor = QuantDescriptor(calib_method="histogram")
    qmodel = FaultyQuantizedModel(
        model, gpu_device, quantization_descriptor=quant_descriptor
    )

    qmodel.generate_calibration_file(
        dataloader_gen.get_dataloader(),
        progress,
        calib_method="histogram",
        method="percentile",
        percentile=99.99,
    )

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    qmodel.save_parameters(filename)
