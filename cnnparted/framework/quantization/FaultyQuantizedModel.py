import torch
import numpy as np
import pandas as pd


from tqdm import tqdm
from torch import nn as torch_nn

from torch.utils.data import DataLoader

import framework.quantization.FaultyQModules as qmodules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization import tensor_quant
from pytorch_quantization.tensor_quant import QuantDescriptor

from model_explorer.exploration.weighting_functions import bits_weighted_linear
from framework.quantization.CustomModel import CustomModel


class FaultyQuantizedModel(CustomModel):
    """The quantized model automatically replaces all Conv2d and Linear modules
    with quantizeable counterparts from the nvidia-quantization library.
    """

    def __init__(self,
                 base_model: torch_nn.Module,
                 device: torch.device,
                 same_bit_for_weight_and_input = False,
                 weighting_function: callable = bits_weighted_linear,
                 quantization_descriptor: QuantDescriptor = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                 dram_analysis_file: str = "") -> None:
        super().__init__(base_model, device)

        self._fault_rates = []
        self.faulty_modules = []
        self.faulty_module_names = []

        self._bit_widths = {}
        self.same_bit_for_weight_and_input = same_bit_for_weight_and_input
        self.weighting_function = weighting_function
        self.quantization_descriptor = quantization_descriptor

        self.input_quantizers = []
        self.weight_quantizers = []

        # supposingly this is not going to change
        self._create_quantized_model()

        # Energy Model ...
        if dram_analysis_file != "":
            self._build_energy_model(dram_analysis_file)

    @property
    def fault_rates(self):
        return self._fault_rates

    @fault_rates.setter
    def fault_rates(self, new_fault_rates):
        assert isinstance(new_fault_rates, list) or isinstance(
            new_fault_rates,
            np.ndarray), "fault_rates have to be a list or ndarray"
        assert len(new_fault_rates) == len(
            self.faulty_modules
        ), "fault_rates list has to match the amount of quantization layers"

        # Update Model ...
        for i, module in enumerate(self.faulty_modules):
            module.fault_rate = new_fault_rates[i]

        self._fault_rates = new_fault_rates

    @property
    def bit_widths(self):
        return self._bit_widths

    @bit_widths.setter
    def bit_widths(self, new_bit_widths):
        assert isinstance(new_bit_widths, list) or isinstance(
            new_bit_widths,
            np.ndarray), "bit_width have to be a list or ndarray"
        assert len(new_bit_widths) == self.get_explorable_parameter_count(
            ), "bit_width list has to match the amount of quantizable layers"

        # Update Model ...
        if self.same_bit_for_weight_and_input:
            for i in range(len(self.input_quantizers)):
                self.input_quantizers[i].num_bits = new_bit_widths[i]
                self.weight_quantizers[i].num_bits = new_bit_widths[i]
        else:
            for i, module in enumerate(self.explorable_modules):
                module.num_bits = new_bit_widths[i]

        self._bit_widths = new_bit_widths

    def get_explorable_parameter_count(self) -> int:
        if self.same_bit_for_weight_and_input:
            assert len(self.input_quantizers) == len(self.weight_quantizers)
            return len(self.input_quantizers)
        else:
            return len(self.explorable_modules)

    def get_bit_weighted(self) -> int:
        return self.weighting_function(self.explorable_modules,
                                       self.explorable_module_names)

    def get_forward_pass_dram_energy(self) -> float:
        dram_energy_sum = 0.0
        i = 0

        for name, module in self.base_model.named_modules():
            # FIXME: This part for now only looks at Conv2D not any Linear Layers
            if isinstance(module, qmodules.FaultyQConv2d):
                w_bits = module._weight_quantizer.num_bits
                i_bits = module._input_quantizer.num_bits

                if i > 0:
                    dram_energy_sum += self.dram_data[i-1]['o_energy'] * (i_bits / 16)

                dram_energy_sum += self.dram_data[i]['i_energy'] * (i_bits / 16)
                dram_energy_sum += self.dram_data[i]['w_energy'] * (w_bits / 16)

                # Last layer has always 16 bit
                if i == len(self.dram_data) - 1:
                    dram_energy_sum += self.dram_data[i]['o_energy'] * 1

                i += 1

        # Timeloop works with pJ as unit, for convenience we use uJ from here on
        return dram_energy_sum / 1_000_000

    def enable_quantization(self):
        [module.enable_quant() for module in self.explorable_modules]

    def disable_quantization(self):
        [module.disable_quant() for module in self.explorable_modules]

    def _build_energy_model(self, fn) -> None:
        dram_data_df = pd.read_csv(fn)
        self.dram_data = {}
        for i, row in dram_data_df.iterrows():
            # realign values to 16 bit
            scale_factor = 16 / row['bitwidth']

            self.dram_data[i] = {
                'w_energy': (row['w_reads'] + row['w_updates'] + row['w_fills']) * row['w_energy'] * scale_factor,
                'i_energy': (row['i_reads'] + row['i_updates'] + row['i_fills']) * row['i_energy'] * scale_factor,
                'o_energy': (row['o_reads'] + row['o_updates'] + row['o_fills']) * row['o_energy'] * scale_factor
            }

    def _create_quantized_model(self) -> None:
        quant_modules = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, torch_nn.Conv2d):
                # The file /pytorch_quantization/nn/modules/_utils.py:L161 has
                # some very strange kwargs check, therefore this simple solution
                # is not possible and we had to make it explicit :/
                # quant_conv = quant_nn.QuantConv2d(**module.__dict__, quant_desc...=...)
                bias_bool = module.bias is not None

                quant_conv = qmodules.FaultyQConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=bias_bool,
                    padding_mode=module.padding_mode,
                    quant_desc_input=self.quantization_descriptor,
                    quant_desc_weight=self.quantization_descriptor
                )

                # copy weights and biases
                quant_conv.weight = module.weight
                quant_conv.bias = module.bias

                quant_modules[name] = quant_conv
            elif isinstance(module, torch_nn.Linear):
                bias_bool = module.bias is not None

                quant_linear = qmodules.FaultyQLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=bias_bool
                )

                quant_linear.weight = module.weight
                quant_linear.bias = module.bias

                quant_modules[name] = quant_linear

        for name, quant_conv in quant_modules.items():
            setattr(self.base_model, name, quant_conv)

        for name, module in self.base_model.named_modules():
            if isinstance(module, qmodules.FaultyQConv2d) or isinstance(module, qmodules.FaultyQLinear):
                self.faulty_module_names.append(name)
                self.faulty_modules.append(module)
            if isinstance(module, quant_nn.TensorQuantizer):
                self.explorable_module_names.append(name)
                self.explorable_modules.append(module)

                if name.endswith('_input_quantizer'):
                    self.input_quantizers.append(module)
                elif name.endswith('_weight_quantizer'):
                    self.weight_quantizers.append(module)


    def generate_calibration_file(self, dataloader: DataLoader, progress=True,
                                  calib_method='histogram', **kwargs):
        assert calib_method in ['max', 'histogram'], "method has to be either max or histogram"
        assert 'method' in kwargs, "you have to specify a method for quantization calibration"

        self.base_model.to(self.device)

        # Enable Calibrators
        for module in self.input_quantizers:
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

        # Run the dataset ...
        for data, *_ in tqdm(dataloader,
                             desc="Calibrating",
                             disable=not progress,
                             ascii=True):
            # no need for actual accuracy function ...
            self.base_model(data.to(self.device))

        # Disable Calibrators
        for module in self.input_quantizers:
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

        # Collect amax statistics
        for module in self.input_quantizers:
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax(strict=False)
                else:
                    module.load_calib_amax(strict=False,**kwargs)
