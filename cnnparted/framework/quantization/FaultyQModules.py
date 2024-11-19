import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from pytorch_quantization.nn.modules.quant_conv import QuantConv2d
from pytorch_quantization.nn.modules.quant_linear import QuantLinear
from pytorch_quantization import tensor_quant

def inject_faults(x, num_bits, fault_rate, faulty_bits):
    quant_x, scale = tensor_quant.tensor_quant(x, x.abs().max())
    quant_x = quant_x.int().view(-1)

    crpt_weights = torch.zeros(int(quant_x.shape[0] * faulty_bits * fault_rate)).uniform_(0,quant_x.shape[0]).int()*num_bits # randomly select tampered weights
    crpt_weights_bits = torch.zeros(crpt_weights.shape[0]).uniform_(0,faulty_bits).int()                                     # randomly choose corrupted bit per tampered weight
    crpt = crpt_weights + crpt_weights_bits
    for i in crpt:
        quant_x[int(i/num_bits)] = torch.bitwise_xor(quant_x[int(i/num_bits)],torch.bitwise_left_shift(torch.tensor(1), i % num_bits))
    quant_x = quant_x.view(x.shape)
    x = quant_x / scale

    return x

class FaultyQConv2d(QuantConv2d):
    """Quantized 2D conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 **kwargs):

        super(FaultyQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)

        self.fault_rate = [0, 0]
        self.faulty_bits = 2

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_input, quant_weight = self._quant(input)

        quant_input = inject_faults(quant_input, self.input_quantizer.num_bits, self.fault_rate[0], self.faulty_bits)
        quant_weight = inject_faults(quant_weight, self.weight_quantizer.num_bits, self.fault_rate[1], self.faulty_bits)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(quant_input, expanded_padding, mode='circular'),
                              quant_weight, self.bias, self.stride,
                              _pair(0), self.dilation, self.groups)
        else:
            output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                              self.groups)

        return output


class FaultyQLinear(QuantLinear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias, **kwargs)

        self.fault_rate = [0, 0]
        self.faulty_bits = 2

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        quant_input = inject_faults(quant_input, self.input_quantizer.num_bits, self.fault_rate[0], self.faulty_bits)
        quant_weight = inject_faults(quant_weight, self.weight_quantizer.num_bits, self.fault_rate[1], self.faulty_bits)

        output = F.linear(quant_input, quant_weight, bias=self.bias)

        return output