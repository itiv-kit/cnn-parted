import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from pytorch_quantization.nn.modules.quant_conv import QuantConv2d
from pytorch_quantization import tensor_quant

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

    def _inject_faults(self, x, fault_rate):
        quant_x, scale = tensor_quant.tensor_quant(x, x.abs().max())
        quant_x = quant_x.int().view(-1)
        crpt = torch.zeros(int(quant_x.shape[0] * 8 * fault_rate)).uniform_(0,quant_x.shape[0]*8).int() # uniformly distribute errors over bits
        for i in crpt:
            quant_x[int(i/8)] = torch.bitwise_xor(quant_x[int(i/8)],torch.bitwise_left_shift(torch.tensor(1), i % 8))
        quant_x = quant_x.view(x.shape)
        x = quant_x / scale

        return x

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_input, quant_weight = self._quant(input)

        quant_input = self._inject_faults(quant_input, self.fault_rate[0])
        quant_weight = self._inject_faults(quant_weight, self.fault_rate[1])

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