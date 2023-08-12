from framework.ModuleThreadInterface import ModuleThreadInterface
from framework.link.LinkModelInterface import LinkModelInterface
import numpy as np

from .EthernetLink import EthernetLink

class LinkThread(ModuleThreadInterface):
    def _get_slice_size(self, size : int, word_bytes : int) -> int:
        slice_size = word_bytes
        for i in size:
            slice_size *= i
        return slice_size

    def _eval(self) -> None:
        if not self.config:
            return

        link = LinkModelInterface
        if self.config.get('ethernet'):
            conf = self.config['ethernet']
            link = EthernetLink(conf['eth_mode'], conf['cable_len_m'],
                                conf['enable_eee'], conf['eee_lmi_ratio'],
                                conf['eee_toff_ms'])
        else:
            raise NotImplementedError

        num_bytes = int(self.config['data_bit_width'] / 8)
        if self.config['data_bit_width'] % 8 > 0:
            num_bytes += 1

        layer_list = self.dnn.partition_points

        # DNN layers
        for layer in layer_list:
            slice_size = self._get_slice_size(layer.get('output_size'), num_bytes)
            try:
                latency = link.get_latency_ms(slice_size, self.config['fps'])
                energy = link.get_pow_cons_mW(slice_size, self.config['fps']) * latency / 1e3
            except ValueError as e:
                print(e)
                latency = 0
                energy = 0

        self.stats[layer.get('name')] = {
                'latency' : latency,
                'energy' : energy
            }
