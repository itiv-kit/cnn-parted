from math import inf
from framework.link.link_model_interface import LinkModelInterface

from framework.link.ethernet_link import EthernetLink
from framework.link.network_on_interposer import NoILink
from framework.link.ble_link import BLELink, BLE_PHY

class Link():
    def __init__(self, config):
        self.config = config

        if not self.config:
            return

        self.link = LinkModelInterface
        if self.config.get('ethernet'):
            conf = self.config['ethernet']
            self.link = EthernetLink(conf['eth_mode'], conf['cable_len_m'],
                                conf['enable_eee'], conf['eee_lmi_ratio'],
                                conf['eee_toff_ms'])
        elif self.config.get('noi'):
            conf = self.config['noi']
            self.link = NoILink(conf['noi_mode'], conf['width'], conf['data_rate_Gbps'],
                           conf['latency_ns'], conf['power_bit_pj'])
        elif self.config.get('ble'):
            conf = self.config['ble']
            self.link = BLELink(mtu=conf['mtu'], ppci=conf['ppci'], ci=conf['ci'],
                           dle=conf['dle'], phy=BLE_PHY[conf['phy']],
                           tx_pow_cons=conf['tx_pow_cons'], ber=conf['ber'])
        else:
            raise NotImplementedError

        self.num_bytes = int(self.config['data_bit_width'] / 8)
        if self.config['data_bit_width'] % 8 > 0:
            self.num_bytes += 1

        # Look-Up Table to increase performance
        self.lut = {}


    def eval(self, slice_size) -> None:
        slice_size *= self.num_bytes

        if slice_size in self.lut.keys():
            return self.lut[slice_size] + [slice_size]
        else:
            try:
                latency = self.link.get_latency_ms(slice_size, self.config['fps'])
                energy = self.link.get_pow_cons_mW(slice_size, self.config['fps']) * latency / 1e3
            except ValueError as e:
                print(e)
                latency = inf
                energy = inf

            self.lut[slice_size] = [latency, energy]

            return latency, energy, slice_size
