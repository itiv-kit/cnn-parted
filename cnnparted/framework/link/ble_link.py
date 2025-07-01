# Created by Alexey Serdyuk

from framework.link.link_model_interface import LinkModelInterface
from typing import Tuple
from enum import Enum
import math
import numpy as np


class BLE_PHY(Enum):
    PHY_1MBPS = 1000000
    PHY_2MBPS = 2000000
    PHY_500KBPS_CODED = 500000 # @TODO Consider LL packet structure changes
    PHY_125KBPS_CODED = 125000 # -//-

class BLELink(LinkModelInterface):
    PRE_SZ = 1 # Preamble (for 1MBPS PHY)
    LL_HEADER_SZ = 7 # 4B Access address + 3B CRC
    DATA_CH_PDU_HEADER_SZ = 2 # Data Channel PDU header size
    L2CAP_HEADER_SZ = 4
    ATT_HEADER_SZ = 3

    LL_PAYLOAD_MAX_SZ = 27 # For DLE = 0
    LL_PACKET_MAX_SZ = PRE_SZ + LL_HEADER_SZ + DATA_CH_PDU_HEADER_SZ + LL_PAYLOAD_MAX_SZ
    LL_COMB_HEADER_SZ = PRE_SZ + LL_HEADER_SZ + DATA_CH_PDU_HEADER_SZ
    LL_MASTER_POLL_SZ = LL_COMB_HEADER_SZ

    T_IFS = 150.0e-6 # Inter frame spacing (time gap between packet and ACK)

    def __init__(self,
        mtu: int = 247,
        ci: float = 30.0e-3,
        ppci: int = 6,
        dle: bool = False,
        phy: BLE_PHY = BLE_PHY.PHY_1MBPS,
        tx_pow_cons: float = 15e-3,
        ber: float = 0):
        """
        Args:
            mtu - maximum transmission unit of GATT layer [23 ... 512]
            ci - connection interval (CI) [7.5 ... 4000] 1.25 ms step
            ppci - packets per connection interval (CI) [1 ... 16], not limited by core spec, but rather by vendor
            dle - data length extension flag
            phy - BLE PHY: 1Mbps, 2Mbps or Coded 500 kBps or 125 KBps
            tx_pow_cons - power consumption during transmission, mW @ 3V VDD
            ber - Bit error rate
        """
        self.mtu = mtu
        self.ci = ci
        self.ppci = ppci
        self.dle = dle # @TODO Consider data length extension for different PHYS
        self.phy = phy
        self.ber = ber # @TODO Consider bit errors
        self.tx_pow_cons = tx_pow_cons

        self.ci_data_capacity = min(mtu, ppci * self.LL_PAYLOAD_MAX_SZ)
        self.round_trip_duration = (self.LL_PAYLOAD_MAX_SZ + self.LL_MASTER_POLL_SZ) / self.phy.value + self.T_IFS

    def __calc_latency_and_power_cons(self, slice_size: int) -> Tuple[float, float]:
        # Calculate how many GATT notifications required to transmit the slice
        ntf_payload_sz = self.mtu - self.ATT_HEADER_SZ - self.L2CAP_HEADER_SZ
        ntfs_num = math.ceil(slice_size / ntf_payload_sz)
        last_ntf_sz = slice_size % ntf_payload_sz

        # Check, how many connection intervals needed for a single notification
        if ntfs_num > 1:
            ci_per_ntf = math.ceil(self.mtu / self.ci_data_capacity)
            last_ci_of_ntf_payload = self.mtu % self.ci_data_capacity
        else:
            ci_per_ntf = math.ceil((last_ntf_sz + self.ATT_HEADER_SZ) / self.ci_data_capacity)
            last_ci_of_ntf_payload = (last_ntf_sz + self.ATT_HEADER_SZ) % self.ci_data_capacity

        # Calculate how many LL packets per notification needed
        ll_packets_num_per_ci = math.ceil(self.mtu / self.LL_PAYLOAD_MAX_SZ)
        # Calculate active transmission duration
        tx_bytes = slice_size + ll_packets_num_per_ci * ci_per_ntf * ntfs_num * self.LL_COMB_HEADER_SZ
        tx_bytes += (self.L2CAP_HEADER_SZ + self.ATT_HEADER_SZ) * ntfs_num
        tx_active_duration = float(tx_bytes) / self.phy.value

        latency = ntfs_num * ci_per_ntf * self.ci
        pow_cons = self._zero_division(self.tx_pow_cons * tx_active_duration, latency)
        return (latency, pow_cons)

    def get_latency_ms(self, slice_size: int, frame_rate : int = 0) -> float:
        latency, _ = self.__calc_latency_and_power_cons(slice_size)
        return latency * 1e3

    def get_pow_cons_mW(self, slice_size: int, frame_rate : int = 0) -> float:
        _, pow_cons = self.__calc_latency_and_power_cons(slice_size)
        return pow_cons * 1e3

    def _zero_division(self, a : float, b : float) -> float:
        return a / b if b else 0


if __name__ == "__main__":
    # Test function
    slice_size = 178
    bleLink = BLELink(mtu=185, ppci=7, ci=15e-3)
    latency = bleLink.get_latency_ms(slice_size)
    power_consumption = bleLink.get_pow_cons_mW(slice_size)
    print("BLE Link characteristics for a slice size {} bytes".format(slice_size))
    print("MTU: ........................ {} bytes".format(bleLink.mtu))
    print("Connection interval (CI): ... {} ms".format(bleLink.ci * 1000.0))
    print("LL Packets per CI: .......... {}".format(bleLink.ppci))
    print("Used PHY: ................... {}".format(bleLink.phy.name))
    print("========= Estimated characteristics ==========")
    print("Latency of the link: ........ {:.3f} ms".format(latency))
    print("Power consumption: .......... {:.6f} mW, @3V".format(power_consumption))
    print("Estimated datarate: ......... {:.3f} kBps".format(8 * slice_size / (latency * 1024)))
