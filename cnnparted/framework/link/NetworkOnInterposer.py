from .LinkModelInterface import LinkModelInterface
import math

class NoILink(LinkModelInterface):
  NoI_modes = [
    'SerDes',
    'AIB',
    'BoW',
    'UCIe'
  ]

  def __init__(self, NoI_mode : str = 'UCIe', 
    number_of_wires : int = 32, data_rate_Gbps : float = 32,
    latency_ns : float = 2, power_per_bit_pj : float = 0.25) -> None:
    if NoI_mode not in self.NoI_modes:
      raise ValueError(f"{NoI_mode} is an invalid NoI mode. Valid modes: {self.NoI_modes}.")
    self.NoI_mode = NoI_mode
    self.number_of_wires = number_of_wires
    self.data_rate_Gbps = data_rate_Gbps
    self.bandwidth_GBps = self.number_of_wires * self.data_rate_Gbps / 8
    self.latency_ns = latency_ns
    self.power_per_bit_pj = power_per_bit_pj
   

  def _compute_power_mW(self, slice_size : int, frame_rate : int) -> float:
    n_data_bytes = slice_size 
    data_rate = n_data_bytes * frame_rate
    if data_rate > self.bandwidth_GBps:
      raise ValueError(f'[NoILink] Not enough bandwidth: data rate {data_rate}, bandwidth {self.bandwidth_GBps}')
    power_mW = self.power_per_bit_pj * 8 * n_data_bytes * frame_rate * 1e-9
    return float(power_mW)

  def _compute_latency_us(self, slice_size : int, frame_rate : int) -> float:
    n_data_bytes = slice_size 
    data_rate = n_data_bytes * frame_rate
    if data_rate > self.bandwidth_GBps:
      raise ValueError(f'[NoILink] Not enough bandwidth: data rate {data_rate}, bandwidth {self.bandwidth_GBps}')
    latency_us = n_data_bytes / self.bandwidth_GBps * 1e3 + self.latency_ns * 1e3
    return float(latency_us)
  


  def get_latency_ms(self, slice_size : int, frame_rate : int) -> float:
    return self._compute_latency_us(slice_size, frame_rate)

  def get_pow_cons_mW(self, slice_size : int, frame_rate : int) -> float:
    return self._compute_power_mW(slice_size, frame_rate)


if __name__ == "__main__":
    # Test function
    slice_size = 802816
    frame_rate = 25
    noi = NoILink()
    latency = noi.get_latency_ms(slice_size, frame_rate)
    power_consumption = noi.get_pow_cons_mW(slice_size, frame_rate)