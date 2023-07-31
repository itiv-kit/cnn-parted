from .LinkModelInterface import LinkModelInterface
import math

class EthernetLink(LinkModelInterface):
  eth_modes = [
    "BASE100-T",
    "BASE1000-T",
    "10GBASE-T"
  ]

  def __init__(self, eth_mode : str ="BASE1000-T", cable_len_m : float = 5,
               enable_eee : bool = False, eee_lmi_ratio : float = 0.1,
               eee_toff_ms : int = 0) -> None:
    if eth_mode not in self.eth_modes:
      raise ValueError(f"{eth_mode} is an invalid Ethernet mode. Valid modes: {self.eth_modes}.")
    self.eth_mode = eth_mode

    self.cable_len_m = cable_len_m
    self.enable_eee = enable_eee
    self.eee_lmi_ratio = eee_lmi_ratio
    self.eee_toff_ms = eee_toff_ms

    # Ethernet format length definitiions
    self.n_packet_bytes = 7 + 1 # Number of bytes in the packet header (PREAMBLE + SFD)
    self.n_frame_bytes = 6 + 6 + 2 + 4 # DSTADDR + SRCADDR + LEN/TYP + FRAMECHK

    if self.eth_mode == "BASE100-T":
      self.base_consumption_mW = 200
      self.line_rate_bytes = 100e6 / 8
      if self.enable_eee:
        raise ValueError(f"Energy-Efficient Ethernet is not supported in {self.eth_mode}.")
    elif self.eth_mode == "BASE1000-T":
      self.base_consumption_mW = 600
      self.line_rate_bytes = 1e9 / 8
      self.Tw_us = 16.5
      self.Ts_us = 182
    elif self.eth_mode == "10GBASE-T":
      self.base_consumption_mW = 4000
      self.line_rate_bytes = 10e9 / 8
      self.Tw_us = 4.48
      self.Ts_us = 2.88

  def _compute_n_data_bytes(self, slice_size : int) -> int:
    n_data_bytes = max([slice_size, 46]) # Padded if < 46
    n_data_bytes = n_data_bytes + math.ceil(slice_size / 1500) * (self.n_packet_bytes + self.n_frame_bytes)
    return n_data_bytes

  def _compute_util_factor(self, slice_size : int, frame_rate : int) -> float:
    # Compute utilization factor: rho = lambda / mu
    # lambda - avg frame arival rate
    # mu - avg frame service rate
    n_data_bytes = self._compute_n_data_bytes(slice_size)

    mu = self.line_rate_bytes / (self.n_packet_bytes + self.n_frame_bytes + n_data_bytes)
    return float(frame_rate / mu)

  def _compute_power_mW(self, slice_size : int, frame_rate : int) -> float:
    if self.enable_eee == True:
      # Model source: https://doi.org/10.1109/TCOMM.2012.081512.120089
      util_factor = self._compute_util_factor(slice_size, frame_rate)
      n_data_bytes = self._compute_n_data_bytes(slice_size)

      if n_data_bytes * frame_rate > self.line_rate_bytes: # Check for maximum packet size, basic frames assumed
        raise ValueError(f"[EthernetLink] Not enough bandwidth: data rate {n_data_bytes * frame_rate}, line rate {self.line_rate_bytes}")

      # Toff is derived as follows:
      # toff = interarrival_period - Tw - Ts - Ton
      # toff = 1 / frame_rate - Tw - Ts - (self.n_packet_bytes + self.n_frame_bytes + n_data_bytes) / self.line_rate_bytes
      toff_s = 1 / frame_rate - ((self.Tw_us - self.Ts_us) / 1e6) - n_data_bytes / self.line_rate_bytes
      toff_us = toff_s * 1e6
      p_off = (1 - util_factor) * (toff_us / (toff_us + self.Tw_us + self.Ts_us))
      return float(self.base_consumption_mW * (1 - (1 - self.eee_lmi_ratio) * p_off))
    else:
      return float(self.base_consumption_mW)

  def _compute_latency_us(self, slice_size : int, frame_rate : int) -> float:
    n_data_bytes = self._compute_n_data_bytes(slice_size)

    if n_data_bytes * frame_rate > self.line_rate_bytes: # Check for maximum packet size, basic frames assumed
      raise ValueError(f"[EthernetLink] Not enough bandwidth: data rate {n_data_bytes * frame_rate}, line rate {self.line_rate_bytes}")

    r_prop_s = self.cable_len_m * 6e-9 # Cable propagation delay, 6 ns/m assumed

    if self.enable_eee == True:
      tx_latency_s = (self.Tw_us / 1e6) + (n_data_bytes / self.line_rate_bytes) + r_prop_s
      return float(tx_latency_s * 1e6)
    else:
      tx_latency_s = (n_data_bytes / self.line_rate_bytes) + r_prop_s
      return float(tx_latency_s * 1e6)

  def get_latency_ms(self, slice_size : int, frame_rate : int) -> float:
    return self._compute_latency_us(slice_size, frame_rate) / 1e3

  def get_pow_cons_mW(self, slice_size : int, frame_rate : int) -> float:
    return self._compute_power_mW(slice_size, frame_rate)


if __name__ == "__main__":
    # Test function
    slice_size = 802816
    frame_rate = 25
    ethLink = EthernetLink(eth_mode="BASE1000-T", cable_len_m=5, enable_eee=True,eee_lmi_ratio=0.1, eee_toff_ms=10)
    latency = ethLink.get_latency_ms(slice_size, frame_rate)
    power_consumption = ethLink.get_pow_cons_mW(slice_size, frame_rate)
    print("Ethernet Link characteristics for a slice size {} bytes".format(slice_size))
    print("Cable Length: ............... {} m".format(ethLink.cable_len_m))
    print("EEE Enabled: ................ {}".format(ethLink.enable_eee))
    print("LMI Ratio: .................. {}".format(ethLink.eee_lmi_ratio))
    print("Mean duration LPI mode: ..... {} ms".format(ethLink.eee_toff_ms))
    print("Used Mode: .................. {}".format(ethLink.eth_mode))
    print("========= Estimated characteristics ==========")
    print("Latency of the link: ........ {:.3f} ms".format(latency))
    print("Power consumption: .......... {:.6f} mW".format(power_consumption))
    print("Estimated datarate: ......... {:.3f} kBps".format(8 * slice_size / (latency * 1024)))
