class MemoryModelInterface:
    def get_latency_ms_and_energy_mW(self, slice_size : int) -> float:
        raise NotImplementedError

