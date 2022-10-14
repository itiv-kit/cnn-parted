class LinkModelInterface:
    def get_latency_ms(self, slice_size : int, frame_rate : int = 0) -> float:
        raise NotImplementedError

    def get_pow_cons_mW(self, slice_size : int, frame_rate : int = 0) -> float:
        raise NotImplementedError
