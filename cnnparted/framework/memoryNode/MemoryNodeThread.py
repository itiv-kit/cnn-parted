from .MemoryModelInterface import MemoryModelInterface
from framework.ModuleThreadInterface import ModuleThreadInterface
from .ddr3memoryNode import DDR3Node

class MemoryNodeThread(ModuleThreadInterface):
 
    def _get_slice_size(self, size : int, word_bytes : int) -> int:
        slice_size = word_bytes
        for i in size:
            slice_size *= i
        return slice_size

    def _eval(self) -> None:
        if not self.config:
            return
        
        memory= MemoryModelInterface
        if self.config.get('ddr3'):
            memory = DDR3Node()
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
                r_energy_pJ , r_cycles, w_energy_pJ , w_cycles = memory.get_latency_ms_and_pow_mW(slice_size)
            except ValueError as e:
                print(e)
                r_energy_pJ =0
                r_cycles=0
                w_energy_pJ =0 
                w_cycles=0

            self.stats[layer.get('name')] = {
                'write_latency_ms' : w_cycles,
                'write_energy' : w_energy_pJ,
                'read_latency' : r_cycles,
                'read_energy' : r_energy_pJ
                }

