from .MemoryModelInterface import MemoryModelInterface
from framework.ModuleThreadInterface import ModuleThreadInterface
from .ddr3memoryNode import DDR3Node
from framework.helpers.ConfigHelper import ConfigHelper

class MemoryNodeThread(ModuleThreadInterface):
 
    def _get_slice_size(self, size : int, word_bytes : int) -> int:
        slice_size = word_bytes
        for i in size:
            slice_size *= i
        return slice_size

    def _eval(self) -> None:

        
        memory= MemoryModelInterface
        memory = DDR3Node()     
        num_bytes = self.dnn.num_bytes
        layer_list = self.dnn.partition_points

        computed = {}  
        for layer in layer_list:
            layer_name = layer.get('name')
            layer_out_size = layer.get('output_size')
            slice_size = self._get_slice_size(layer_out_size, num_bytes)

            if slice_size in computed:
                self.stats[layer_name] = computed[slice_size]
            else:
                

                try:
                    r_energy_pJ, r_cycles, w_energy_pJ, w_cycles = memory.get_latency_ms_and_enrgy_mW(slice_size / 8)
                except ValueError as e:
                    print(e)
                    r_energy_pJ = 0
                    r_cycles = 0
                    w_energy_pJ = 0
                    w_cycles = 0

                write_latency_ms = 2.5 * w_cycles / 1e6
                write_energy = w_energy_pJ / 1e9
                read_latency_ms = 2.5 * r_cycles / 1e6
                read_energy = r_energy_pJ / 1e9

                self.stats[layer_name] = {
                    'latency_ms': write_latency_ms + read_latency_ms,
                    'energy': write_energy + read_energy
                }

                computed[slice_size] = self.stats[layer_name]

