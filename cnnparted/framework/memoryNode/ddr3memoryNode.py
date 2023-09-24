from .MemoryModelInterface import MemoryModelInterface
from .Ramulator import Ramulator
from .Vampire import Vampire
import csv

class DDR3Node(MemoryModelInterface):
    def __init__(self):
        self.ramulator= Ramulator()
        self.vampire = Vampire()
        

    def get_latency_ms_and_enrgy_mW(self, slice_size : int) -> float:
        trace = self.ramulator.run(slice_size,operation='R')
        stats_file_path = self.vampire.run(trace,"stats_file.csv")
        r_energy_pJ , r_cycles = self._get_energy_and_cyles_from_csv(stats_file_path)
        
        trace = self.ramulator.run(slice_size,operation='W')
        stats_file_path = self.vampire.run(trace,"stats_file.csv")
        w_energy_pJ , w_cycles = self._get_energy_and_cyles_from_csv(stats_file_path)
        print(r_energy_pJ,r_cycles,w_energy_pJ,w_cycles)

        return r_energy_pJ , r_cycles, w_energy_pJ , w_cycles
        

    def _get_energy_and_cyles_from_csv(self, csv_file_path):
        total_energy = None
        total_cycles = None

        # Open the CSV file for reading
        with open(csv_file_path, newline='') as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:

                if len(row) == 4:
                    if row[0] == 'total energy':
                        total_energy = row[1]  

                    elif row[0] == 'totalCycleCount':
                        total_cycles = row[1]  

        if total_energy is not None and total_cycles is not None:
            return total_energy,total_cycles
        else:
            raise Exception(f"Total Energy and/or Total Cycles not found in the CSV file: {csv_file_path}")
        
        
