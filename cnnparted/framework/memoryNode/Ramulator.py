import subprocess
import random
import struct
import os
from framework.constants import ROOT_DIR
import subprocess

class Ramulator:
    def __init__(self):
        self.ramulator_path = os.path.join(ROOT_DIR,'tools','ramulator')
        self.config_file_name = "ddr3.cfg"
        self.trace_file_name = "read.trace"
        self.cmd ="./ramulator"
        self.vampire_trace ="cmd-trace-chan-0-rank-0.cmdtrace"
        self.vampire_input_trace="vampire-trace.cmdtrace"

    def run(self,size,operation):

        self._generate_memory_trace_file(size, operation)
        conf_path =  os.path.join(ROOT_DIR,'configs','mem-configs', self.config_file_name)
        trace_path = os.path.join(self.ramulator_path, self.trace_file_name)

        command = [
            self.cmd,  
            conf_path, 
            "--mode=dram", 
            trace_path
        ]

        try:           
            subprocess.run(command,stdout=subprocess.PIPE, cwd=self.ramulator_path,check=True)
            return self._modify_trace_file()
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")

        

   
    def _modify_trace_file(self):
        
        commands_to_modify = {'ACT', 'RD', 'WR'}
        commands_to_keep = {'RD', 'WR', 'PRE','ACT'}# vampire supports solely these commands
        
        in_path = os.path.join(self.ramulator_path, self.vampire_trace)
        out_path =os.path.join(self.ramulator_path,  self.vampire_input_trace)
        
        with open(in_path, 'r') as infile, open(out_path, 'w') as outfile:
            for line in infile:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    timestamp, command, value = parts
                    if command in commands_to_keep:
                        if command in commands_to_modify:
                            outfile.write(f"{timestamp},{command},{value},0\n")
                        else:
                            outfile.write(f"{timestamp},{command},{value}\n")                 
        
        return out_path
        

    def _generate_memory_trace_file(self,size, operation):
        
        #size : how many 64 bits should be written or read

        if operation not in ['R', 'W']:
            raise ValueError("Invalid operation. Choose 'R' or 'W'.")
        
        file_path=os.path.join(self.ramulator_path, self.trace_file_name)

        with open(file_path, "w") as file:
            for _ in range(int(size)):
                random_number = random.randint(0x00000000, 0xFFFFFFFF)
                modulo_result = random_number % 0x80
                power_of_0x80 = random_number - modulo_result
                address = f"{power_of_0x80:08X}"  # Format as 8-digit hexadecimal

                file.write(f"0x{address} {operation}\n")
