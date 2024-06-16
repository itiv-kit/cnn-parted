import copy
from math import sqrt
import pathlib
import yaml
from typing import Dict
import shutil

from framework.dse.ArchitectureMutator import ArchitectureMutator, ArchitectureConfig

class SimbaConfig(ArchitectureConfig):
    def __init__(self, num_pes, lmacs, wbuf_size, accbuf_size, globalbuf_size, inbuf_size):
        #Constants
        self.word_bits = 8
        self.word_bits_acc = 24

        self.block_size_input_buf = 8
        self.block_size_acc_buf = 1
        self.block_size_weight_buf = 8
        self.block_size_global_buf = 32

        self.weight_bufs = 4
        self.acc_bufs = 4

        self.num_pes = num_pes
        self.lmacs = lmacs

        self.wbuf_size = wbuf_size
        self.wbuf_depth = int(wbuf_size*8*1024// (self.word_bits*self.block_size_weight_buf))
        self.accbuf_size = accbuf_size
        self.accbuf_depth = int(accbuf_size*8*1024// (self.word_bits_acc*self.block_size_acc_buf))
        self.globalbuf_size = globalbuf_size
        self.globalbuf_depth = int(globalbuf_size*8*1024 // (self.word_bits*self.block_size_global_buf))
        self.inbuf_size = inbuf_size
        self.inbuf_depth = int(inbuf_size*8*1024 // (self.word_bits*self.block_size_input_buf))



class SimbaArchitectureMutator(ArchitectureMutator):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        search_space_constraints = cfg["constraints"]
        
        #Constants related to memory width
        self.word_bits = 8
        self.word_bits_acc = 24
        self.block_size_input_buf = 8
        self.block_size_acc_buf = 1
        self.block_size_weight_buf = 8
        self.block_size_global_buf = 32
        self.weight_bufs = 4
        self.acc_bufs = 4

        #Compute elements
        self.pe_nums = search_space_constraints.get("num_pes", [16])
        self.lmac_nums = search_space_constraints.get("num_lmacs", [16])

        #weight and accumulator buffers are in equal numbers
        self.buf_nums = 4

        self.min_wbuf_size =   search_space_constraints.get("min_wbuf_size", 32)
        self.max_wbuf_size =   search_space_constraints.get("max_wbuf_size", 64)
        self.min_accbuf_size = search_space_constraints.get("min_accbuf_size", 0.375)
        self.max_accbuf_size = search_space_constraints.get("max_accbuf_size", 3)
        self.min_gbuf_size =   search_space_constraints.get("min_gbuf_size", 64)
        self.max_gbuf_size =   search_space_constraints.get("max_gbuf_size", 128)
        self.min_inbuf_size =  search_space_constraints.get("min_inbuf_size", 64)
        self.max_inbuf_size =  search_space_constraints.get("max_inbuf_size", 128)

        self.wbuf_sizes = self._calc_mem_sizes(self.min_wbuf_size, self.max_wbuf_size, self.block_size_weight_buf*self.word_bits)
        self.accbuf_sizes = self._calc_mem_sizes(self.min_accbuf_size, self.max_accbuf_size, self.block_size_acc_buf*self.word_bits_acc)
        self.globalbuf_sizes = self._calc_mem_sizes(self.min_gbuf_size, self.max_gbuf_size, self.block_size_global_buf*self.word_bits)
        self.inbuf_sizes = self._calc_mem_sizes(self.min_inbuf_size, self.max_inbuf_size, self.block_size_input_buf*self.word_bits)

        # Generate valid configuration
        self._construct_design_space() 

    def _calc_mem_sizes(self, min_sz, max_sz, mem_width):
        n=1
        mem_sizes = []
        while True:
            rows = n**2
            size = rows*mem_width / (8*1024) #kB
            if size < min_sz:
                n += 1
                continue
            elif min_sz <= size <= max_sz:
                mem_sizes.append(size)
                n += 1
            elif size > max_sz:
                break
        return mem_sizes

    def _construct_design_space(self):
        for pes in self.pe_nums:
            for lmacs in self.lmac_nums:
                for wbuf_size in self.wbuf_sizes:
                    for accbuf_size in self.accbuf_sizes:
                        for globalbuf_size in self.globalbuf_sizes:
                            for inbuf_size in self.inbuf_sizes:
                                self.design_space.append(copy.copy(SimbaConfig(pes, lmacs, wbuf_size, accbuf_size, globalbuf_size, inbuf_size))) 

    def mutate_arch(self):
        base_arch = pathlib.Path(self.tl_in_configs_dir, "archs", "simba_like.yaml")
        arch_out = pathlib.Path(self.tl_out_configs_dir, "archs", "simba_like.yaml")
        with open(base_arch, "r") as f:
            arch = yaml.safe_load(f)

        #Modify the arch parameters
        arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["name"] = "PE[0..{}]".format(self.config.num_pes-1)
        arch["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"]["memory_depth"] = self.config.globalbuf_depth

        pe_template = arch["architecture"]["subtree"][0]["subtree"][0]["subtree"][0]["local"]
        for component in pe_template:
            component["attributes"]["meshX"] = self.config.num_pes
            if "PEWeightBuffer" in component["name"]:
                #component["name"] = "PEWeightBuffer[0..{}]".format(self.config.weight_bufs-1)
                component["attributes"]["memory_depth"] = self.config.wbuf_depth
            elif "PEAccuBuffer" in component["name"]:
                #component["name"] = "PEAccuBuffer[0..{}]".format(self.config.acc_bufs-1)
                component["attributes"]["memory_depth"] = self.config.accbuf_depth
            elif "PEWeightRegs" in component["name"]:
                pass
            elif "MAC" in component["name"]:
                component["name"] = "MAC[0..{}]".format(self.config.lmacs-1)
            elif "PEInputBuffer" in component["name"]:
                component["attributes"]["memory_depth"] = self.config.inbuf_depth

        with open(arch_out, "w") as f:
            y = yaml.safe_dump(arch, sort_keys=False)
            f.write(y)

    def mutate_arch_constraints(self):
        base_arch_constraints = pathlib.Path(self.tl_in_configs_dir, "constraints", "eyeriss_like_map_constraints.yaml")
        constraints_out = pathlib.Path(self.tl_out_configs_dir, "constraints", "eyeriss_like_map_constraints.yaml")
        shutil.copy(base_arch_constraints, constraints_out)
    
    def mutate_map_constraints(self):
        base_map_constraints = pathlib.Path(self.tl_in_configs_dir, "constraints", "eyeriss_like_map_constraints.yaml")
        constraints_out = pathlib.Path(self.tl_out_configs_dir, "constraints", "eyeriss_like_map_constraints.yaml")
        shutil.copy(base_map_constraints, constraints_out)

    def run(self):
        return super().run()



if __name__ == "__main__":
    ds = {"pe_nums": [8, 16, 32],
          "lmac_nums": [16, 32],
          "min_wbuf_size": 32,
          "max_wbuf_size": 64}







