import copy
from math import sqrt
import pathlib
import yaml
import shutil
import numpy as np

from framework.dse.architecture_config import ArchitectureConfig
from framework.dse.pymoo_interface import PymooInterface
from framework.dse.timeloop_interface import TimeloopInterface

class SimbaConfig(ArchitectureConfig, PymooInterface):
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

    def get_config(self) -> dict:
        cfg = {}
        cfg["num_pes"] = self.num_pes
        cfg["lmacs"] = self.lmacs
        cfg["globalbuf_size"] = self.globalbuf_size
        cfg["wbuf_size"] = self.wbuf_size
        cfg["accbuf_size"] = self.accbuf_size
        cfg["inbuf_size"] = self.inbuf_size
        return cfg

    @classmethod
    def from_genome(cls, genome: list):
        pe_dims = genome[0:1]
        pe_mems = genome[2:5]
        local_mems = genome[5]
        simba = cls(1, 1, 1, 1, 1, 1)
        simba.pe_array_dim = pe_dims
        simba.pe_array_mem = pe_mems
        simba.local_mems = local_mems
        return simba

    def to_genome(self) -> list:
        return self.pe_array_dim + self.pe_array_mem + self.local_mems

    @property
    def pe_array_dim(self):
        return [self.num_pes, self.lmacs]

    @pe_array_dim.setter
    def pe_array_dim(self, dims):
        self.num_pes = dims[0]
        self.lmacs = dims[1]
    
    @property
    def pe_array_mem(self):
        return [self.inbuf_depth, self.wbuf_depth, self.accbuf_depth]

    @pe_array_mem.setter
    def pe_array_mem(self, mem_depths):
        self.inbuf_depth = mem_depths[0]
        self.inbuf_size = int(mem_depths[0] * self.word_bits * self.block_size_input_buf // (8*1024))
        self.wbuf_depth = mem_depths[1]
        self.inbuf_size = int(mem_depths[1] * self.word_bits * self.block_size_weight_buf // (8*1024))
        self.accbuf_depth = mem_depths[2]
        self.inbuf_size = int(mem_depths[2] * self.word_bits * self.block_size_acc_buf // (8*1024))

    @property
    def local_mems(self):
        return [self.globalbuf_depth]

    @local_mems.setter
    def local_mems(self, mem_depths):
        self.globalbuf_depth = mem_depths
        self.globalbuf_size = int(self.globalbuf_depth * self.word_bits * self.block_size_global_buf // (8*1024))


class SimbaArchitectureMutator(TimeloopInterface):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.config: SimbaConfig = None
        search_space_constraints = cfg.get("constraints", {})
        
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
        self.num_bufs = 4
        # LMAC must be a multiple of num_bufs due to Timeloop requirements 
        self.lmac_nums = list(filter(lambda x: x % self.num_bufs == 0, self.lmac_nums))
        assert self.lmac_nums, f"No valid number of LMAC specified, must be multiple of num_bufs={self.num_bufs}"

        self.inbuf_sizes = search_space_constraints.get("inbuf_sizes", [64])
        self.wbuf_sizes = search_space_constraints.get("wbuf_sizes", [32])
        self.accbuf_sizes = search_space_constraints.get("accbuf_sizes", [0.375])
        self.globalbuf_sizes = search_space_constraints.get("gbuf_sizes", [128])

        # Generate valid configuration
        self.generate_design_space() 

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

    def generate_design_space(self):
        for pes in self.pe_nums:
            for lmacs in self.lmac_nums:
                for wbuf_size in self.wbuf_sizes:
                    for accbuf_size in self.accbuf_sizes:
                        for globalbuf_size in self.globalbuf_sizes:
                            for inbuf_size in self.inbuf_sizes:
                                self.design_space.append(copy.copy(SimbaConfig(pes, lmacs, wbuf_size, accbuf_size, globalbuf_size, inbuf_size))) 

    def write_tl_arch(self, config:SimbaConfig = None, outdir=None):
        if config is None:
            config = self.config
        if outdir is None:
            outdir = self.tl_out_configs_dir
        base_arch = pathlib.Path(self.tl_in_configs_dir, "archs", "simba_like.yaml")
        arch_out = pathlib.Path(outdir, "archs", "simba_like.yaml")
        with open(base_arch, "r") as f:
            arch = yaml.safe_load(f)

        #Modify the arch parameters
        simba = arch["architecture"]["subtree"][0]["subtree"][0]
        global_buffer = simba["local"][0]
        pe = simba["subtree"][0]
        pe_input_buffer = pe["local"][0]
        pe_wght_buffer = pe["local"][1]
        pe_accu_buffer = pe["local"][2]
        pe_wght_regs = pe["local"][3]
        lmac = pe["local"][4]

        # Sanity checks, should keys be reordered for any reason
        assert(global_buffer["name"] == "GlobalBuffer")
        assert("PE" in pe["name"])
        assert(pe_input_buffer["name"] == "PEInputBuffer")
        assert("PEWeightBuffer" in pe_wght_buffer["name"])
        assert("PEAccuBuffer" in pe_accu_buffer["name"])
        assert("PEWeightRegs" in pe_wght_regs["name"])
        assert("LMAC" in lmac["name"])

        global_buffer["attributes"]["memory_depth"] = config.globalbuf_depth

        pe["name"] = f"PE[0..{config.num_pes-1}]"

        pe_input_buffer["attributes"]["memory_depth"] = config.inbuf_depth
        pe_input_buffer["attributes"]["meshX"] = config.num_pes

        pe_wght_buffer["attributes"]["memory_depth"] = config.wbuf_depth
        pe_wght_buffer["attributes"]["meshX"] = config.num_pes

        pe_accu_buffer["attributes"]["memory_depth"] = config.accbuf_depth
        pe_accu_buffer["attributes"]["meshX"] = config.num_pes

        pe_wght_regs["name"] = f"PEWeightRegs[0..{config.lmacs-1}]"
        pe_wght_regs["attributes"]["meshX"] = config.num_pes
        pe_wght_regs["attributes"]["cluster-size"] = config.lmacs*4

        lmac["name"] = f"LMAC[0..{config.lmacs-1}]"
        lmac["attributes"]["meshX"] = config.num_pes

        with open(arch_out, "w") as f:
            y = yaml.safe_dump(arch, sort_keys=False)
            f.write(y)

    def write_tl_arch_constraints(self, config:SimbaConfig = None, outdir=None):
        if config is None:
            config = self.config
        if outdir is None:
            outdir = self.tl_out_configs_dir
        base_arch_constraints = pathlib.Path(self.tl_in_configs_dir, "constraints", "simba_like_arch_constraints.yaml")
        constraints_out = pathlib.Path(outdir, "constraints", "simba_like_arch_constraints.yaml")
        shutil.copy(base_arch_constraints, constraints_out)
    
    def write_tl_map_constraints(self, config:SimbaConfig = None, outdir=None):
        if config is None:
            config = self.config
        if outdir is None:
            outdir = self.tl_out_configs_dir
        base_map_constraints = pathlib.Path(self.tl_in_configs_dir, "constraints", "simba_like_map_constraints.yaml")
        constraints_out = pathlib.Path(outdir, "constraints", "simba_like_map_constraints.yaml")
        shutil.copy(base_map_constraints, constraints_out)

    def run(self):
        return super().run()

    def run_from_config(self, config: SimbaConfig, outdir=None):
        return super().run_from_config(config, outdir)


if __name__ == "__main__":
    ds = {"pe_nums": [8, 16, 32],
          "lmac_nums": [16, 32],
          "min_wbuf_size": 32,
          "max_wbuf_size": 64}







