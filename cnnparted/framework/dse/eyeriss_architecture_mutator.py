import copy
from math import sqrt
import pathlib
import yaml
import shutil
import numpy as np

from framework.dse.interfaces.architecture_config import ArchitectureConfig
from framework.dse.interfaces.genome_interface import GenomeInterface
from framework.dse.interfaces.timeloop_interface import TimeloopInterface
from framework.dse.interfaces.exhaustive_search import ExhaustiveSearch

class EyerissConfig(ArchitectureConfig, GenomeInterface):
    def __init__(self, pe_dim_y, pe_dim_x, glb_size, ifmap_spad_size, weight_spad_size, psum_spad_size):
        self.word_bits = 16
        self.glb_block_size = 4
        self.spad_block_size = 1

        self.glb_size = glb_size
        self.glb_depth = int(glb_size*8*1024 // (self.word_bits*self.glb_block_size))

        self.ifmap_spad_size = ifmap_spad_size
        self.ifmap_spad_depth = int(ifmap_spad_size*8 // (self.word_bits*self.spad_block_size))
        self.weight_spad_size = weight_spad_size
        self.weight_spad_depth = int(weight_spad_size*8 // (self.word_bits*self.spad_block_size))
        self.psum_spad_size = psum_spad_size
        self.psum_spad_depth = int(psum_spad_size*8 // (self.word_bits*self.spad_block_size))

        self.pe_dim_y = pe_dim_y
        self.pe_dim_x = pe_dim_x
        self.num_pes = pe_dim_y * pe_dim_x

    def get_config(self) -> dict:
        cfg = {}
        cfg["pe_dim_y"] = self.pe_dim_y
        cfg["pe_dim_x"] = self.pe_dim_x
        cfg["glb_size"] = self.glb_size
        cfg["ifmap_spad_size"] = self.ifmap_spad_size
        cfg["weight_spad_size"] = self.weight_spad_size
        cfg["psum_spad_size"] = self.psum_spad_size
        return cfg

    @classmethod
    def from_genome(cls, genome: list):
        pe_dims = genome[0:1]
        pe_mems = genome[2:5]
        local_mems = genome[5:6]
        eyeriss = cls(2, 2, 1024, 128, 128, 128)
        eyeriss.pe_array_dim = pe_dims
        eyeriss.pe_array_mem = pe_mems
        eyeriss.local_mems = local_mems
        return eyeriss

    def to_genome(self) -> list:
        return self.pe_array_dim + self.pe_array_mem + self.local_mems

    @property
    def pe_array_dim(self):
        return [self.pe_dim_y, self.pe_dim_x]

    @pe_array_dim.setter
    def pe_array_dim(self, dims):
        self.pe_dim_y = dims[0]
        self.pe_dim_x = dims[1]
        self.num_pes = dims[0] * dims[1]
    
    @property
    def pe_array_mem(self):
        return [self.ifmap_spad_depth, self.weight_spad_depth, self.psum_spad_depth]

    @pe_array_mem.setter
    def pe_array_mem(self, mem_depths):
        self.ifmap_spad_depths = mem_depths[0]
        self.ifmap_spad_size = int(mem_depths[0] * self.word_bits * self.spad_block_size // 8)
        self.weight_spad_depth = mem_depths[1]
        self.weight_spad_size = int(mem_depths[1] * self.word_bits * self.spad_block_size // 8)
        self.psum_spad_depth = mem_depths[2]
        self.psum_spad_size = int(mem_depths[2] * self.word_bits * self.spad_block_size // 8)

    @property
    def local_mems(self):
        return [self.glb_depth]

    @local_mems.setter
    def local_mems(self, glb_depth):
        self.glb_depth = glb_depth
        self.glb_size = int(glb_depth * self.word_bits * self.glb_block_size // (8*1024))


class EyerissArchitectureAdaptor(TimeloopInterface, ExhaustiveSearch):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config: EyerissConfig = None

    def read_space_cfg(self, cfg):
        search_space_constraints = cfg.get("constraints", {})

        self.pe_dims_y = search_space_constraints.get("pe_dims_y", [12])
        self.pe_dims_x = search_space_constraints.get("pe_dims_x", [14])

        self.ifmap_spad_sizes = search_space_constraints.get("ifmap_spad_sizes", [24]) #Byte
        self.weight_spad_sizes = search_space_constraints.get("weight_spad_sizes", [384]) #Byte
        self.psum_spad_sizes = search_space_constraints.get("psum_spad_sizes", [32]) #Byte

        self.glb_sizes = search_space_constraints.get("gbuf_sizes", [128]) #kB

        # Generate valid configuration
        self.generate_design_space() 


    def _generate_mem_sizes(self, lower: int, upper: int, step: int):
        mem_sizes = list(range(lower, upper, step))
        return mem_sizes

    def write_tl_arch(self, config: EyerissConfig = None, outdir=None):
        if config is None:
            config = self.config
        if outdir is None:
            outdir = self.tl_out_configs_dir
        base_arch = pathlib.Path(self.tl_in_configs_dir, "archs", "eyeriss_like.yaml")
        arch_out = pathlib.Path(outdir, "archs", "eyeriss_like.yaml")
        with open(base_arch, "r") as f:
            arch = yaml.safe_load(f)

        #Modify the arch parameters
        eyeriss = arch["architecture"]["subtree"][0]["subtree"][0]
        glb = eyeriss["local"][0]
        dummy_buffer = eyeriss["local"][1]
        pe = eyeriss["subtree"][0]
        ifmap_spad = eyeriss["subtree"][0]["local"][0]
        wght_spad = eyeriss["subtree"][0]["local"][1]
        psum_spad = eyeriss["subtree"][0]["local"][2]
        mac = eyeriss["subtree"][0]["local"][3]

        # Sanity checks, should keys be reordered for any reason
        assert(eyeriss["name"] == "eyeriss")
        assert(glb["name"] == "shared_glb")
        assert(ifmap_spad["name"] == "ifmap_spad")
        assert(wght_spad["name"] == "weights_spad")
        assert(psum_spad["name"] == "psum_spad")
        assert("DummyBuffer" in dummy_buffer["name"])
        assert(mac["name"] == "mac")

        pe["name"] = f"PE[0..{config.num_pes-1}]"
        glb["attributes"]["memory_depth"] = config.glb_depth

        dummy_buffer["name"] = f"DummyBuffer[0..{config.pe_dim_x-1}]"
        dummy_buffer["attributes"]["meshX"] = config.pe_dim_x

        ifmap_spad["attributes"]["memory_depth"] = config.ifmap_spad_depth
        ifmap_spad["attributes"]["meshX"] = config.pe_dim_x

        wght_spad["attributes"]["memory_depth"] = config.weight_spad_depth
        wght_spad["attributes"]["meshX"] = config.pe_dim_x

        psum_spad["attributes"]["memory_depth"] = config.psum_spad_depth
        psum_spad["attributes"]["meshX"] = config.pe_dim_x

        mac["attributes"]["meshX"] = config.pe_dim_x

        with open(arch_out, "w") as f:
            y = yaml.safe_dump(arch, sort_keys=False)
            f.write(y)
    
    def write_tl_arch_constraints(self, config:EyerissConfig = None, outdir=None):
        if config is None:
            config = self.config
        if outdir is None:
            outdir = self.tl_out_configs_dir
        base_arch_constraints = pathlib.Path(self.tl_in_configs_dir, "constraints", "eyeriss_like_arch_constraints.yaml")
        constraints_out = pathlib.Path(outdir, "constraints", "eyeriss_like_arch_constraints.yaml")
        shutil.copy(base_arch_constraints, constraints_out)
    
    def write_tl_map_constraints(self, config: EyerissConfig=None, outdir=None):
        if config is None:
            config = self.config
        if outdir is None:
            outdir = self.tl_out_configs_dir
        base_map_constraints = pathlib.Path(self.tl_in_configs_dir, "constraints", "eyeriss_like_map_constraints.yaml")
        constraints_out = pathlib.Path(outdir, "constraints", "eyeriss_like_map_constraints.yaml")
        shutil.copy(base_map_constraints, constraints_out)

    def generate_design_space(self):
        for pe_dim_y in self.pe_dims_y:
            for pe_dim_x in self.pe_dims_x:
                for glb_size in self.glb_sizes:
                    for ifmap_spad_size in self.ifmap_spad_sizes:
                        for weight_spad_size in self.weight_spad_sizes:
                            for psum_spad_size in self.psum_spad_sizes:
                                self.design_space.append(copy.copy(EyerissConfig(pe_dim_y, pe_dim_x, glb_size, ifmap_spad_size, weight_spad_size, psum_spad_size))) 


    def run(self):
        return super().run()
        
    def run_from_config(self, config: EyerissConfig, outdir=None):
        return super().run_from_config(config, outdir)

if __name__ == "__main__":
    ds = {"PE": [336],
        "meshX": [28]}

    eyeriss = EyerissConfig(12, 14, 1024, 128, 128, 128)
    breakpoint()

