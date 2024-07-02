import copy
from math import sqrt
import pathlib
from typing import Dict
import yaml
import shutil

from framework.dse.ArchitectureMutator import ArchitectureMutator, ArchitectureConfig


class EyerissConfig(ArchitectureConfig):
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

        #TODO Some of these vales are redundant and ahould be cleaned up
        self.meshX = pe_dim_x #yes, this is correct
        self.pe_dim_y = pe_dim_y
        self.pe_dim_x = pe_dim_x
        self.num_pes = pe_dim_y * pe_dim_x
        self.pes_per_mesh = self.num_pes / self.meshX

    def get_config(self) -> Dict:
        cfg = {}
        cfg["pe_dim_y"] = self.pe_dim_y
        cfg["pe_dim_x"] = self.pe_dim_x
        cfg["glb_size"] = self.glb_size
        cfg["ifmap_spad_size"] = self.ifmap_spad_size
        cfg["weight_spad_size"] = self.weight_spad_size
        cfg["psum_spad_size"] = self.psum_spad_size
        return cfg


class EyerissArchitectureMutator(ArchitectureMutator):
    def __init__(self, cfg):
        super().__init__(cfg)
        search_space_constraints = cfg["constraints"]

        self.pe_dims_y = search_space_constraints.get("pe_dims_y", [12])
        self.pe_dims_x = search_space_constraints.get("pe_dims_x", [14])

        self.glb_sizes = search_space_constraints.get("gbuf_sizes", [128]) #kB
        self.ifmap_spad_sizes = search_space_constraints.get("ifmap_spad_sizes", [24]) #Byte
        self.weight_spad_sizes = search_space_constraints.get("weight_spad_sizes", [384]) #Byte
        self.psum_spad_sizes = search_space_constraints.get("psum_spad_sizes", [32]) #Byte

        # Generate valid configuration
        self._construct_design_space() 


    def mutate_arch(self):
        base_arch = pathlib.Path(self.tl_in_configs_dir, "archs", "eyeriss_like.yaml")
        arch_out = pathlib.Path(self.tl_out_configs_dir, "archs", "eyeriss_like.yaml")
        with open(base_arch, "r") as f:
            arch = yaml.safe_load(f)

        #Modify the arch parameters
        eyeriss = arch["architecture"]["subtree"][0]["subtree"][0]
        ifmap_spad = eyeriss["subtree"][0]["local"][0]
        wght_spad = eyeriss["subtree"][0]["local"][1]
        psum_spad = eyeriss["subtree"][0]["local"][2]
        glb = eyeriss["local"][0]
        dummy_buffer = eyeriss["local"][1]

        # Sanity checks, should yaml.dump reorder the keys
        assert(eyeriss["name"] == "eyeriss")
        assert(glb["name"] == "shared_glb")
        assert(ifmap_spad["name"] == "ifmap_spad")
        assert(wght_spad["name"] == "weights_spad")
        assert(psum_spad["name"] == "psum_spad")
        assert("DummyBuffer" in dummy_buffer["name"])

        glb["attributes"]["memory_depth"] = self.config.glb_depth
        dummy_buffer["name"] = f"DummyBuffer[0..{self.config.meshX-1}]"

        eyeriss["subtree"][0]["name"] = f"PE[0..{self.config.num_pes-1}]"
        for component in eyeriss["subtree"][0]["local"]:
            component["attributes"]["meshX"] = self.config.meshX

        ifmap_spad["attributes"]["memory_depth"] = self.config.ifmap_spad_depth
        wght_spad["attributes"]["memory_depth"] = self.config.weights_spad_depth
        psum_spad["attributes"]["memory_depth"] = self.config.psum_spad_depth

        with open(arch_out, "w") as f:
            y = yaml.safe_dump(arch, sort_keys=False)
            f.write(y)
    
    def mutate_arch_constraints(self):
        base_arch_constraints = pathlib.Path(self.tl_in_configs_dir, "constraints", "eyeriss_like_arch_constraints.yaml")
        constraints_out = pathlib.Path(self.tl_out_configs_dir, "constraints", "eyeriss_like_arch_constraints.yaml")
        shutil.copy(base_arch_constraints, constraints_out)
    
    def mutate_map_constraints(self):
        base_map_constraints = pathlib.Path(self.tl_in_configs_dir, "constraints", "eyeriss_like_map_constraints.yaml")
        constraints_out = pathlib.Path(self.tl_out_configs_dir, "constraints", "eyeriss_like_map_constraints.yaml")
        shutil.copy(base_map_constraints, constraints_out)

    def _construct_design_space(self):
        for pe_dim_y in self.pe_dims_y:
            for pe_dim_x in self.pe_dims_x:
                for glb_size in self.glb_sizes:
                    for ifmap_spad_size in self.ifmap_spad_sizes:
                        for weight_spad_size in self.weight_spad_sizes:
                            for psum_spad_size in self.psum_spad_sizes:
                                self.design_space.append(copy.copy(EyerissConfig(pe_dim_y, pe_dim_x, glb_size, ifmap_spad_size, weight_spad_size, psum_spad_size))) 


    def run(self):
        return super().run()
        

if __name__ == "__main__":
    ds = {"PE": [336],
        "meshX": [28]}

