import copy
from math import log2
import pathlib
import yaml

from framework.dse.gemmini_architecture_mutator import GemminiArchitectureAdaptor, GemminiConfig
from framework.dse.timeloop_interface import TimeloopInterface
from framework.dse.architecture_config import ArchitectureConfig

class GemminiFreeConfig(ArchitectureConfig):
    def _is_pow_2(self, n):
        return (n & (n-1) == 0) and n != 0


    def __init__(self, mesh_dim_x, mesh_dim_y, spad_size, acc_size):
        self.mesh_dim_x = mesh_dim_x
        self.mesh_dim_y = mesh_dim_y
        self.tile_dim = 1 #const for now
        self.dim = self.mesh_dim_x * self.tile_dim

        self.data_w = 8 #Width of input data in bit
        self.output_w = 20 #Width of MAC result
        self.acc_w = 32 # Width of accumulator data type in bit

        self.spad_size = spad_size
        self.spad_banks = 4
        self.spad_width = self.dim * self.data_w
        self.spad_bank_rows = (self.spad_size*1024*8)//(self.spad_banks*self.spad_width)
        self.spad_rows = self.spad_bank_rows * self.spad_banks

        self.acc_size = acc_size
        self.acc_banks = 2
        self.acc_width = self.dim * self.acc_w

        self.acc_bank_rows = (self.acc_size*1024*8)//(self.acc_banks*self.acc_width)
        self.acc_rows = self.acc_bank_rows * self.acc_banks

    def __str__(self):
        return f"GemminiConfig(dim={self.dim}, spad={self.spad_size}kB, acc={self.acc_size}kB)"

    def get_config(self):
        cfg = {}
        cfg["mesh_dim_x"] = self.mesh_dim_x
        cfg["mesh_dim_y"] = self.mesh_dim_y
        #cfg["tile_dim"] = self.tile_dim
        cfg["acc_size"] = self.acc_size
        cfg["spad_size"] = self.spad_size
        return cfg

# This is a version of the Gemmini mutator that is not restricted by Gemminis architecture
# constraints to enable a more free design space exploration
class GemminiFreeArchitectureMutator(TimeloopInterface):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.config: GemminiFreeConfig = None
        search_space_constraints = cfg.get("constraints", {})

        self.tile_dim = 1
        self.mesh_dims_x = search_space_constraints.get("mesh_dims_x", [16])
        self.mesh_dims_y = search_space_constraints.get("mesh_dims_y", [16])

        #Boundaries of scratchpad sizes
        self.min_spad_size = search_space_constraints.get("min_spad_size", 256)
        self.max_spad_size = search_space_constraints.get("max_spad_size", 256)

        #Boundaries of accumulator sizes 
        self.min_acc_size = search_space_constraints.get("min_acc_size", 64)
        self.max_acc_size = search_space_constraints.get("max_acc_size", 64)

        self.spad_banks = 4
        self.acc_banks = 2

        # Element width for different memories in bit
        self.spad_data_width = 8
        self.acc_data_width = 32

    def _calc_valid_mem_sizes(self, mem_banks, mem_width, min_mem_size, max_mem_size, rows_max=None, enable_checks=False):
        mem_sizes = []

        n = 4
        while True:
            #spad_bank_rows = 2**n # bank rows must be power of two
            bank_rows = 2**n
            mem_size_kilobytes = (bank_rows * mem_banks * mem_width)/(1024*8) 

            if (mem_size_kilobytes < min_mem_size):
                n += 1
                continue
            elif (mem_size_kilobytes <= max_mem_size):
                n += 1
                mem_sizes.append(mem_size_kilobytes)
            else:
                break

        return mem_sizes


    def generate_design_space(self):
        for mesh_dim_x in self.mesh_dims_x:
            for mesh_dim_y in self.mesh_dims_y:
                dim = mesh_dim_x * self.tile_dim
                spad_width = dim * self.spad_data_width
                acc_width = dim * self.acc_data_width

                spad_sizes = self._calc_valid_mem_sizes(self.spad_banks, spad_width, self.min_spad_size, self.max_spad_size)
                acc_sizes = self._calc_valid_mem_sizes(self.acc_banks, acc_width, self.min_acc_size, self.max_acc_size)

                for spad_size in spad_sizes:
                    for acc_size in acc_sizes:
                        self.design_space.append(copy.copy(GemminiFreeConfig(mesh_dim_x, mesh_dim_y, spad_size, acc_size)))

    def mutate_arch_constraints(self, config=None, outdir=None):
        pass

    def mutate_arch(self, config=None, outdir=None):
        if config is None:
            config = self.config
        if outdir is None:
            outdir = self.tl_out_configs_dir
        base_arch = pathlib.Path(self.tl_in_configs_dir, "archs", "gemmini_like.yaml")
        arch_out = pathlib.Path(outdir, "archs", "gemmini_like.yaml")
        with open(base_arch, "r") as f:
            arch = yaml.safe_load(f)

        chip = arch["architecture"]["subtree"][0]["subtree"][0]
        scratchpad = chip["local"][0]
        pe_cols = chip["subtree"][0]
        accumulator = pe_cols["local"][0]
        pe_rows = pe_cols["subtree"][0]
        registers = pe_rows["local"][0]
        macc = pe_rows["local"][1]

        scratchpad["attributes"]["depth"] = int(config.spad_rows)
        scratchpad["attributes"]["width"] = config.mesh_dim_x * config.data_w
        scratchpad["attributes"]["entries"] = int(config.spad_rows * config.mesh_dim_x)
        scratchpad["attributes"]["n_banks"] = config.spad_banks
        scratchpad["attributes"]["block_size"] = config.mesh_dim_x
        #scratchpad["attributes"]["word-bits"] = config.data_w

        pe_cols["name"] = f"PECols[0..{config.mesh_dim_x-1}]"

        accumulator["attributes"]["entries"] = int(config.acc_rows)
        accumulator["attributes"]["depth"] = int(config.acc_rows)
        accumulator["attributes"]["width"] = config.acc_w #in bit
        accumulator["attributes"]["instances"] = config.mesh_dim_x
        accumulator["attributes"]["n_banks"] = config.acc_banks

        pe_rows["name"] = f"PERows[0..{config.mesh_dim_y-1}]"
        registers["attributes"]["width"] = config.data_w
        registers["attributes"]["instances"] = config.dim*config.dim

        #macc["attributes"]["datawidth"] = config.data_w
        #macc["attributes"]["word-bits"] = config.data_w

        with open(arch_out, "w") as f:
            y = yaml.safe_dump(arch, sort_keys=False)
            f.write(y)


    def run_from_config(self, config: GemminiConfig, outdir=None):
        return super().run_from_config(config, outdir)

    def run(self):
        return super().run()