import copy
from math import log2
import pathlib
import yaml

from framework.dse.architecture_mutator import ArchitectureMutator, ArchitectureConfig

class GemminiConfig(ArchitectureConfig):
    def _is_pow_2(self, n):
        return (n & (n-1) == 0) and n != 0


    def __init__(self, mesh_dim, spad_size, acc_size, enable_checks=False):
        self.mesh_dim = mesh_dim
        self.tile_dim = 1 #const for now
        self.dim = self.mesh_dim * self.tile_dim

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

        #Check if given configuration is valid
        if enable_checks:
            assert self._is_pow_2(self.spad_bank_rows), "each SRAM bank must have a power-of-2 rows, to simplify address calculations"
            assert self.spad_bank_rows % (self.dim) == 0, "the number of rows in a bank must be a multiple of the dimensions of the systolic array"
            assert self.dim >= 2, "the systolic array must have a dimension of at least 2"
            assert self._is_pow_2(self.dim), "the systolic array's dimensions must be powers of 2"
            assert self.acc_bank_rows % (self.dim) == 0, "the number of rows in an accumulator bank must be a multiple of the dimensions of the systolic array"
            assert ((self.spad_rows/self.dim) - 2) >= (self.acc_rows/self.dim), "the number os usable scratchpad tiles must be greater than or equal the number of total accumulator tiles"

    def __str__(self):
        return f"GemminiConfig(dim={self.dim}, spad={self.spad_size}kB, acc={self.acc_size}kB)"

    def get_config(self):
        cfg = {}
        cfg["mesh_dim"] = self.mesh_dim
        #cfg["tile_dim"] = self.tile_dim
        cfg["acc_size"] = self.acc_size
        cfg["spad_size"] = self.spad_size
        return cfg
    


class GemminiArchitectureMutator(ArchitectureMutator):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.config: GemminiConfig = None
        search_space_constraints = cfg.get("constraints", {})

        #Boundaries of scratchpad sizes
        self.min_spad_size = search_space_constraints.get("min_spad_size", 256)
        self.max_spad_size = search_space_constraints.get("max_spad_size", 256)

        #Boundaries of accumulator sizes 
        self.min_acc_size = search_space_constraints.get("min_acc_size", 64)
        self.max_acc_size = search_space_constraints.get("max_acc_size", 64)

        self.spad_sizes = []
        self.acc_sizes = []
    
        # Mesh dim parameters
        self.mesh_dim_min = search_space_constraints.get("mesh_dim_min", 16)
        self.mesh_dim_max = search_space_constraints.get("mesh_dim_max", 16)
        _min = int(log2(self.mesh_dim_min))
        _max = int(log2(self.mesh_dim_max))
        self.mesh_dims = [2**x for x in range(_min, _max+1) ]

        if mesh_dim_fix := search_space_constraints.get("mesh_dims_fix"):
            self.mesh_dims = mesh_dim_fix

        # Tile dim parameters
        self.tile_dims = [1] #search_space_constraints.get("tile_dim", [1])

        self.spad_banks = 4
        self.acc_banks = 2

        self.spad_data_width = 8
        self.acc_data_width = 32

        # Generate valid configuration
        self.generate_design_space() 

    def _calc_valid_mem_sizes(self, dim, mem_banks, mem_width, min_mem_size, max_mem_size, rows_max=None, enable_checks=False):
        mem_sizes = []
        if not enable_checks and (dim % 16) == 0:
            enable_checks = True

        n = 4
        while True:
            #spad_bank_rows = 2**n # bank rows must be power of two
            bank_rows = 2**n

            #Consider constraint for scrachpad size in relation to accumulator size
            if enable_checks and rows_max is not None and bank_rows>rows_max:
                break

            # Bank rows must be a multiple of array dimension
            if enable_checks and (bank_rows % dim != 0):
                n += 1
                continue

            mem_size_kilobytes = (bank_rows * mem_banks * mem_width)/(1024*8) 

            if (mem_size_kilobytes < min_mem_size):
                n += 1
                continue
            elif (mem_size_kilobytes <= max_mem_size):
                n += 1
                if enable_checks and (mem_size_kilobytes==int(mem_size_kilobytes)):
                    mem_sizes.append(int(mem_size_kilobytes))
                elif not enable_checks:
                    mem_sizes.append(mem_size_kilobytes)
            else:
                break

        return mem_sizes

    def generate_design_space(self):
        # Generate all possible configs
        for mesh_dim in self.mesh_dims:
            for tile_dim in self.tile_dims:
                dim = mesh_dim*tile_dim
                spad_width = dim * self.spad_data_width
                acc_width  = dim * self.acc_data_width
            
                spad_sizes = self._calc_valid_mem_sizes(dim, self.spad_banks, spad_width, self.min_spad_size, self.max_spad_size)
                acc_sizes = self._calc_valid_mem_sizes(dim, self.acc_banks, acc_width, self.min_acc_size, self.max_acc_size)

                for spad_size in spad_sizes:
                    acc_bank_rows_max = int(((spad_size*1024*8)//(spad_width) - 2*dim) / self.acc_banks)
                    acc_sizes = self._calc_valid_mem_sizes(dim, self.acc_banks, acc_width, self.min_acc_size, self.max_acc_size, rows_max=acc_bank_rows_max)
                    for acc_size in acc_sizes:
                        self.design_space.append(copy.copy(GemminiConfig(mesh_dim, spad_size, acc_size) )) 

    def mutate_arch_constraints(self, config=None, outdir=None):
        # no arch constraints for Gemmini-like config
        pass
    
    def mutate_map_constraints(self, config=None, outdir=None):
        if config is None:
            config = self.config
        if outdir is None:
            outdir = self.tl_out_configs_dir
        base_map_constraints = pathlib.Path(self.tl_in_configs_dir, "constraints", "gemmini_like_map_constraints.yaml")
        constraints_out = pathlib.Path(outdir, "constraints", "gemmini_like_map_constraints.yaml")
        with open(base_map_constraints, "r") as f:
            constraints = yaml.safe_load(f)

        accumulator = constraints["mapspace_constraints"][5] #TODO Magic numbers        
        scratchpad  = constraints["mapspace_constraints"][7] #TODO Magic numbers        

        accumulator["factors"] = f"R=1 S=1 P=1 Q=1 C<={config.dim} M=1 N=1"
        scratchpad["factors"] = f"R=1 S=1 P=1 Q=1 N=1 C=1 M<={config.dim}"

        with open(constraints_out, "w") as f:
            y = yaml.safe_dump(constraints, sort_keys=False)
            f.write(y)


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
        scratchpad["attributes"]["width"] = config.dim * config.data_w
        scratchpad["attributes"]["entries"] = int(config.spad_rows * config.dim)
        scratchpad["attributes"]["n_banks"] = config.spad_banks
        scratchpad["attributes"]["block_size"] = config.dim
        #scratchpad["attributes"]["word-bits"] = config.data_w

        pe_cols["name"] = f"PECols[0..{config.dim-1}]"

        accumulator["attributes"]["entries"] = int(config.acc_rows)
        accumulator["attributes"]["depth"] = int(config.acc_rows)
        accumulator["attributes"]["width"] = config.acc_w #in bit
        accumulator["attributes"]["instances"] = config.dim
        accumulator["attributes"]["n_banks"] = config.acc_banks

        pe_rows["name"] = f"PERows[0..{config.dim-1}]"
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


if __name__ == "__main__":
    #Setup helper classes
    constraints = {"mesh_dim_max": 32,
                "min_spad_size": 512,
                "max_spad_size": 1024,
                "min_acc_size": 256,
                "max_acc_size": 2048}
    cfg = {"dse": {"constraints": None}, "work_dir": {"tl_configs": None}}
    cfg["dse"]["constraints"] = constraints
    cfg["work_dir"]["tl_configs"] = "/home/rh8588/Dokumente/git/cnn-parted/cnnparted/framework/dse/my_work_dir"
    generator = GemminiArchitectureMutator(cfg)
    generator.run()

    print("Generated Gemmini Config:")
    print("  Mesh Dim: ", generator.config.mesh_dim)
    print("  Tile Dim: ", generator.config.tile_dim)
    print("  Array Dim: ", generator.config.mesh_dim*generator.config.tile_dim)
    print("  Spad Capacity: ", generator.config.spad_size)
    print("  Acc Capacity: ", generator.config.acc_size)
