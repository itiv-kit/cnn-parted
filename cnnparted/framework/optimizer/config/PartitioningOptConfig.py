import numpy as np

from  framework.optimizer.config.OptimizerConfig import OptimizerConfig

class PartitioningOptConfig(OptimizerConfig):
    def __init__(self, node_stats: dict, num_pp: int, num_layers: int):
        super().__init__()
        num_platforms = len(node_stats)

        self.n_var_part = (num_pp + 1) + (num_pp) # platform IDs + mapping partition to platform
        self.n_var      = self.n_var_part

        self.n_obj = 6 # latency, energy, throughput, area + link latency + link energy
        self.n_constr = num_platforms + 1 + (num_pp + 1) * 2 + (num_pp + 1) * 2 # design_ids + num_real_pp + latency/energy per partition + latency/energy per link

        self.xl = 0
        self.xu_part = num_platforms * num_layers - 1 #partitioning
        self.xu = [self.xu_part for _ in range(self.n_var_part)]
        self.xu = np.array(self.xu)

        # Aliases
        self.x_len_part = self.n_var_part
        self.x_len = self.x_len_part
        self.g_len = self.n_constr
        self.f_len = self.n_obj
