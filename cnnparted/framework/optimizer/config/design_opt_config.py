import numpy as np

from framework.optimizer.config.optimizer_config import OptimizerConfig
from framework.optimizer.config.partitioning_opt_config import PartitioningOptConfig

class DesignOptConfig(OptimizerConfig):
    def __init__(self, 
                n_var_per_node, 
                node_constraints, 
                part_opt_cfg: PartitioningOptConfig,
                dse_config: dict):
        self.n_var_per_node = n_var_per_node
        self.n_var = sum(n_var_per_node)

        match dse_config["optimization"]:
            case "edp":
                self.n_obj = 1
            case "edap":
                self.n_obj = 1
            #case "ppa":
            #    self.n_obj = 3
            case _:
                raise RuntimeError(f"Invalid optimization option for DSE")

        self.part_opt_cfg = part_opt_cfg
        self.n_constr = part_opt_cfg.x_len + part_opt_cfg.g_len + part_opt_cfg.f_len + 1 + 1

        self.xl = np.array([node_constraint[0] for node_constraint in node_constraints]).flatten()
        self.xu = np.array([node_constraint[1] for node_constraint in node_constraints]).flatten()

        # Aliases
        self.x_len = self.n_var
        self.g_len = self.n_constr
        self.f_len = self.n_obj
