from ruamel.yaml import YAML
from .config import (CnnpartedArguments, GeneralConfig, WorkloadConfig, 
                     PipelineConfig, DseStageConfig, PartitioningStageConfig,
                     MultiTenancyStageConfig, AccuracyConfig, SystemConfig,
                     )

yaml = YAML(typ="safe")

class ConfigHelper:
    def __init__(self, args: CnnpartedArguments):
        self.args = args
        conf_in = args.conf_file_path
        if isinstance(conf_in, str):
            self.fname = conf_in
            self.config = self.__load_config()
        elif isinstance(conf_in, dict):
            self.fname = ""
            self.config = conf_in
        else:
            raise RuntimeError(f"Invalid argument to ConfigHelper. Expected str or dict but received {type(conf_in)}")
        
        # Parse the configuration
        self.general = GeneralConfig(self.config["general"])
        self.workload = WorkloadConfig(self.config["workload"])
        self.pipeline = PipelineConfig(self.config["pipeline"])
        self.dse = DseStageConfig(self.config["dse"]) if "dse" in self.config else None
        #self.partitioning = PartitioningStageConfig(self.config["partitioning"]) if "partitioning" in self.config else None
        self.partitioning = PartitioningStageConfig(self.config.get("partitioning", {})) #TODO Currently DSE relies on this existing
        self.mt = MultiTenancyStageConfig(self.config["multi-tenancy"]) if "multi-tenancy" in self.config else None
        self.accuracy = AccuracyConfig(self.config["accuracy"]) if "accuracy" in self.config else None
        self.system = SystemConfig(self.config["system"])

    def __load_config(self) -> dict:
        with open(self.fname) as f:
            return yaml.load(f)

    def get_config(self) -> dict:
        return self.config

    def get_system_components(self):
        return self.system.compute, self.system.link

    def get_node_constraints(self):
        node_components, _ = self.get_system_components()
        constr_sys = []
        for node in node_components:
            xl = []
            xu = []
            if node.dse is not None:
                dse_cfg = node.dse
                mem_steps = dse_cfg.step_mem_shared

                xl += dse_cfg.min_pe
                min_size_mem_shared = dse_cfg.min_size_mem_shared
                xl += [min_size // step for (min_size, step) in zip(min_size_mem_shared, mem_steps)]
                xl += dse_cfg.min_size_mem_local

                xu += dse_cfg.max_pe
                max_size_mem_shared = dse_cfg.max_size_mem_shared
                xu += [max_size // step for (max_size, step) in zip(max_size_mem_shared, mem_steps)]
                xu += dse_cfg.max_size_mem_local

                constr_sys += [[xl, xu]]

        return constr_sys

    def get_node_mem_steps(self):
        node_components, _ = self.get_system_components()
        step_sizes = []
        for node in node_components:
            if "dse" in node:
                step_sizes.append(node["dse"]["step_mem_shared"])
        return step_sizes

    def print_all_keys(self,data_dict, indent=''):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                print(f"{indent}{key}:")
                self.print_all_keys(value, indent + '  ')
            else:
                print(f"{indent}{key}")

    def get_system_constraints(self):
        default_constraints = {
            "latency": None,
            "energy": None,
            "area": None,
            "throughput": None,
        }
        #constraints = config.get("system_constraints", default_constraints)
        return default_constraints #TODO Integrate these

    def get_node_memory_sizes(self, node_components ):
        memories=[]
        for node in node_components:
            memories.append( node["max_memory_size"])

        return memories
