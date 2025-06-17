import os
import shutil
import pickle
from itertools import zip_longest

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from framework.optimizer.config import design_opt_config
from framework.optimizer.config.partitioning_opt_config import PartitioningOptConfig
from framework.optimizer.config.design_opt_config import DesignOptConfig
from framework.helpers.config_helper import ConfigHelper
from framework.stages.artifacts import Artifacts
from framework.stages.evaluation.node_evaluation import NodeEvaluation
from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.inputs.system_parser import SystemParser
from framework.node.node_thread import NodeThread
from framework.dse.simba_adaptor import SimbaConfig, SimbaArchitectureAdaptor
from framework.dse.eyeriss_adaptor import EyerissConfig, EyerissArchitectureAdaptor
from framework.dse.gemmini_adaptor import GemminiConfig, GemminiArchitectureAdaptor

from framework.optimizer.partitioning_optimizer import PartitioningOptimizer

from framework.node.node_evaluator import LayerResult, DesignResult, NodeResult, SystemResult

N_VAR_ACC = {
    "gemmini_like": 6,
    "eyeriss_like": 6,
    "simba_like": 6
}

ACCELERATOR_CONFIG_MAP = {
    "gemmini_like": GemminiConfig,
    "eyeriss_like": EyerissConfig,
    "simba_like": SimbaConfig
}

ACCELERATOR_ADAPTOR_MAP = {
    GemminiConfig: GemminiArchitectureAdaptor,
    EyerissConfig: EyerissArchitectureAdaptor,
    SimbaConfig: SimbaArchitectureAdaptor,
}


# This class does the following:
#   - Get a system vector x which is a system design
#   - Parse the vector into a specific system w.r.t. number of PEs, memory sizes etc.
#   - Evaluate the system with the NodeEvaluator, unless the results are already present in a LUT
class DesignProblem(ElementwiseProblem):
    def __init__(self, 
                 node_components, link_components,
                 node_constraints, node_mem_steps,
                 q_constr: dict,
                 artifacts: Artifacts,
                 partitioning_optimizer_cls,
                 system_dse_config):
        self.work_dir = artifacts.config["general"]["work_dir"]
        self.run_name = artifacts.args["run_name"]
        self.ga = artifacts.get_stage_result(GraphAnalysis, "ga")
        self.show_progress = artifacts.args["p"]
        self.num_pp = artifacts.config["num_pp"]
        self.config = artifacts.config
        self.q_constr = q_constr
        self.system_dse_config = system_dse_config

        self.dse_results_dir = os.path.join(self.work_dir, "dse_results")
        if os.path.exists(self.dse_results_dir):
            shutil.rmtree(self.dse_results_dir)
        os.makedirs(self.dse_results_dir, exist_ok=True)

        self.node_components = node_components
        self.link_components = link_components
        self.node_constraints = node_constraints
        self.num_platforms = len(node_components)
        self.node_mem_steps = node_mem_steps
        
        # Initialize variable to gather all results for later export
        self.system_results = SystemResult()
        for node in self.node_components:
            self.system_results.register_platform(node["id"], node_config=node)

        self.n_var_per_node = [N_VAR_ACC[node["evaluation"]["accelerator"]]  for node in node_components if "dse" in node]

        # We return all values of the partitioning algorithm as "constraint"
        self.accelerator_configs = [ACCELERATOR_CONFIG_MAP[node["evaluation"]["accelerator"]] for node in node_components if "dse" in node]
        self.accelerator_adaptors=[ACCELERATOR_ADAPTOR_MAP[cfg] for cfg in self.accelerator_configs]

        # LUT to prevent evaluating duplicates
        self.dse_node_lut = {}
        for node in node_components:
            if "dse" in node:
                self.dse_node_lut[node["id"]] = {}

        self.partitioning_optimizer_cls: type[PartitioningOptimizer] = partitioning_optimizer_cls

        # Perform the evaluation for all nodes that not suject to DSE
        # to prevent simulating them multiple times
        fixed_nodes = [node for node in node_components if "dse" not in node]
        self.fixed_node_stats = self._eval_nodes(fixed_nodes)
        for ((id, node_stats), node_config)  in zip(self.fixed_node_stats.platforms.items(), fixed_nodes):
            self.system_results.add_platform(id, node_stats)
        print("Done evaluating fixed nodes!")

        num_platforms = sum([node.get("instances", 1) for node in self.node_components])
        part_opt_cfg = PartitioningOptConfig(num_platforms, self.num_pp, len(self.ga.schedules[self.ga.networks[0]]) ) #TODO Mulitple networks

        self.design_opt_config = DesignOptConfig(self.n_var_per_node,
                                                 node_constraints,
                                                 part_opt_cfg,
                                                 self.config["dse"])

        n_var, n_obj, n_constr, xl, xu = self.design_opt_config.get()
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)


    def _eval_nodes(self, nodes, acc_adaptors=[]):
        node_eval_stats = SystemResult()
        
        # If a acc_adaptors have been passed this is a DSE run. Else, init with sane default values to not break zip operation
        if not acc_adaptors:
            cfgs = [[] for _ in nodes]
            cfgs_already_in_lut = [False for _ in nodes]
        else:
            cfgs = [acc_adaptor.config.to_genome() for acc_adaptor in acc_adaptors]
            cfgs_already_in_lut = [self._cfg_in_lut(acc_adaptor.config.to_genome()) for acc_adaptor in acc_adaptors]

        node_threads = [
                NodeThread(component.get('id'), self.ga, component, self.work_dir, self.run_name, 
                           self.show_progress, acc_adaptor=acc_adaptor, save_results=False,
                           dse_system_config=self.system_dse_config)
                for (component, acc_adaptor) in zip_longest(nodes, acc_adaptors)
            ]
        # A thread is instantiated for every accelerator, but only started if it is not present in the LUT
        for (t, in_lut) in zip(node_threads, cfgs_already_in_lut):
            if (not in_lut) and (t.config["evaluation"]["simulator"] not in ["timeloop", "zigzag"]):
                t.start()

        for (t, in_lut) in zip(node_threads, cfgs_already_in_lut):
            if (not in_lut) and (t.config["evaluation"]["simulator"] in ["timeloop", "zigzag"]): # run them simply on main thread
                t.run()
            else:
                t.join()

        for (node_thread, cfg, in_lut) in zip(node_threads, cfgs, cfgs_already_in_lut):
            id,stats = node_thread.getStats()

            # Overwrite empty stats if results are in LUT (stats will be an empty dict otherwise)
            if in_lut:
                stats = self.dse_node_lut[id][tuple(cfg)]

            instances = node_thread.config.get("instances", 1)
            if instances == 1:
                node_eval_stats.add_platform(id, stats)
            else:
                # If the accelerator should be instatiated multiple times, copy the results and generate a unique id
                for i in range(0, instances):
                    id_str = "10" + str(id) + str(i) # generate a unique id for instances
                    node_eval_stats.add_platform(int(id_str), stats)

        # ensure IDs are actually all unique
        all_ids = list(node_eval_stats.platforms.keys())
        assert len(all_ids) == len(set(all_ids)), f"Component IDs are not unique. Found IDs: {all_ids}"

        return node_eval_stats

    def _split_system_input(self, sys_vec: np.ndarray):
        sys_vec = sys_vec.tolist()
        split_vec = []
        start = 0
        for i, n_var in enumerate(self.n_var_per_node):
            split_vec.append(sys_vec[start:start+n_var])
            start += n_var
        return split_vec


    def _cfg_in_lut(self, cfg):
        if not isinstance(cfg, tuple):
            cfg = tuple(cfg)

        for id in self.dse_node_lut.keys():
            if cfg in self.dse_node_lut[id]:
                return True
        return False

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        valid = True

        xs = self._split_system_input(x)
        acc_cfgs = [cfg.from_genome(param, mem_step) for (cfg, param, mem_step) in zip(self.accelerator_configs, xs, self.node_mem_steps, strict=True)]
        acc_adaptors = [adaptor() for adaptor in self.accelerator_adaptors] 

        # Attach the config we want to run to the adaptor
        for (adaptor, cfg) in zip(acc_adaptors, acc_cfgs):
            design_string = "_".join(map(str, cfg.to_genome()))
            adaptor.tl_out_design_name = design_string
            adaptor.config = cfg

        # For every DSE enabled node we call a simulator for evaluation
        dse_nodes = [node for node in self.node_components if "dse" in node]
        dse_node_stats = self._eval_nodes(dse_nodes, acc_adaptors)

        # Prevent adding duplicates to the SystemResult, only unique designs should be found there
        cfgs_already_in_lut = [self._cfg_in_lut(acc_adaptor.config.to_genome()) for acc_adaptor in acc_adaptors]

        # Check if the simulation failed for any of the nodes and save valid results to LUT
        cost = []
        constraints = []
        for (cfg, in_lut, (id, stats)) in zip(xs, cfgs_already_in_lut, dse_node_stats.platforms.items()):
            if not stats.to_dict()["eval"]:
                #simulation failed
                breakpoint()
                valid = False
                cost.append(float(1000000000))
                constraints.append([1,1,1])
            else:
                self.dse_node_lut[id][tuple(cfg)] = dse_node_stats[id]
                design_result = stats.designs[0]

                if not in_lut:
                    self.system_results[id].add_design(design_result)

                layers_data = stats.designs[0].layers_dict
                energy = [data.energy for (key, data) in layers_data.items()]
                latency = [data.latency for (key, data) in layers_data.items()]
                area = stats.designs[0].area
                energy_total = sum(energy)
                latency_total = sum(latency)

                cost.append(float(energy_total*latency_total))
                constraints.append([-energy_total , -latency_total, -area])

        if self.n_obj == 1:
            out["F"] = cost[0]
        else:
            out["F"] = cost
        
        if self.n_constr == 1:
            out["G"] = constraints[0]
        else:
            out["G"] = np.array(constraints).flatten().tolist()

