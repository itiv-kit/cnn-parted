import os
import shutil
from itertools import zip_longest

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from framework.helpers.config import NodeConfig, LinkConfig, DseStageConfig

from framework.optimizer.config.partitioning_opt_config import PartitioningOptConfig
from framework.optimizer.config.design_opt_config import DesignOptConfig
from framework.dse.interfaces import ArchitectureConfig, Genome
from framework.stages.artifacts import Artifacts
from framework.node.node_thread import NodeThread
from framework.node.evaluation_registry import AcceleratorRegistry

from framework.optimizer.partitioning_optimizer import PartitioningOptimizer
from framework.node.node_evaluator import SystemResult

type DseArchitecture = ArchitectureConfig | Genome



# This class does the following:
#   - Get a system vector x which is a system design
#   - Parse the vector into a specific system w.r.t. number of PEs, memory sizes etc.
#   - Evaluate the system with the NodeEvaluator, unless the results are already present in a LUT
class DesignProblem(ElementwiseProblem):
    def __init__(self, 
                 node_components: list[NodeConfig], link_components: list[LinkConfig],
                 node_constraints, node_mem_steps,
                 q_constr: dict,
                 artifacts: Artifacts,
                 partitioning_optimizer_cls,
                 system_dse_config: DseStageConfig):
        self.work_dir = artifacts.config.general.work_dir
        self.run_name = artifacts.config.args.run_name
        self.ga = artifacts.graph_analysis.ga 
        self.show_progress = artifacts.config.args.show_progress
        self.num_pp = artifacts.config.partitioning.num_pp
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
            self.system_results.register_platform(node.id, node_config=node)

        self.accelerator_registry = AcceleratorRegistry()

        # We return all values of the partitioning algorithm as "constraint"
        self.accelerator_names: list[str] = [node.evaluation.accelerator for node in node_components if node.dse is not None] #TODO Why was the name a string of its own?
        self.accelerator_config_types: list[Genome|ArchitectureConfig] = [self.accelerator_registry[name] for name in self.accelerator_names]
        self.n_var_per_node: list[int] = [accelerator_config_t.n_var_opt for accelerator_config_t in self.accelerator_config_types]
        #self.n_var_per_node : list[int] = [N_VAR_ACC[node.evaluation.accelerator]  for node in node_components if node.dse is not None]
        self.accelerator_adaptors=[AcceleratorRegistry.get_adaptor(cfg) for cfg in self.accelerator_names]

        # LUT to prevent evaluating duplicates
        self.dse_node_lut = {}
        for node in node_components:
            if node.dse is not None:
                self.dse_node_lut[node.id] = {}

        self.partitioning_optimizer_cls: type[PartitioningOptimizer] = partitioning_optimizer_cls

        # Perform the evaluation for all nodes that not suject to DSE
        # to prevent simulating them multiple times
        fixed_nodes = [node for node in node_components if node.dse is not None]
        self.fixed_node_stats = self._eval_nodes(fixed_nodes)
        for ((id, node_stats), node_config)  in zip(self.fixed_node_stats.platforms.items(), fixed_nodes):
            self.system_results.add_platform(id, node_stats)
        print("Done evaluating fixed nodes!")

        num_platforms = sum([node.instances for node in self.node_components])
        part_opt_cfg = PartitioningOptConfig(num_platforms, self.num_pp, len(self.ga.schedules[self.ga.networks[0]]) ) #TODO Mulitple networks

        self.design_opt_config = DesignOptConfig(self.n_var_per_node,
                                                 node_constraints,
                                                 part_opt_cfg)

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
            if (not in_lut) and (t.config.evaluation.simulator not in ["timeloop", "zigzag"]):
                t.start()

        for (t, in_lut) in zip(node_threads, cfgs_already_in_lut):
            if (not in_lut) and (t.config.evaluation.simulator in ["timeloop", "zigzag"]): # run them simply on main thread
                t.run()
            else:
                t.join()

        for (node_thread, cfg, in_lut) in zip(node_threads, cfgs, cfgs_already_in_lut):
            id,stats = node_thread.getStats()

            # Overwrite empty stats if results are in LUT (stats will be an empty dict otherwise)
            if in_lut:
                stats = self.dse_node_lut[id][tuple(cfg)]

            instances = node_thread.config.instances
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
        #valid = True

        xs = self._split_system_input(x)
        acc_cfgs = [cfg.from_genome(param, mem_step) for (cfg, param, mem_step) in zip(self.accelerator_config_types, xs, self.node_mem_steps, strict=True)]
        acc_adaptors = [adaptor() for adaptor in self.accelerator_adaptors] 

        # Attach the config we want to run to the adaptor
        for (adaptor, cfg) in zip(acc_adaptors, acc_cfgs):
            design_string = "_".join(map(str, cfg.to_genome()))
            adaptor.tl_out_design_name = design_string
            adaptor.config = cfg

        # For every DSE enabled node we call a simulator for evaluation
        dse_nodes = [node for node in self.node_components if node.dse is not None]
        dse_node_stats = self._eval_nodes(dse_nodes, acc_adaptors)

        # Prevent adding duplicates to the SystemResult, only unique designs should be found there
        cfgs_already_in_lut = [self._cfg_in_lut(acc_adaptor.config.to_genome()) for acc_adaptor in acc_adaptors]

        # Check if the simulation failed for any of the nodes and save valid results to LUT
        cost = []
        constraints = []
        for (cfg, in_lut, (id, stats)) in zip(xs, cfgs_already_in_lut, dse_node_stats.platforms.items()):
            if not stats.to_dict()["eval"]:
                #simulation failed
                #valid = False
                cost.append(float(1000000000))
                constraints.append([1,1,1])
            else:
                self.dse_node_lut[id][tuple(cfg)] = dse_node_stats[id]
                design_result = stats.designs[0]

                if not in_lut:
                    self.system_results[id].add_design(design_result)

                network_result = stats.designs[0].networks[self.ga.networks[0]] #TODO Multiple networks
                energy = [data.energy for _, data in network_result]
                latency = [data.latency for _, data in network_result]
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

