from itertools import zip_longest

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from framework.optimizer.config.partitioning_opt_config import PartitioningOptConfig
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

N_VAR_ACC = {
    "gemmini_like": 4,
    "eyeriss_like": 6,
    "simba_like": 7
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
#   - After that, call the PartitioningOptimizer with the system that was just evaluated
class DesignProblem(ElementwiseProblem):
    def __init__(self, node_components, link_components,
                 node_constraints, artifacts: Artifacts,
                 partitioning_optimizer_cls):
        self.work_dir = artifacts.config["general"]["work_dir"]
        self.run_name = artifacts.args["run_name"]
        self.ga = artifacts.get_stage_result(GraphAnalysis, "ga")
        self.show_progress = artifacts.args["p"]
        self.num_pp = artifacts.config["num_pp"]
        self.config = artifacts.config

        self.node_components = node_components
        self.link_components = link_components
        self.node_constraints = node_constraints
        self.num_platforms = len(node_components)

        self.n_var_per_node = [N_VAR_ACC[node["timeloop"]["accelerator"]]  for node in node_components if "dse" in node]
        n_var = sum(self.n_var_per_node)

        match self.config["dse"]["optimization"]:
            case "ppa":
                n_obj = 3
            case "edp":
                n_obj = 1
            case "edap":
                n_obj = 1
            case _:
                breakpoint()
                raise RuntimeError(f"Invalid optimization option for DSE")

        num_platforms = sum([node.get("instances", 1) for node in self.node_components])
        part_opt_cfg = PartitioningOptConfig(num_platforms, self.num_pp, len(self.ga.schedules) )

        # We return all values of the partitioning algorithm as "constraint"
        n_constr = part_opt_cfg.x_len + part_opt_cfg.g_len + part_opt_cfg.f_len + 1 + 1
        self.accelerator_configs = [ACCELERATOR_CONFIG_MAP[node["timeloop"]["accelerator"]] for node in node_components if "dse" in node]
        self.accelerator_adaptors=[ACCELERATOR_ADAPTOR_MAP[cfg] for cfg in self.accelerator_configs]

        xl = np.array([node_constraint[0] for node_constraint in node_constraints]).flatten()
        xu = np.array([node_constraint[1] for node_constraint in node_constraints]).flatten()

        self.partitioning_optimizer_cls: type[PartitioningOptimizer] = partitioning_optimizer_cls

        fixed_nodes = [node for node in node_components if "dse" not in node]
        self.fixed_node_stats = self._eval_nodes(fixed_nodes)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)


    def _eval_nodes(self, nodes, acc_adaptors=[]):
        node_stats = {}
        node_threads = [
                NodeThread(component.get('id'), self.ga, component, self.work_dir, self.run_name, self.show_progress, acc_adaptor=acc_adaptor)
                for (component, acc_adaptor) in zip_longest(nodes, acc_adaptors)
            ]

        for t in node_threads:
            if not t.config.get("timeloop") and not t.config.get("zigzag"):
                t.start()

        for t in node_threads:
            if t.config.get("timeloop") or t.config.get("zigzag"): # run them simply on main thread
                t.run()
            else:
                t.join()

        for node_thread in node_threads:
            id,stats = node_thread.getStats()

            instances = node_thread.config.get("instances", 1)
            if instances == 1:
                node_stats[id] = stats
            else:
                # If the accelerator should be instatiated multiple times, copy the results and generate a unique id
                for i in range(0, instances):
                    id_str = "10" + str(id) + str(i) # generate a unique id for instances
                    node_stats[int(id_str)] = stats

        # ensure IDs are actually all unique
        all_ids = list(node_stats.keys())
        assert len(all_ids) == len(set(all_ids)), f"Component IDs are not unique. Found IDs: {all_ids}"

        return node_stats

    def _split_system_input(self, sys_vec: np.ndarray):
        sys_vec = sys_vec.tolist()
        split_vec = []
        start = 0
        for i, n_var in enumerate(self.n_var_per_node):
            split_vec.append(sys_vec[start:start+n_var])
            start += n_var
        return split_vec

    def _pareto_edp(self, comp_paretos : np.ndarray) -> np.ndarray:
        comp_paretos = np.delete(comp_paretos, np.s_[2:], axis=1) # only consider latency and energy
        comp_paretos = np.hstack([comp_paretos, np.expand_dims(np.prod(comp_paretos, axis=1), 1)])
        return comp_paretos
    
    def _pareto_ppa(self, comp_paretos : np.ndarray) -> np.ndarray:
        comp_paretos = np.delete(comp_paretos, np.s_[-2:], axis=1) # remove link metrics
        comp_paretos = np.delete(comp_paretos, np.s_[-2], axis=1) # remove throughput metric
        return comp_paretos

    def _calc_cost(self, objectives: np.ndarray) -> np.ndarray:
        match self.config["dse"]["optimization"]:
            case "ppa":
                objectives_cut = self._pareto_ppa(objectives)
                cost = objectives_cut
            case "edp":
                objectives_cut = self._pareto_edp(objectives)
                cost = np.prod(objectives_cut, axis=1)
            case "edap":
                objectives_cut = self._pareto_ppa(objectives)
                cost = np.prod(objectives_cut, axis=1)
            case _:
                raise RuntimeError(f"Invalid optimization option for DSE")

        return cost
        

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        valid = True #TODO: Can this ever be invalid?

        x = self._split_system_input(x)
        acc_cfgs = [cfg(*param) for (cfg, param) in zip(self.accelerator_configs, x, strict=True)]
        acc_adaptors = [adaptor({}) for adaptor in self.accelerator_adaptors] 

        # Set the config we want to run
        for (adaptor, cfg) in zip(acc_adaptors, acc_cfgs):
            adaptor.config = cfg

        dse_nodes = [node for node in self.node_components if "dse" in node]
        dse_node_stats = self._eval_nodes(dse_nodes, acc_adaptors)

        node_stats = self.fixed_node_stats | dse_node_stats

        q_constr = {}
        part_opt = self.partitioning_optimizer_cls(self.ga, self.num_pp, node_stats, self.link_components, self.show_progress)
        n_constr, n_var, sol = part_opt.optimize(q_constr, self.config)
        n_var_total = part_opt.optimizer_cfg.n_var + part_opt.optimizer_cfg.f_len + part_opt.optimizer_cfg.g_len + 1 + 1

        nondom = np.array(sol["nondom"])
        objectives = [data[n_constr+n_var+1:] for data in nondom]

        cost = self._calc_cost(objectives)
        out["F"] = np.array(min(cost))

        res = np.hstack( (nondom, np.reshape(cost,shape=(-1, 1)) ) )
        res = res[cost.argmin()] #TODO Workaround until I have a better solution for variable number of constraints

        if valid:
            out["G"] = -res
        else:
            out["G"] = res





