import numpy as np

from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.evaluation.node_evaluation import NodeEvaluation
from framework.stages.inputs.system_parser import SystemParser
from framework.stages.optimization.robustness_optimization import RobustnessOptimization
from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts

from framework.optimizer.design_optimizer import DesignOptimizer
from framework.optimizer.partitioning_optimizer import PartitioningOptimizer
from framework.optimizer.design_problem import DesignProblem

from framework.helpers.design_metrics import calc_metric
from framework.helpers.config_helper import ConfigHelper

@register_required_stage(GraphAnalysis, SystemParser)
class DesignPartitioningOptimization(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):
        self._take_artifacts(artifacts)

        cfg_helper = ConfigHelper(self.config)
        node_components, link_components = cfg_helper.get_system_components()
        node_constraints = cfg_helper.get_node_constraints()
        node_mem_steps = cfg_helper.get_node_mem_steps()

        # Instatiate the problem and optimizers
        # TODO: Design Problem and Optimizer selected by config
        design_problem = DesignProblem(node_components, link_components, 
                                       node_constraints, node_mem_steps,
                                       self.q_constr,
                                       artifacts, PartitioningOptimizer,
                                       self.config["dse"])

        design_optimizer = DesignOptimizer(node_components, design_problem, self.config["dse"], self.work_dir)

        node_stats = design_optimizer.optimize(self.q_constr, artifacts.config)

        top_k = self.config["dse"].get("top_k", -1)
        metric = self.config["dse"].get("optimizattion", "edap")
        strict_mode = self.config["dse"].get("prune_strict", False)
        for id, stats in node_stats.items():
            pruned_stats = self._prune_accelerator_designs(stats["eval"], top_k=top_k, metric=metric, is_dse=True, strict=strict_mode)
            node_stats[id]["eval"] = pruned_stats

        self._update_artifacts(artifacts, node_stats, design_problem.design_opt_config)

    def _prune_accelerator_designs(self, stats: dict[str, dict], top_k: int, metric: str, is_dse: bool, strict=False):
        # If there are less designs than top_k simply return the given list
        if len(stats) <= top_k or not is_dse or top_k == -1:
            return stats

        # The metric_per_design array has this structure, with
        # every cell holding EAP, EDP or some other metric:
        #  x | l0 | l1 | l2 | l3 |
        # ------------------------
        # d0 | ...| ...| ...| ...|
        # d1 | ...| ...| ...| ...|
        metric_per_design = []
        energy_per_design = []
        latency_per_design = []
        area_per_design = []

        for tag, design in stats.items():
            #tag = design["tag"]
            layers = design["layers"]
            energy_per_layer = []
            latency_per_layer = []
            for name, layer in layers.items():
                energy_per_layer.append(layer["energy"])
                latency_per_layer.append(layer["latency"])

            energy_per_design.append(energy_per_layer)
            latency_per_design.append(latency_per_layer)
            area_per_design.append([layer["area"]])

        metric_per_design = calc_metric(np.array(energy_per_design), np.array(latency_per_design), np.array(area_per_design), metric, reduction=strict)

        # Now, we need to find the top_k designs per layer
        design_candidates = []
        if not strict:
            for col in metric_per_design.T:
                metric_for_layer = col.copy()
                metric_for_layer = np.argsort(metric_for_layer)

                for i in metric_for_layer[0:top_k]:
                    design_candidates.append(f"design_{i}")
        else:
            metric_per_design_sort = np.argsort(metric_per_design.flatten())
            for i in metric_per_design_sort[0:top_k]:
                design_candidates.append(f"design_{i}")

        design_candidates = np.unique(design_candidates)

        pruned_stats = {tag: results for tag, results in stats.items() if tag in design_candidates}

        return pruned_stats
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.ga = artifacts.get_stage_result(GraphAnalysis, "ga")
        self.num_pp = artifacts.config["num_pp"]
        self.link_components = artifacts.get_stage_result(SystemParser, "links")
        self.work_dir = artifacts.config["general"]["work_dir"]

        self.show_progress = artifacts.args["p"]
        self.config = artifacts.config

        if RobustnessOptimization in artifacts.stages:
            self.q_constr = artifacts.get_stage_result(RobustnessOptimization, "q_constr")
        else:
            self.q_constr = {}
        

    def _update_artifacts(self, artifacts: Artifacts, node_stats, optimizer_config):
        artifacts.config["num_platforms"] = len(node_stats)
        artifacts.set_stage_result(DesignPartitioningOptimization, "node_stats", node_stats)
        artifacts.set_stage_result(DesignPartitioningOptimization, "optimizer_cfg", optimizer_config)
