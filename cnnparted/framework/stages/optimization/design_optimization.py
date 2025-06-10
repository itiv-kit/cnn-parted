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

        # If no exhaustive search should be performed, instantiate the
        # problem for evaluation
        design_problem = None
        design_opt_config = None #TODO
        if self.config["dse"]["optimizer"]["algorithm"] != "exhaustive":
            node_constraints = cfg_helper.get_node_constraints()
            node_mem_steps = cfg_helper.get_node_mem_steps()

            # TODO: Design Problem and Optimizer selected by config
            design_problem = DesignProblem(node_components, link_components, 
                                        node_constraints, node_mem_steps,
                                        self.q_constr,
                                        artifacts, PartitioningOptimizer,
                                        self.config["dse"])
            design_opt_config = design_problem.design_opt_config

        design_optimizer = DesignOptimizer(node_components, 
                                           design_problem, 
                                           self.config["dse"], 
                                           self.work_dir,
                                           self.ga,
                                           self.run_name,
                                           self.show_progress)

        node_stats = design_optimizer.optimize(self.q_constr, artifacts.config)

        top_k = self.config["dse"].get("top_k", -1)
        metric = self.config["dse"].get("optimizattion", "edap")
        strict_mode = self.config["dse"].get("prune_strict", False)
        for id, stats in node_stats.platforms.items():
            stats.prune_designs(top_k, metric, strict_mode)

        self._update_artifacts(artifacts, node_stats, design_opt_config)

    
    def _take_artifacts(self, artifacts: Artifacts):
        self.run_name = artifacts.args["run_name"]
        self.ga = artifacts.get_stage_result(GraphAnalysis, "ga")
        self.show_progress = artifacts.args["p"]
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
        artifacts.config["num_platforms"] = node_stats.get_num_platforms()
        artifacts.set_stage_result(DesignPartitioningOptimization, "node_stats", node_stats)
        artifacts.set_stage_result(DesignPartitioningOptimization, "optimizer_cfg", optimizer_config)

        #num_pp = artifacts.config["general"]["num_pp"]
        #if num_pp == -1:
        #    #num_pp = len(node_stats[list(node_stats.keys())[0]]["eval"]["design_0"]["layers"].keys()) - 1
        #    platform_id = node_stats.get_platform_ids()[0]
        #    breakpoint()
        #    num_pp = len(node_stats[platform_id].designs[0].layers) - 1
        #elif len(node_stats) == 1:
        #    num_pp = 0
        #artifacts.config["num_pp"] = num_pp
