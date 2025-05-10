from framework.optimizer.design_optimizer import DesignOptimizer
from framework.helpers.config_helper import ConfigHelper
from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.evaluation.node_evaluation import NodeEvaluation
from framework.stages.inputs.system_parser import SystemParser
from framework.stages.optimization.robustness_optimization import RobustnessOptimization

from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.optimizer.partitioning_optimizer import PartitioningOptimizer
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

from framework.optimizer.design_problem import DesignProblem
from framework.optimizer.design_partitioning_cooptimizer import DesignPartitioningOptimizer

@register_required_stage(GraphAnalysis, SystemParser)
class DesignPartitioningOptimization(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):
        self._take_artifacts(artifacts)

        cfg_helper = ConfigHelper(self.config)
        node_components, link_components = cfg_helper.get_system_components()
        node_constraints = cfg_helper.get_node_constraints()

        # Instatiate the problem and optimizers
        # TODO: Design Problem and Optimizer selected by config
        design_problem = DesignProblem(node_components, link_components, 
                                       node_constraints, 
                                       self.q_constr,
                                       artifacts, PartitioningOptimizer)

        design_optimizer = DesignOptimizer(node_components, design_problem, self.config["dse"], self.work_dir)

        #opt = DesignPartitioningOptimizer(self.ga, 
        #                                  self.num_pp, 
        #                                  self.link_components,
        #                                  self.show_progress,
        #                                  design_optimizer,
        #                                  partitioning_optimizer)
        node_stats = design_optimizer.optimize(self.q_constr, artifacts.config)

        self._update_artifacts(artifacts, node_stats, design_problem.design_opt_config)
    
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
