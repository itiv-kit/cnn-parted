from dataclasses import dataclass
from loguru import logger

from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.evaluation.node_evaluation import NodeEvaluation
from framework.stages.inputs.system_parser import SystemParser
from framework.stages.optimization.robustness_optimization import RobustnessOptimization
from framework.stages.optimization.design_optimization import DesignPartitioningOptimization

from framework.node.node_evaluator import SystemResult
from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.optimizer.partitioning_optimizer import PartitioningOptimizer
from framework.optimizer.config.partitioning_opt_config import PartitioningOptConfig

@register_required_stage(GraphAnalysis, (NodeEvaluation, DesignPartitioningOptimization) , SystemParser)
class PartitioningOptimization(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):
        logger.info("Running partitioning optimization stage")
        self._take_artifacts(artifacts)
        optimizer = PartitioningOptimizer(self.ga, self.num_pp, self.node_stats, self.link_components, self.show_progress)
        n_constr, n_var, sol = optimizer.optimize(self.q_constr, self.config)
        self._update_artifacts(artifacts, n_constr, n_var, sol, optimizer.optimizer_cfg)
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.ga = artifacts.graph_analysis.ga
        self.ga.find_schedules(artifacts.config.partitioning.num_topos)
        self.node_stats: SystemResult = artifacts.get_oneof_stage_result((NodeEvaluation, DesignPartitioningOptimization), "node_stats")

        num_pp = artifacts.config.partitioning.num_pp

        # TODO Not sure if num_pp should depend on GraphAnalysis or Platform evaluation
        if num_pp == -1:
            num_pp = len(self.ga.schedules[self.ga.networks[0]][0]) - 1
        #elif len(node_stats) == 1:
        #    num_pp = 0

        #if num_pp == -1:
        #    platform_id = self.node_stats.get_platform_ids()[0]
        #    design = self.node_stats[platform_id].designs[0]
        #    network =  self.node_stats[platform_id].designs[0].get_networks()[0]
        #    num_pp = len(design.networks[network].layers) - 1
        #elif len(node_stats) == 1:
        #    num_pp = 0

        artifacts.config.partitioning.num_pp = num_pp
        self.num_pp = artifacts.config.partitioning.num_pp

        self.link_components = artifacts.system_parser.links
        self.show_progress = artifacts.config.args.show_progress
        self.config = artifacts.config
        if RobustnessOptimization in artifacts.stages:
            self.q_constr = artifacts.robustness_optimization.q_constr
        else:
            self.q_constr = {}
        

    def _update_artifacts(self, artifacts: Artifacts, n_constr, n_var, sol, optimizer_config):
        artifacts.partitioning_optimization.n_constr = n_constr
        artifacts.partitioning_optimization.n_var = n_var
        artifacts.partitioning_optimization.sol = sol
        artifacts.partitioning_optimization.optimizer_cfg = optimizer_config


@dataclass
class PartitioningOptimizationArtifacts:
    n_constr: int = None
    n_var: int = None
    sol: dict = None
    optimizer_cfg: PartitioningOptConfig = None
