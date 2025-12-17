from dataclasses import dataclass
from loguru import logger

from framework.model.scheduling import SystemSchedule
from framework.optimizer.multitenancy_partitioning import LayerTracker
from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.evaluation.node_evaluation import NodeEvaluation
from framework.stages.inputs.system_parser import SystemParser
from framework.stages.optimization.design_optimization import DesignPartitioningOptimization

from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.optimizer.multitenancy_partitioning import MultiTenancyPartitioning

@register_required_stage(GraphAnalysis, (NodeEvaluation, DesignPartitioningOptimization) , SystemParser)
class MultiTenancyOptimization(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts):
        logger.info("Running multi-tenancy optimization stage")
        self._take_artifacts(artifacts)

        mt_optimizer = MultiTenancyPartitioning(self.work_dir, self.node_results, self.ga, self.config)
        solutions = mt_optimizer.optimize()

        self._update_artifacts(artifacts, solutions)
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.ga = artifacts.graph_analysis.ga
        self.node_results = artifacts.get_oneof_stage_result((NodeEvaluation, DesignPartitioningOptimization), "node_stats")
        self.work_dir = artifacts.config.general.work_dir
        self.config = artifacts.config
    
    def _update_artifacts(self, artifacts: Artifacts, solutions):
        artifacts.multitenancy_optimization.solutions = solutions


@dataclass
class MultiTenancyOptimizationArtifacts():
    solutions: list[tuple[LayerTracker, SystemSchedule]] = None