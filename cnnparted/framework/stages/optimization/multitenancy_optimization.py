from dataclasses import dataclass

from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.evaluation.node_evaluation import NodeEvaluation
from framework.stages.inputs.system_parser import SystemParser
from framework.stages.optimization.robustness_optimization import RobustnessOptimization
from framework.stages.optimization.design_optimization import DesignPartitioningOptimization
from framework.node.node_evaluator import SystemResult

from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.optimizer.multitenancy_partitioning import MultiTenancyPartitioning
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

@register_required_stage(GraphAnalysis, (NodeEvaluation, DesignPartitioningOptimization) , SystemParser)
class MultiTenancyOptimization(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts):
        self._take_artifacts(artifacts)

        mt_optimizer = MultiTenancyPartitioning(self.work_dir, self.node_results, self.ga) #not it
        node_stats = mt_optimizer.optimize()

        self._update_artifacts(artifacts, node_stats)
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.ga = artifacts.graph_analysis.ga
        self.node_results = artifacts.get_oneof_stage_result((NodeEvaluation, DesignPartitioningOptimization), "node_stats")
        self.work_dir = artifacts.config.general.work_dir
    
    def _update_artifacts(self, artifacts: Artifacts, node_stats: SystemResult):
        artifacts.multitenancy_optimization.node_stats = node_stats


@dataclass
class MultiTenancyOptimizationArtifacts():
    node_stats: SystemResult = None