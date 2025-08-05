from dataclasses import dataclass

from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.evaluation.node_evaluation import NodeEvaluation
from framework.stages.inputs.system_parser import SystemParser
from framework.stages.optimization.robustness_optimization import RobustnessOptimization
from framework.stages.optimization.design_optimization import DesignPartitioningOptimization

from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.optimizer.partitioning_optimizer import PartitioningOptimizer
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

@register_required_stage(GraphAnalysis, (NodeEvaluation, DesignPartitioningOptimization) , SystemParser)
class MultiTenancyOptimization(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts):
        return super().run(artifacts)
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.ga = artifacts.get_oneof_stage_result((NodeEvaluation, DesignPartitioningOptimization), )
    
    def _update_artifacts(self, artifacts):
        return super()._update_artifacts(artifacts)

@dataclass
class MultiTenancyOptimizationArtifacts():
    ...