from framework.stages.inputs.system_parser import SystemParser, SystemParserArtifacts
from framework.stages.inputs.workload_parser import WorkloadParser, WorkloadParserArtifacts
from framework.stages.analysis.graph_analysis  import GraphAnalysis, GraphAnalysisArtifacts
from framework.stages.evaluation.node_evaluation import NodeEvaluation, NodeEvaluationArtifacts
from framework.stages.optimization.design_optimization import DesignPartitioningOptimization, DesignOptimizationArtifacts
from framework.stages.optimization.partitioning_optimization import PartitioningOptimization, PartitioningOptimizationArtifacts
from framework.stages.optimization.multitenancy_optimization import MultiTenancyOptimization, MultiTenancyOptimizationArtifacts 
from framework.stages.optimization.robustness_optimization import RobustnessOptimization, RobustnessOptimizationArtifacts
from framework.stages.evaluation.accuracy_evaluation import AccuracyEvaluation
from framework.stages.export.export_partition_results import ExportPartitionResults
from framework.stages.export.visualize_schedule import VisualizeSchedule
from framework.stages.export.export_multitenancy_results import ExportMultiTenancyResults

ARTIFACTS_MAP = {
    SystemParser: SystemParserArtifacts,
    WorkloadParser: WorkloadParserArtifacts,
    GraphAnalysis: GraphAnalysisArtifacts,
    NodeEvaluation: NodeEvaluationArtifacts,
    DesignPartitioningOptimization: DesignOptimizationArtifacts,
    PartitioningOptimization: PartitioningOptimizationArtifacts,
    MultiTenancyOptimization: MultiTenancyOptimizationArtifacts,
    RobustnessOptimization: RobustnessOptimizationArtifacts,
    AccuracyEvaluation: None,
    ExportPartitionResults: None,
    ExportMultiTenancyResults: None,
    VisualizeSchedule: None,
}

__all__ = [
    SystemParser, 
    WorkloadParser,
    GraphAnalysis, 
    NodeEvaluation,
    DesignPartitioningOptimization,
    PartitioningOptimization,
    MultiTenancyOptimization,
    RobustnessOptimization,
    AccuracyEvaluation, 
    ExportPartitionResults, 
    ExportMultiTenancyResults,
    VisualizeSchedule,
]