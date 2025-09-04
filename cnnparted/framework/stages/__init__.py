try:
    from cnnparted.framework.stages.evaluation.accuracy_evaluation import (
        AccuracyEvaluation,
    )
    from cnnparted.framework.stages.optimization.robustness_optimization import (
        RobustnessOptimization,
    )

    ROBUSTNESS_AVAILABLE = True
except ImportError:
    # Create dummy classes if dependencies are missing
    class RobustnessOptimization:
        pass

    class AccuracyEvaluation:
        pass

    ROBUSTNESS_AVAILABLE = False

from cnnparted.framework.stages.analysis.graph_analysis import GraphAnalysis
from cnnparted.framework.stages.evaluation.node_evaluation import NodeEvaluation
from cnnparted.framework.stages.export.export_partition_results import (
    ExportPartitionResults,
)
from cnnparted.framework.stages.export.visualize_schedule import VisualizeSchedule
from cnnparted.framework.stages.inputs.system_parser import SystemParser
from cnnparted.framework.stages.inputs.workload_parser import WorkloadParser
from cnnparted.framework.stages.optimization.design_optimization import (
    DesignPartitioningOptimization,
)
from cnnparted.framework.stages.optimization.partitioning_optimization import (
    PartitioningOptimization,
)

__all__ = [
    GraphAnalysis,
    RobustnessOptimization,
    AccuracyEvaluation,
    NodeEvaluation,
    ExportPartitionResults,
    VisualizeSchedule,
    SystemParser,
    WorkloadParser,
    PartitioningOptimization,
    DesignPartitioningOptimization,
]
