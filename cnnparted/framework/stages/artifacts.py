import importlib

from framework.stages.stage_base import Stage
from framework.helpers.config import *
from framework import ConfigHelper

import framework.stages as stages

class Artifacts:
    def __init__(self, config: ConfigHelper, device: str):
        self.config = config
        self.device = device

        # Set type hints and initialize all values to None, if a stage
        # is active it will be initialized in parse_stages()
        self.system_parser: stages.SystemParserArtifacts = None
        self.workload_parser: stages.WorkloadParserArtifacts = None
        self.graph_analysis: stages.GraphAnalysisArtifacts = None
        self.node_evaluation: stages.NodeEvaluationArtifacts = None
        self.design_optimization: stages.DesignOptimizationArtifacts = None
        self.partitioning_optimization: stages.PartitioningOptimizationArtifacts = None
        self.multitenancy_optimization: stages.MultiTenancyOptimizationArtifacts = None
        self.robustness_optimization: stages.RobustnessOptimizationArtifacts = None

        self.parse_stages()
    

    def parse_stages(self):
        package = importlib.import_module("framework.stages")
        name_to_class_map = {stage.__name__: stage for stage in package.__all__}

        self.stage_names = self.config.pipeline.stages
        self.stages = [name_to_class_map[stage] for stage in self.stage_names]
        for stage in self.stage_names:

            stage_cls = name_to_class_map[stage]

            # Initialize artifacts to the actual class
            match stage_cls:
                case stages.SystemParser:
                    self.system_parser = stages.SystemParserArtifacts()
                case stages.WorkloadParser:
                    self.workload_parser = stages.WorkloadParserArtifacts()
                case stages.GraphAnalysis:
                    self.graph_analysis = stages.GraphAnalysisArtifacts()
                case stages.NodeEvaluation:
                    self.node_evaluation = stages.NodeEvaluationArtifacts()
                case stages.DesignPartitioningOptimization:
                    self.design_optimization = stages.DesignOptimizationArtifacts()
                case stages.PartitioningOptimization:
                    self.partitioning_optimization = stages.PartitioningOptimizationArtifacts()
                case stages.RobustnessOptimization:
                    self.robustness_optimization = stages.RobustnessOptimizationArtifacts()
                case stages.MultiTenancyOptimization:
                    self.multitenancy_optimization = stages.MultiTenancyOptimizationArtifacts()

        # Map artifact cls to members, needed for get_oneof method
        self.mapping = {
            stages.SystemParserArtifacts: self.system_parser,
            stages.WorkloadParserArtifacts: self.workload_parser,
            stages.GraphAnalysisArtifacts: self.graph_analysis,
            stages.NodeEvaluationArtifacts: self.node_evaluation,
            stages.DesignOptimizationArtifacts: self.design_optimization,
            stages.PartitioningOptimizationArtifacts: self.partitioning_optimization,
            stages.MultiTenancyOptimizationArtifacts: self.multitenancy_optimization, 
            stages.RobustnessOptimizationArtifacts: self.robustness_optimization,
        }

    def get_oneof_stage_result(self, stage_candidates: tuple[Stage], name: str):
        for stage_candidate in stage_candidates:
            artifact_cls = stages.ARTIFACTS_MAP[stage_candidate]
            if self.mapping[artifact_cls] is not None: 
                target =  self.mapping[artifact_cls]
                return getattr(target, name)
        raise RuntimeError(f"Could not find any of {stage_candidates} in Artifacts")


class StageArtifacts:
    def __init__(self) -> None:
        self.name = None

    def get_result(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            k = vars(self).keys()
            raise AttributeError(f"Field {name} not found in artifacts for \"{self.name}\". Fields are: {k}")

    def set_result(self, name: str, value):
        setattr(self, name, value)