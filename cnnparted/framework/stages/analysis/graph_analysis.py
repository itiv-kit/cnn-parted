from loguru import logger

from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.graph_analyzer import GraphAnalyzer
from framework.stages.inputs.workload_parser import WorkloadParser

@register_required_stage(WorkloadParser)
class GraphAnalysis(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):
        logger.info("Running GraphAnalysis stage")
        self._take_artifacts(artifacts)
        ga = GraphAnalyzer(self.work_dir, self.run_name, self.workloads, artifacts.config.args.show_progress, self.config)

        self._update_artifacts(artifacts, ga)
        
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.config = artifacts.config
        self.work_dir = artifacts.config.general.work_dir
        self.run_name = artifacts.config.args.run_name
        self.workloads = artifacts.workload_parser.workloads 

    def _update_artifacts(self, artifacts: Artifacts, ga: GraphAnalyzer):
        artifacts.graph_analysis.ga = ga



class GraphAnalysisArtifacts:
    ga: GraphAnalyzer = None
