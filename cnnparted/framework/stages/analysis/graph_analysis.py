from framework.stages.stage_base import Stage
from framework.stages.artifacts import Artifacts
from framework.graph_analyzer import GraphAnalyzer
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

class GraphAnalysis(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):
        self._take_artifacts(artifacts)
        ga = GraphAnalyzer(self.work_dir, self.run_name, tuple(self.input_size), artifacts.args["p"])
        ga.find_schedules(self.num_topos)

        self._update_artifacts(artifacts, ga)
        
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.work_dir = artifacts.config["general"]["work_dir"]
        self.run_name = artifacts.args["run_name"]
        self.input_size =  artifacts.config["workload"]["input-size"]
        self.num_topos = artifacts.config["general"]["num_topos"]

    def _update_artifacts(self, artifacts: Artifacts, ga: GraphAnalyzer):
        artifacts.set_stage_result(GraphAnalysis, "ga", ga)

    