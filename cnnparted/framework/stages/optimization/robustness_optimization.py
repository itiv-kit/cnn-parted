from framework.stages.artifacts import Artifacts
from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.inputs.workload_parser import WorkloadParser
from framework.stages.stage_base import Stage, register_required_stage
from framework.optimizer.robustness_optimizer import RobustnessOptimizer
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

@register_required_stage(GraphAnalysis, WorkloadParser)
class RobustnessOptimization(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):
        self._take_artifacts(artifacts)

        robustnessAnalyzer = RobustnessOptimizer(self.work_dir, self.run_name, self.torch_models, 
                                                self.accuracy_function, self.accuracy_cfg,
                                                self.device, self.show_progress)
        q_constr = robustnessAnalyzer.optimize()
        self._update_artifacts(artifacts, q_constr)

    
    def _take_artifacts(self, artifacts: Artifacts):
        self.device = artifacts.device 
        self.work_dir = artifacts.config["general"]["work_dir"]
        self.accuracy_cfg = artifacts.config["accuracy"]
        self.run_name = artifacts.args["run_name"]
        self.show_progress = artifacts.args["p"]
        self.torch_models = artifacts.get_stage_result(GraphAnalysis, "ga").torchmodels
        self.accuracy_function = artifacts.get_stage_result(WorkloadParser, "accuracy_function")


    def _update_artifacts(self, artifacts: Artifacts, q_constr):
        artifacts.set_stage_result(RobustnessOptimization, "q_constr", q_constr)
    