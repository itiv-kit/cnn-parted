from framework.stages.analysis.GraphAnalysis import GraphAnalysis
from framework.stages.evaluation.NodeEvaluation import NodeEvaluation
from framework.stages.inputs.SystemParser import SystemParser
from framework.stages.optimization.RobustnessOptimization import RobustnessOptimization

from framework.stages.StageBase import Stage, register_required_stage
from framework.stages.Artifacts import Artifacts
from framework.optimizer.PartitioningOptimizer import PartitioningOptimizer
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

@register_required_stage("GraphAnalysis", "NodeEvaluation", "SystemParser")
class PartitioningOptimization(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):
        self._take_artifacts(artifacts)
        optimizer = PartitioningOptimizer(self.ga, self.num_pp, self.node_stats, self.link_components, self.show_progress)
        n_constr, n_var, sol = optimizer.optimize(self.q_constr, self.config)
        self._update_artifacts(artifacts, n_constr, n_var, sol, optimizer.optimizer_cfg)
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.ga = artifacts.get_stage_result(GraphAnalysis, "ga")
        self.num_pp = artifacts.config["num_pp"]
        self.node_stats = artifacts.get_stage_result(NodeEvaluation, "node_stats")
        self.link_components = artifacts.get_stage_result(SystemParser, "links")
        self.show_progress = artifacts.args["p"]
        self.config = artifacts.config
        if RobustnessOptimization in artifacts.stages:
            self.q_constr = artifacts.get_stage_result(RobustnessOptimization, "q_constr")
        else:
            self.q_constr = {}
        

    def _update_artifacts(self, artifacts: Artifacts, n_constr, n_var, sol, optimizer_config):
        artifacts.set_stage_result(PartitioningOptimization, "n_constr", n_constr)
        artifacts.set_stage_result(PartitioningOptimization, "n_var", n_var)
        artifacts.set_stage_result(PartitioningOptimization, "sol", sol)
        artifacts.set_stage_result(PartitioningOptimization, "optimizer_cfg", optimizer_config)