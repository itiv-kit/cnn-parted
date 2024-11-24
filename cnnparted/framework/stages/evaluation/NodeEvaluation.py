#from framework.stages import GraphAnalysis, SystemParser
from framework.stages.analysis.GraphAnalysis import GraphAnalysis
from framework.stages.inputs.SystemParser import SystemParser
from framework.stages.Artifacts import Artifacts
from framework.stages.StageBase import Stage, register_required_stage
from framework.node.NodeThread import NodeThread
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

@register_required_stage("GraphAnalysis", "SystemParser")
class NodeEvaluation(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):

        self._take_artifacts(artifacts)

        nodeStats = {}
        node_threads = [
                NodeThread(component.get('id'), self.ga, component, self.work_dir, self.run_name, self.show_progress)
                for component in self.node_components
            ]

        for t in node_threads:
            if not t.config.get("timeloop") and not t.config.get("zigzag"):
                t.start()

        for t in node_threads:
            if t.config.get("timeloop") or t.config.get("zigzag"): # run them simply on main thread
                t.run()
            else:
                t.join()

        for node_thread in node_threads:
            id,stats = node_thread.getStats()
            nodeStats[id] = stats

        self._update_artifacts(artifacts, nodeStats)
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.work_dir = artifacts.config["general"]["work_dir"]
        self.run_name = artifacts.args["run_name"]
        self.ga = artifacts.get_stage_result(GraphAnalysis, "ga")
        self.show_progress = artifacts.args["p"]
        self.node_components = artifacts.get_stage_result(SystemParser, "nodes")

    def _update_artifacts(self, artifacts: Artifacts, nodeStats):
        artifacts.set_stage_result(NodeEvaluation, "nodeStats", nodeStats)
        artifacts.config["num_platforms"] = len(nodeStats)
        
        # Update number of partitioning points
        num_pp = artifacts.config["general"]["num_pp"]
        if num_pp == -1:
            num_pp = len(nodeStats[list(nodeStats.keys())[0]]["eval"]["design_0"]["layers"].keys()) - 1
        elif len(nodeStats) == 1:
            num_pp = 0
        artifacts.config["num_pp"] = num_pp
    