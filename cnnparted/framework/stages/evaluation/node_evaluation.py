#from framework.stages import GraphAnalysis, SystemParser
from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.inputs.system_parser import SystemParser
from framework.stages.artifacts import Artifacts
from framework.stages.stage_base import Stage, register_required_stage
from framework.node.node_thread import NodeThread
from framework.node.node_evaluator import SystemResult, NodeResult, DesignResult, LayerResult
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

@register_required_stage(GraphAnalysis, SystemParser)
class NodeEvaluation(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):

        self._take_artifacts(artifacts)

        node_eval_stats = SystemResult()
        node_threads = [
                NodeThread(component.get('id'), self.ga, component, self.work_dir, self.run_name, self.show_progress, 
                           dse_system_config=self.dse_system_config)
                for component in self.node_components
            ]

        for t in node_threads:
            if t.config["evaluation"]["simulator"] not in ["timeloop", "zigzag"]:
                t.start()

        for t in node_threads:
            if t.config["evaluation"]["simulator"] in ["timeloop", "zigzag"]: # run them simply on main thread
                t.run()
            else:
                t.join()

        for node_thread in node_threads:
            id,stats = node_thread.getStats()

            instances = node_thread.config.get("instances", 1)
            if instances == 1:
                node_eval_stats.add_platform(id, stats)
            else:
                # If the accelerator should be instatiated multiple times, copy the results and generate a unique id
                for i in range(0, instances):
                    id_str = "10" + str(id) + str(i) # generate a unique id for instances
                    node_eval_stats.add_platform(int(id_str), stats)

        # ensure IDs are actually all unique
        all_ids = list(node_eval_stats.platforms.keys())
        assert len(all_ids) == len(set(all_ids)), f"Component IDs are not unique. Found IDs: {all_ids}"

        self._update_artifacts(artifacts, node_eval_stats)
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.dse_system_config = artifacts.config.get("dse", {})
        self.work_dir = artifacts.config["general"]["work_dir"]
        self.run_name = artifacts.args["run_name"]
        self.ga = artifacts.get_stage_result(GraphAnalysis, "ga")
        self.show_progress = artifacts.args["p"]
        self.node_components = artifacts.get_stage_result(SystemParser, "nodes")

    def _update_artifacts(self, artifacts: Artifacts, node_stats: SystemResult):
        artifacts.set_stage_result(NodeEvaluation, "node_stats", node_stats)
        artifacts.config["num_platforms"] = node_stats.get_num_platforms()
        
        # Update number of partitioning points
        #num_pp = artifacts.config["general"]["num_pp"]
        #if num_pp == -1:
        #    #num_pp = len(node_stats[list(node_stats.keys())[0]]["eval"]["design_0"]["layers"].keys()) - 1
        #    platform_id = node_stats.get_platform_ids()[0]
        #    num_pp = (node_stats[platform_id].designs[0].layers) - 1
        #elif len(node_stats) == 1:
        #    num_pp = 0
        #artifacts.config["num_pp"] = num_pp
    