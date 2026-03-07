from dataclasses import dataclass
import copy
from loguru import logger

#from framework.stages import GraphAnalysis, SystemParser
from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.inputs.system_parser import SystemParser
from framework.stages.artifacts import Artifacts
from framework.stages.stage_base import Stage, register_required_stage
from framework.node.node_thread import NodeThread
from framework.node.node_evaluator import SystemResult

@register_required_stage(GraphAnalysis, SystemParser)
class NodeEvaluation(Stage):
    def __init__(self):
        super().__init__()
    
    def run(self, artifacts: Artifacts):
        logger.info("Running NodeEvaluation stage")
        self._take_artifacts(artifacts)

        node_eval_stats = SystemResult()
        node_threads = [
                NodeThread(component.id, self.ga, component, self.work_dir, self.run_name, self.show_progress, 
                           dse_system_config=self.dse_system_config)
                for component in self.node_components
            ]

        for t in node_threads:
            if t.config.evaluation.simulator not in ["timeloop", "zigzag"]:
                t.start()

        for t in node_threads:
            if t.config.evaluation.simulator in ["timeloop", "zigzag"]: # run them simply on main thread
                t.run()
            else:
                t.join()

        for node_thread in node_threads:
            id,stats = node_thread.getStats()

            node_eval_stats.add_platform(id, stats)
            if node_thread.config.virtual_ids is not None:
                # If the accelerator should be instatiated multiple times, copy the results and look-up the unique id
                for v_id_int in node_thread.config.virtual_ids.values():
                    node_eval_stats.add_platform(v_id_int, copy.deepcopy(stats))
        
        # For all models that have been evaluated: check if the deadline should be automatically set
        # based on the estimated computation times on the different platforms
        for name, model in self.ga.models.items():
            if model.has_auto_deadline:
                model_latencies = []
                for id in node_eval_stats.get_unique_platforms():
                    network_on_id_latency = sum([lyr.latency for lyr in node_eval_stats.platforms[id].designs[0].networks[name].layers.values()])
                    model_latencies.append(network_on_id_latency)
                model.deadline = ( max(model_latencies) * model.deadline_mul ) / model.deadline_div
        
        self._update_artifacts(artifacts, node_eval_stats)
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.dse_system_config = artifacts.config.dse
        self.work_dir = artifacts.config.general.work_dir
        self.run_name = artifacts.config.args.run_name
        self.show_progress = artifacts.config.args.show_progress
        self.ga = artifacts.graph_analysis.ga
        self.node_components = artifacts.system_parser.nodes

    def _update_artifacts(self, artifacts: Artifacts, node_stats: SystemResult):
        artifacts.node_evaluation.node_stats = node_stats
        artifacts.config.system.num_platforms = node_stats.get_num_platforms()
        
@dataclass
class NodeEvaluationArtifacts:
    node_stats: SystemResult = None