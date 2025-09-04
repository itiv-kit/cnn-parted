# from cnnparted.framework.stages import GraphAnalysis, NodeEvaluation, PartitioningOptimization, WorkloadParser

import numpy as np

from cnnparted.framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER
from cnnparted.framework.quantization.accuracy_evaluator import AccuracyEvaluator
from cnnparted.framework.stages.analysis.graph_analysis import GraphAnalysis
from cnnparted.framework.stages.artifacts import Artifacts
from cnnparted.framework.stages.evaluation.node_evaluation import NodeEvaluation
from cnnparted.framework.stages.inputs.workload_parser import WorkloadParser
from cnnparted.framework.stages.optimization.partitioning_optimization import (
    PartitioningOptimization,
)
from cnnparted.framework.stages.stage_base import Stage, register_required_stage


@register_required_stage(
    GraphAnalysis, NodeEvaluation, PartitioningOptimization, WorkloadParser
)
class AccuracyEvaluation(Stage):
    def __init__(self):
        super().__init__()

    def run(self, artifacts: Artifacts):
        self._take_artifacts(artifacts)
        if accuracy_cfg := self.config["accuracy"]:
            print("Found: ")
            for pareto, sched in self.sol.items():
                print(pareto, len(sched))
            print("Evaluating accuracy...")

            quant = AccuracyEvaluator(
                self.torch_models,
                self.node_stats,
                accuracy_cfg,
                self.device,
                self.show_progress,
            )
            quant.eval(
                self.sol["nondom"],
                self.n_constr,
                self.n_var,
                self.schedules,
                self.accuracy_function,
            )
        else:
            for i, p in enumerate(self.sol["nondom"]):  # achieving aligned csv file
                self.sol["nondom"][i] = np.append(p, float(0))
        for i, p in enumerate(self.sol["dom"]):  # achieving aligned csv file
            self.sol["dom"][i] = np.append(p, float(0))
        self._update_artifacts(artifacts)

    def _take_artifacts(self, artifacts: Artifacts):
        self.config = artifacts.config
        self.torch_models = artifacts.get_stage_result(GraphAnalysis, "ga").torchmodels
        self.node_stats = artifacts.get_stage_result(NodeEvaluation, "node_stats")
        self.device = artifacts.device
        self.show_progress = artifacts.args["p"]
        self.schedules = artifacts.get_stage_result(GraphAnalysis, "ga").schedules
        self.sol = artifacts.get_stage_result(PartitioningOptimization, "sol")
        self.n_constr = artifacts.get_stage_result(PartitioningOptimization, "n_constr")
        self.n_var = artifacts.get_stage_result(PartitioningOptimization, "n_var")
        self.accuracy_function = artifacts.get_stage_result(
            WorkloadParser, "accuracy_function"
        )

    def _update_artifacts(self, artifacts: Artifacts):
        artifacts.set_stage_result(PartitioningOptimization, "sol", self.sol)
