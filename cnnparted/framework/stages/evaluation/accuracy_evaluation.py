import numpy as np
from loguru import logger

from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.stages.evaluation.node_evaluation import NodeEvaluation
from framework.stages.optimization.partitioning_optimization import PartitioningOptimization
from framework.stages.inputs.workload_parser import WorkloadParser
from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.quantization.accuracy_evaluator import AccuracyEvaluator


@register_required_stage(GraphAnalysis, NodeEvaluation, PartitioningOptimization, WorkloadParser)
class AccuracyEvaluation(Stage):
    def __init__(self):
        super().__init__()

    def run(self, artifacts: Artifacts):
        logger.info("Running AccuracyEvaluation stage")
        self._take_artifacts(artifacts)
        if self.config.accuracy is not None:
            accuracy_cfg = self.config.accuracy
            print("Found: ")
            for pareto, sched in self.sol.items():
                print(pareto, len(sched))
            print("Evaluating accuracy...")

            quant = AccuracyEvaluator(self.torch_models, self.node_stats, accuracy_cfg, self.device, self.show_progress)
            quant.eval(self.sol["nondom"], self.n_constr, self.n_var, self.schedules, self.accuracy_function)
        else:
            for i, p in enumerate(self.sol["nondom"]): # achieving aligned csv file
                self.sol["nondom"][i] = np.append(p, float(0))
        for i, p in enumerate(self.sol["dom"]): # achieving aligned csv file
            self.sol["dom"][i] = np.append(p, float(0))
        self._update_artifacts(artifacts)

    def _take_artifacts(self, artifacts: Artifacts):
        self.config = artifacts.config
        self.device = artifacts.device
        self.show_progress = artifacts.config.args.show_progress

        self.accuracy_function = artifacts.workload_parser.accuracy_function
        self.torch_models = artifacts.graph_analysis.ga.torchmodels
        self.schedules = artifacts.graph_analysis.ga.schedules
        self.node_stats = artifacts.node_evaluation.node_stats
        self.sol = artifacts.partitioning_optimization.sol
        self.n_constr = artifacts.partitioning_optimization.n_constr
        self.n_var = artifacts.partitioning_optimization.n_var



    def _update_artifacts(self, artifacts: Artifacts):
        artifacts.partitioning_optimization.sol = self.sol
