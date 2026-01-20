import os
from loguru import logger

from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.stages.optimization.multitenancy_optimization import MultiTenancyOptimization
from framework.optimizer.solutions.multitenancy_solution import MultitenancySolution
from framework.stages.evaluation.node_evaluation import NodeEvaluation
from framework.stages.optimization.design_optimization import DesignPartitioningOptimization

@register_required_stage(MultiTenancyOptimization)
class ExportMultiTenancyResults(Stage):
    def __init__(self):
        super().__init__()
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.node_results = artifacts.get_oneof_stage_result((NodeEvaluation, DesignPartitioningOptimization), "node_stats")
        self.mt_solutions = artifacts.multitenancy_optimization.solutions
        self.config = artifacts.config
        self.run_name = artifacts.config.args.run_name
    
    def run(self, artifacts: Artifacts):
        logger.info("Exporting multi-tenancy results")
        self._take_artifacts(artifacts)
        mt_sol = MultitenancySolution(node_stats=self.node_results, mt_solutions=self.mt_solutions, config=self.config)
        
        # Store results as json file
        out_dir = os.path.join(self.config.general.work_dir, self.run_name + "_" + "multitenancy_solution.json")
        mt_sol.to_json(out_path=out_dir)

    
    def _update_artifacts(self, artifacts):
        ...