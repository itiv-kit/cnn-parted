import os
import numpy as np
import pandas as pd

from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.stages.optimization.multitenancy_optimization import MultiTenancyOptimization
from framework.optimizer.solutions.multitenancy_solution import MultitenancySolution
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

@register_required_stage(MultiTenancyOptimization)
class ExportMultiTenancyResults(Stage):
    def __init__(self):
        super().__init__()
    
    def _take_artifacts(self, artifacts: Artifacts):
        self.mt_node_stats = artifacts.multitenancy_optimization.node_stats
        self.config = artifacts.config
        self.run_name = artifacts.args["run_name"]
    
    def run(self, artifacts: Artifacts):
        self._take_artifacts(artifacts)
        mt_sol = MultitenancySolution(node_stats=self.mt_node_stats, config=self.config)
        
        # Store results as json file
        out_dir = os.path.join(self.config.general.work_dir, self.run_name + "_" + "multitenancy_solution.json")
        mt_sol.to_json(out_path=out_dir)

    
    def _update_artifacts(self, artifacts):
        ...