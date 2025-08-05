import os
import numpy as np
import pandas as pd

from framework.stages.stage_base import Stage, register_required_stage
from framework.stages.artifacts import Artifacts
from framework.stages.optimization.multitenancy_optimization import MultiTenancyOptimization
from framework.stages.analysis.graph_analysis import GraphAnalysis
from framework.constants import MODEL_PATH, ROOT_DIR, WORKLOAD_FOLDER

class ExportMultiTenancyResults(Stage):
    def __init__(self):
        super().__init__()
    
    def _take_artifacts(self, artifacts):
        ...
    
    def run(self, artifacts):
        self._take_artifacts(artifacts)
        ...
    
    def _update_artifacts(self, artifacts):
        ...