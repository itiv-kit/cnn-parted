from cnnparted.framework.constants import ROOT_DIR
from cnnparted.framework.helpers.design_metrics import SUPPORTED_METRICS, calc_metric
from cnnparted.framework.helpers.visualizer import plotMetricPerConfigPerLayer
from cnnparted.framework.node.node_evaluator import (
    DesignResult,
    LayerResult,
    NodeEvaluator,
    NodeResult,
)


class GenericNode(NodeEvaluator):
    def __init__(self, config):
        self.config = config

    def set_workdir(self, work_dir: str, runname: str, id: int):
        return super().set_workdir(work_dir, runname, id)

    def run(self, layers: list):
        raise NotImplementedError
