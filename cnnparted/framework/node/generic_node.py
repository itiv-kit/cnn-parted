from framework.constants import ROOT_DIR
from framework.helpers.visualizer import plotMetricPerConfigPerLayer
from framework.helpers.design_metrics import calc_metric, SUPPORTED_METRICS
from framework.node.node_evaluator import LayerResult, DesignResult, NodeResult, NodeEvaluator

class GenericNode(NodeEvaluator):
    def __init__(self, config):
        self.config = config

    def set_workdir(self, work_dir: str, runname: str, id: int):
        return super().set_workdir(work_dir, runname, id)

    def run(self, layers: list):
        raise NotImplementedError