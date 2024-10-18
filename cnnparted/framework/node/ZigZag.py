from framework.constants import ROOT_DIR
from framework.helpers.Visualizer import plotMetricPerConfigPerLayer
from framework.helpers.DesignMetrics import calc_metric, SUPPORTED_METRICS
from framework.dse.ArchitectureMutator import ArchitectureMutator
from framework.node.NodeEvaluator import LayerResult, DesignResult, NodeResult, NodeEvaluator

class Zigzag(NodeEvaluator):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.fname_result = "zigzag_layers.csv"

    def set_workdir(self, work_dir: str, runname: str, id: int):
        return super().set_workdir(work_dir, runname, id)

    def run(self, layers: list):
        node_result = NodeResult()
        design_result = DesignResult()
        for layer in layers:
            layer_name = layer.get("name")
            stats = self.run_layer()

            layer_result = LayerResult()
            layer_result.name = layer_name
            layer_result.latency = stats["latency"]
            layer_result.energy = stats["energy"]
            layer_result.area = stats["area"]

            design_result.add_layer(layer_result)
        
        node_result.add_design(design_result)

    def _run_layer(self):
        ...

    def _get_accelerator_config(self):
        ...

    def _get_workload(self):
        ...

