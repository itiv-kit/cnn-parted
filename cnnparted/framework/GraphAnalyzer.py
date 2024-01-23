
from .model.model import TreeModel
from .model.graph import LayersGraph

from .model.scheduling import topo_sort_random_start_node

class GraphAnalyzer:
    def __init__(self, run_name : str, input_size : tuple, progress : bool) -> None:
        self.input_size = input_size
        self._Tree_Model = TreeModel(run_name, self.input_size)
        self._tree = self._Tree_Model.get_Tree()

        self.graph = LayersGraph(self._tree)
        self.schedules = topo_sort_random_start_node(G=self.graph.get_Graph(), n=1, seed=0, as_ndarray=True, n_jobs=1, progress=progress)
        self.conv_layers = self.get_conv2d_layers()

    def get_conv2d_layers(self):
        output = [layer for layer in self._tree if layer.get("op_type") == "Conv"]
        return output

    def get_gemm_layers(self):
        output = [layer for layer in self._tree if layer.get("op_type") == "Gemm"]
        return output
