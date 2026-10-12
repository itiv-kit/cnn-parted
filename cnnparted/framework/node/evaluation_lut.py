import copy

from framework.node.node_evaluator import LayerResult
from framework.model.layer_parameters import ComputeParameters

class EvaluationLut:
    def __init__(self):
        ## TODO: Needs support for DSE
        self.lut: dict[str, dict[ComputeParameters, LayerResult]] = {}
        self._seen_params:dict[str, list[ComputeParameters]] = {}
    
    def add_entry(self,  layer_params: ComputeParameters, 
                  acc_name: str, entry: LayerResult) -> None:
        if acc_name not in self.lut:
            self.lut[acc_name] = {}
            self._seen_params[acc_name] = []

        self.lut[acc_name][layer_params] = entry
        self._seen_params[acc_name].append(layer_params)
    
    def try_find_entry(self, layer_params: ComputeParameters, acc_name: str) -> LayerResult | None:
        if acc_name not in self.lut:
            return None
        if layer_params not in self._seen_params[acc_name]:
            return None

        # Return deepcoy to enable changing of layer name
        entry = self.lut[acc_name].get(layer_params, None)
        return copy.deepcopy(entry)
