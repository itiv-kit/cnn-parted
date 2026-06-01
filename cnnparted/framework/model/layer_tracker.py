from dataclasses import dataclass
import numpy as np

from framework.model.onnx_model import OnnxModel

@dataclass
class LayerBindings:
    """
    This class is used to keep track of all layers of all networks that are being evaluated. 
    For each layer, it keeps the name of the layer and the binding, i.e. the assignment to a platform.
    Detailed layer stats, like energy or latency, must be retrieved from the SystemResult class.
    """
    network_info: dict[str, dict[str, int]]

    def __iter__(self):
        return iter([ (nw,layer_name,binding) for nw, layers in self.network_info.items() for layer_name, binding in layers.items()])
    
    def get_binding(self, network: str, layer_name: str) -> int:
        if layer_name in self.network_info[network]:
            return self.network_info[network][layer_name]
        raise KeyError(f"Tried to get layer {layer_name} of {network} which does not exist")

    def get_all_layers_on_platform(self, platform: int) -> list[tuple[str, str]]:
        nw_layer = [ (nw,layer_name) for nw, layers in self.network_info.items() for layer_name, binding in layers.items() if binding == platform]
        return nw_layer

    def get_all_layer_names(self) -> list[tuple[str, str]]:
        nw_layer = [ (nw,layer_name) for nw, layers in self.network_info.items() for layer_name, binding in layers.items()]
        return nw_layer

    def get_num_layers(self) -> int:
        layer_ctr = [ len(layers) for layers in self.network_info.values() ]
        return sum(layer_ctr)
    
    def __len__(self):
        layer_ctr = [ len(layers) for layers in self.network_info.values() ]
        return sum(layer_ctr)
        
    def set_all_bindings(self, new_bindings: list[int]):
        """Takes a list[int] where each entry denotes the platform on which a layer should be executed"""
        idx = 0
        for nw, layers in self.network_info.items():
            for layer in layers:
                self.network_info[nw][layer] = new_bindings[idx]
                idx += 1
        return self

    def swap_layers(self, nws_layers: list[tuple[str,str]]):
        """Swap the binding of two layers that are on different platforms"""
        bindings = np.array([self.network_info[nw][layer] for nw, layer in nws_layers])
        (first_nw, first_layer), (second_nw, second_layer) = nws_layers
        first_b, second_b = bindings
        self.network_info[first_nw][first_layer] = second_b
        self.network_info[second_nw][second_layer] = first_b

    def move_layer(self, nw_layer: tuple[str,str], to: int):
        """Move one layer to a specified platform"""
        nw, layer = nw_layer
        self.network_info[nw][layer] = to

    def swap_platforms(self, platform_a: int, platform_b: int):
        """Swap all layers assigned to two differnt platforms"""
        for nw, layer_bindings in self.network_info.items():
            for layer_name, binding in layer_bindings.items():
                if binding == platform_a:
                    self.network_info[nw][layer_name] = platform_b
                elif binding == platform_b:
                    self.network_info[nw][layer_name] = platform_a