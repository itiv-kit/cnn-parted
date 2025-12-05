import copy
import numpy as np

from framework.model.tensor_wrapper import TensorWrapper
from framework.model.layers.compute_layer import ComputationLayer
from framework.model.onnx_model import OnnxModel
from framework.model.tiling.iterators import LoopIterator, ConcreteLoopIterator, LoopDim
from framework.helpers.config_helper import ConfigHelper

class LayerTiling:
    def __init__(self, models: dict[str, OnnxModel], config: ConfigHelper):
        self.models = models
        # Tiling will insert new compute nodes into the graph and remove old ones with smaller size
        # Or maybe it should just update the layer information? And make a note about tiling?

        # Map network to num of tiles
        # These are "outer" tiles
        self.tiling = {wl.name: wl.tiling for wl in config.workload.workload_spec}
    
    def run(self):
        self.tile_layers()

    def tile_layers(self):
        for network, model in self.models.items():
            tiling_data = self.tiling[network]
            layers_to_remove = []
            layers_to_add: list[tuple[ComputationLayer, list[ComputationLayer], list[ComputationLayer]]] = []

            for layer in model.nodes:
                if not isinstance(layer, ComputationLayer):
                    continue
                
                # TODO Currently assumes that only one dimension is tiled
                for dim, num_tiles in tiling_data:
                    if not (layer.has_loop_dim(dim)) or (num_tiles is None):
                        continue

                    if not (layer.get_loop_iterator(dim).extent % num_tiles == 0):
                        continue

                    # This layer will be split into smaller tiles, so we remove the 
                    # original layer at a later point
                    layers_to_remove.append(layer)
                    
                    # We have a computation layer that has the corresponding dimension
                    new_tiles = self.generate_tiles(layer, dim, num_tiles)

                    # Add all new tiles to the graph
                    for tile in new_tiles:
                        preds = []
                        succs = []
                        for pred in model.predecessors(layer):
                            preds.append(pred)
                        for succ in model.successors(layer):
                            succs.append(succ)
                        layers_to_add.append((tile, preds, succs))            
            
            # Add new layer to the graph and connect it
            for new_layer, preds, succs in layers_to_add:
                model.add_layer(new_layer)
                for pred in preds:
                    model.add_edge(pred, new_layer)
                for succ in succs:
                    model.add_edge(new_layer, succ)

            # All layers of the current network have been tiled, remove original layer from the graph
            for rm_layer in layers_to_remove:
                model.remove_layer(rm_layer)

    
    def generate_tiles(self, layer: ComputationLayer, dim: LoopDim, n_outer_tiles: int) -> list[ComputationLayer]:
        generated_tiles: list[ComputationLayer] = []
        for num in range(n_outer_tiles):

            new_inputs = []
            for inp in layer.inputs:
                if dim not in inp.loop_iterators:
                    new_inputs.append(inp)
                    continue
                
                tiled_inp = copy.deepcopy(inp)
                tiled_inp.sub_id = num
                new_inputs.append(tiled_inp)
                #TODO We need to perform the tiling here
            
            new_output = copy.deepcopy(layer.output)
            new_output.sub_id = num

            # Modify the layer parameter to reflect the tiling of the operation
            # I.e., we tile the loop with range N into two loop with ranges (outer, inner) respectively
            loop_var = dim.name.lower()
            new_params = copy.deepcopy(layer.layer_params)
            param_to_tile = getattr(new_params, loop_var)
            param_to_tile = param_to_tile // n_outer_tiles
            setattr(new_params, loop_var, param_to_tile) #TODO This requires params and loopdims to be an exact match

            tiled_layer = ComputationLayer(
                layer_name=layer.layer_name + f"tile_{dim.name}_{num}",
                layer_params=new_params,
                op_type=layer.op_type,
                inputs=new_inputs,
                output=new_output,
                loop_dims=copy.deepcopy(layer.loop_dims), #TODO
            )
            generated_tiles.append(tiled_layer)
        
        return generated_tiles
    
    def tile_tensors(self, tensor: TensorWrapper, dim: LoopDim, n_outer_tiles: int) -> list[TensorWrapper]:
        ...

