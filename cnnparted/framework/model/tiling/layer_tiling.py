import copy
import numpy as np
from loguru import logger

from framework.model.tensor_wrapper import TensorWrapper
from framework.model.layers.compute_layer import ComputationLayer
from framework.model.onnx_model import OnnxModel
from framework.model.tiling.iterators import LoopDim
from framework.helpers.config_helper import ConfigHelper

class LayerTiling:
    def __init__(self, config: ConfigHelper):
        # Tiling will insert new compute nodes into the graph and remove old ones with smaller size
        # Or maybe it should just update the layer information? And make a note about tiling?

        # Map network to num of tiles, these are "outer" tiles
        self.tiling = {wl.name: wl.tiling for wl in config.workload.workload_spec}
        for name, tiling in self.tiling.items():
            logger.debug(f"Initialize layer tiling for network {name} with tiling={tiling}")

    
    def _should_skip_tile(self, layer: ComputationLayer, dim: LoopDim, num_tiles: int):
        if not (layer.has_loop_dim(dim)): 
            return True
        if (num_tiles is None) or (num_tiles == 1):
            return True
        if not (layer.get_loop_iterator(dim).extent % num_tiles == 0):
            return True
        return False

    
    def tile_layers_inplace(self, models: dict[str, OnnxModel]):
        for network, model in models.items():
            tiling_data = self.tiling[network]
            layers_to_remove = []
            layers_to_add: list[tuple[ComputationLayer, list[ComputationLayer], list[ComputationLayer]]] = []
            logger.debug(f"Begin tiling for network {network}")

            for layer in model.nodes:
                if not isinstance(layer, ComputationLayer):
                    continue

                tiling_data_op_type = copy.deepcopy(tiling_data[layer.op_type])
                layer_was_tiled = False
                tiles_to_process = [layer]
                logger.debug(f"Tiling layer {layer.layer_name}...")
                
                # Tile the current layer across all specified dimensions
                # In the outer while-loop iterate over all dimensions that should be tiled
                while len(tiling_data_op_type.tiling) > 0:
                    dim, num_tiles = tiling_data_op_type.tiling.popitem()
                    generated_tiles = []
                    logger.debug(f"Tiling along dimension \"{dim.name}\" with num_tiles={num_tiles}")

                    # The inner while loops iterates over all currently existing tiles and applies
                    # further tiling. In essence, this models a recursive tiling where each layer is
                    # successively tiled, one dimension after the other. The order is irrelevant here.
                    while len(tiles_to_process) > 0:
                        current_tile = tiles_to_process.pop(0)
                        if self._should_skip_tile(current_tile, dim, num_tiles):
                            generated_tiles.append(current_tile)
                        else: 
                            # This layer will be split into smaller tiles, so we remove the 
                            # original layer at a later point
                            layer_was_tiled = True

                            # We have a computation layer that has the corresponding dimension
                            new_tiles = self.generate_tiles(current_tile, dim, num_tiles)
                            generated_tiles.extend(new_tiles)
                    
                    tiles_to_process = copy.deepcopy(generated_tiles)

                final_generated_tiles = generated_tiles
                logger.debug(f"Generated a total of {len(final_generated_tiles)} tiles")

                # If tiling was performed for the current layer we have to remove it from
                # the graph and add the newly generated tiles to it
                if layer_was_tiled:
                    layers_to_remove.append(layer)

                    # Add all new tiles to the graph
                    for tile in final_generated_tiles:
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

            # The newly generated tiles have been inserted into the graph, now we delete
            # the untiled version of the layer
            for rm_layer in layers_to_remove:
                model.remove_layer(rm_layer)
        return models 
                    

    def _find_inputs_with_dim(self, layer: ComputationLayer, dim_to_tile: LoopDim):
        new_inputs = []
        for inp in layer.inputs:
            if dim_to_tile not in inp.loop_iterators:
                # The dimension being tiled is not part of the tensor
                new_inputs.append(copy.deepcopy(inp))
                continue
            
            tiled_inp = copy.deepcopy(inp)
            new_inputs.append(tiled_inp)
        return new_inputs

    
    def generate_tiles(self, layer: ComputationLayer, dim: LoopDim, n_outer_tiles: int) -> list[ComputationLayer]:
        generated_tiles: list[ComputationLayer] = []
        new_inputs = self._find_inputs_with_dim(layer, dim)

        for num in range(n_outer_tiles):
            tiled_inputs = self.tile_tensors(copy.deepcopy(new_inputs), dim, n_outer_tiles, num)
            tiled_output = self.tile_tensors([copy.deepcopy(layer.output)], dim, n_outer_tiles, num)

            # Modify the layer parameter to reflect the tiling of the operation
            # I.e., we tile the loop with range N into two loop with ranges (outer, inner) respectively
            loop_var = dim.name.lower()
            new_params = copy.deepcopy(layer.layer_params)
            param_to_tile = getattr(new_params, loop_var)
            param_to_tile = param_to_tile // n_outer_tiles
            setattr(new_params, loop_var, param_to_tile)

            # Apply tiling information to the loop dimension
            tiled_loop_dims = copy.deepcopy(layer.loop_dims)
            loop_dim_to_tile_idx = [dim.iterator for dim in layer.loop_dims].index(dim)
            loop_dim_to_tile = tiled_loop_dims[loop_dim_to_tile_idx]

            # TODO Unify with tensor tiling
            tiled_extent = loop_dim_to_tile.extent // n_outer_tiles
            loop_dim_to_tile.lower_bound = tiled_extent * num
            loop_dim_to_tile.upper_bound = (tiled_extent * (num+1)) - 1
            loop_dim_to_tile.extent = tiled_extent
            
            tiled_layer = ComputationLayer(
                layer_name=f"{layer.layer_name}/tile_{dim.name}_{num}",
                layer_params=new_params,
                op_type=layer.op_type,
                inputs=tiled_inputs,
                output=tiled_output,
                loop_dims=tiled_loop_dims,
            )
            logger.debug(f"Generated tile {tiled_layer.layer_name}")

            generated_tiles.append(tiled_layer)
        
        return generated_tiles
    
    def tile_tensors(self, tensors: list[TensorWrapper], dim: LoopDim, n_outer_tiles: int, dim_value: int) -> list[TensorWrapper]:
        for tensor in tensors:
            if dim not in tensor:
                continue
            loop_dim = tensor.get_loop_dim(dim)
            assert (loop_dim.extent % n_outer_tiles) == 0
            tiled_extent = loop_dim.extent // n_outer_tiles
            loop_dim.lower_bound = tiled_extent * dim_value
            loop_dim.upper_bound = (tiled_extent * (dim_value+1)) - 1
            loop_dim.extent = tiled_extent
            tensor.sub_id = tensor.sub_id*dim_value+1 #TODO
        
        return tensors

