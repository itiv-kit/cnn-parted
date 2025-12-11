import copy
import numpy as np

from framework.model.tensor_wrapper import TensorWrapper
from framework.model.layers.compute_layer import ComputationLayer
from framework.model.onnx_model import OnnxModel
from framework.model.tiling.iterators import LoopIterator, ConcreteLoopIterator, LoopDim
from framework.helpers.config_helper import ConfigHelper
from framework.helpers.config.general_config import TilingConfig

class LayerTiling:
    def __init__(self, config: ConfigHelper):
        # Tiling will insert new compute nodes into the graph and remove old ones with smaller size
        # Or maybe it should just update the layer information? And make a note about tiling?

        # Map network to num of tiles, these are "outer" tiles
        self.tiling = {wl.name: wl.tiling for wl in config.workload.workload_spec}

    
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

            for layer in model.nodes:
                if not isinstance(layer, ComputationLayer):
                    continue

                tiling_data_op_type = copy.deepcopy(tiling_data[layer.op_type])
                layer_was_tiled = False

                tiles_to_process = [layer]
                
                # Tile the current layer across all specified dimensions
                # In the outer while-loop iterate over all dimensions that should be tiled
                while len(tiling_data_op_type.tiling) > 0:
                    dim, num_tiles = tiling_data_op_type.tiling.popitem()
                    generated_tiles = []

                    # The inner while loops iterates over all currently existing tiles and applies
                    # further tiling. In essence, this models a recursive tiling where each layer is
                    # successively tiled, one dimension after the other. The order is irrelevant here.
                    while len(tiles_to_process) > 0:
                        current_tile = tiles_to_process.pop(0)
                        if isinstance(dim, list):
                            breakpoint()
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
                new_inputs.append(copy.deepcopy(inp))
                continue
            
            tiled_inp = copy.deepcopy(inp)
            tiled_inp.sub_id = 999 #TODO
            new_inputs.append(tiled_inp)
            #TODO Perform tiling
        return new_inputs

    
    def _tile_layers_inplace_old(self, models: dict[str, OnnxModel]):
        for network, model in models.items():
            tiling_data = self.tiling[network]
            layers_to_remove = []
            layers_to_add: list[tuple[ComputationLayer, list[ComputationLayer], list[ComputationLayer]]] = []

            for layer in model.nodes:
                if not isinstance(layer, ComputationLayer):
                    continue

                tiling_data_op_type = tiling_data[layer.op_type]
                
                for dim, num_tiles in tiling_data_op_type:
                    if not (layer.has_loop_dim(dim)): 
                        continue
                    if (num_tiles is None) or (num_tiles == 1):
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

            # The newly generated tiles have been inserted into the graph, now we delete
            # the untiled version of the layer
            for rm_layer in layers_to_remove:
                model.remove_layer(rm_layer)
        
    def _tile_layer_recursive(self, tiling_cfg_optype: TilingConfig, input_tiles: list[ComputationLayer], ) -> list[ComputationLayer]:
        if not tiling_cfg_optype.tiling:
            # Check if any dimensions remain that should be tiled
            return input_tiles
        else:
            output_tiles = []

            dim_to_tile, n_outer_tiles = tiling_cfg_optype.tiling.popitem()

            new_inputs = []
            num = 999 #TODO 

            #Tile every layer
            for layer in input_tiles:
                if not (layer.has_loop_dim(dim_to_tile)): 
                    output_tiles.append(layer)
                    continue
                if (n_outer_tiles is None) or (n_outer_tiles == 1):
                    output_tiles.append(layer)
                    continue
                if not (layer.get_loop_iterator(dim_to_tile).extent % n_outer_tiles == 0):
                    output_tiles.append(layer)
                    continue

                # Copy all inputs and check if they need to be tiled, same for the output
                new_inputs = self._find_inputs_with_dim(layer, dim_to_tile)
                
                new_output = copy.deepcopy(layer.output)
                new_output.sub_id = num

                # Modify the layer parameter to reflect the tiling of the operation
                # I.e., we tile the loop with range N into two loop with ranges (outer, inner) respectively
                loop_var = dim_to_tile.name.lower()
                new_params = copy.deepcopy(layer.layer_params)
                param_to_tile = getattr(new_params, loop_var)
                param_to_tile = param_to_tile // n_outer_tiles
                setattr(new_params, loop_var, param_to_tile) #TODO This requires params and loopdims to be an exact match

                for n in range(n_outer_tiles):
                    tiled_layer = ComputationLayer(
                        layer_name=layer.layer_name + f"tile_{dim_to_tile.name}_{num}",
                        layer_params=new_params,
                        op_type=layer.op_type,
                        inputs=new_inputs,
                        output=new_output,
                        loop_dims=copy.deepcopy(layer.loop_dims), #TODO
                    )

                    # Modify loop_dims to reflect tiling
                    ldim = tiled_layer.get_loop_iterator(dim_to_tile)
                    ldim_idx = tiled_layer.loop_dims.index(ldim)
                    new_outer, new_inner = ldim.tile(factors=[n_outer_tiles, None])
                    tiled_layer.loop_dims[ldim_idx] = [new_outer, new_inner]

                    output_tiles.append(tiled_layer)

            new_output_tiles = self._tile_layer_recursive(tiling_cfg_optype, output_tiles)
            return new_output_tiles
    
    def generate_tiles(self, layer: ComputationLayer, dim: LoopDim, n_outer_tiles: int) -> list[ComputationLayer]:
        generated_tiles: list[ComputationLayer] = []
        for num in range(n_outer_tiles):

            new_inputs = self._find_inputs_with_dim(layer, dim) #[]
            
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
                layer_name=layer.layer_name + f"_tile_{dim.name}_{num}",
                layer_params=new_params,
                op_type=layer.op_type,
                inputs=new_inputs,
                output=new_output,
                loop_dims=copy.deepcopy(layer.loop_dims), #TODO
            )

            # Modify loop_dims to reflect tiling
            #ldim = tiled_layer.get_loop_iterator(dim)
            #ldim_idx = tiled_layer.loop_dims.index(ldim)
            #new_outer, new_inner = ldim.tile(factors=[n_outer_tiles, None])
            #tiled_layer.loop_dims[ldim_idx] = [new_outer, new_inner]

            generated_tiles.append(tiled_layer)
        
        return generated_tiles
    
    def tile_tensors(self, tensor: TensorWrapper, dim: LoopDim, n_outer_tiles: int) -> list[TensorWrapper]:
        ...

