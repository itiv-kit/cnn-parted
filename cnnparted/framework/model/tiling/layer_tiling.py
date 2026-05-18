import copy
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
from loguru import logger

from framework.model.layer_parameters import ConvParameters, GemmParameters
from framework.model.tensor_wrapper import TensorWrapper, OperandKind
from framework.model.layers.compute_layer import ComputationLayer
from framework.model.onnx_model import OnnxModel
from framework.model.tiling.iterators import LoopDim
from framework.helpers.config_helper import ConfigHelper


@dataclass
class TilingContext:
    """Context for a single tensor tiling operation"""
    dim: LoopDim
    n_outer_tiles: int
    dim_value: int
    params: GemmParameters | ConvParameters


@dataclass
class ConvolutionTilingContext(TilingContext):
    """Extended context for convolution tiling with receptive field information"""
    kernel_size: int = 0      # Kernel size for the tiled dimension (R or S)
    stride: int = 1           # Stride for the tiled dimension
    padding: int = 0          # Padding for the tiled dimension
    dilation: int = 1         # Dilation for the tiled dimension
    
    @property
    def effective_kernel_size(self) -> int:
        """Compute effective kernel size accounting for dilation"""
        return self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
    
    def compute_input_extent_from_output(self, output_extent: int, tile_idx: int) -> tuple[int, int]:
        """
        Compute input spatial extent needed to produce a tiled output.
        Returns (input_start, input_size) accounting for receptive field, dilation, and overlaps.
        
        For output tile covering [out_start, out_end), input covers:
            in_start = out_start * stride - padding
            in_size = (out_end - out_start) * stride + effective_kernel - stride
        """
        out_start = tile_idx * output_extent
        out_end = (tile_idx + 1) * output_extent
        
        # Receptive field start
        in_start = out_start * self.stride - self.padding
        # Receptive field size (accounts for overlap with next tile and dilation)
        eff_k = self.effective_kernel_size
        in_size = (out_end - out_start) * self.stride + (eff_k - self.stride)
        
        return in_start, in_size
    
    def compute_output_extent_from_input(self, input_extent: int) -> int:
        """Inverse: compute output extent that can be produced from input extent accounting for dilation"""
        # out = (in + 2*padding - effective_kernel) / stride + 1
        eff_k = self.effective_kernel_size
        return max(1, (input_extent - eff_k) // self.stride + 1)
    
    def compute_overlap_with_next_tile(self) -> int:
        """
        Compute how much input data overlaps between consecutive output tiles.
        This informs data reuse and cache efficiency.
        """
        eff_k = self.effective_kernel_size
        overlap = eff_k - self.stride
        return overlap


class TensorTilingStrategy(ABC):
    """Base class for tensor-specific tiling strategies"""
    
    @abstractmethod
    def can_tile(self, tensor: TensorWrapper, dim: LoopDim) -> bool:
        """Check if this strategy can tile the given tensor for this dimension"""
        pass
    
    @abstractmethod
    def compute_tiled_extent(self, tensor: TensorWrapper, context: TilingContext) -> int:
        """Compute the extent after tiling, accounting for any operation-specific adjustments"""
        pass


class ConvInputTilingStrategy(TensorTilingStrategy):
    """Strategy for tiling INPUT tensors in convolution operations
    
    Accounts for receptive field requirements when tiling spatial dimensions.
    """
    
    def can_tile(self, tensor: TensorWrapper, dim: LoopDim) -> bool:
        return dim in [LoopDim.N, LoopDim.M, LoopDim.P, LoopDim.Q]
    
    def compute_tiled_extent(self, tensor: TensorWrapper, context: TilingContext) -> int:
        iterator = tensor.get_loop_dim(context.dim)
        base_extent = iterator.extent // context.n_outer_tiles
        
        # Apply stride adjustments for spatial dimensions
        if context.dim == LoopDim.P:
            return base_extent * context.params.wstride
        elif context.dim == LoopDim.Q:
            return base_extent * context.params.hstride
        else:
            return base_extent
    
    def compute_tiled_extent_with_padding(self, tensor: TensorWrapper, context: "ConvolutionTilingContext") -> int:
        """
        Compute input extent accounting for kernel receptive field and dilation.
        This is called when tiling spatial dimensions (P or Q).
        """
        if not isinstance(context, ConvolutionTilingContext):
            # Fallback to simple tiling if no conv context provided
            return self.compute_tiled_extent(tensor, context)
        
        # For spatial dims, account for receptive field overlap and dilation
        if context.dim in [LoopDim.P, LoopDim.Q]:
            iterator = tensor.get_loop_dim(context.dim)
            
            # Compute required input size for the tiled output (accounts for dilation)
            out_extent = iterator.extent // context.n_outer_tiles
            in_start, in_size = context.compute_input_extent_from_output(out_extent, context.dim_value)
            
            # Clamp to tensor bounds
            in_size = min(in_size, iterator.extent - max(0, in_start))
            
            # Apply stride correction if needed
            if context.dim == LoopDim.P:
                return in_size * context.params.wstride
            else:
                return in_size * context.params.hstride
        
        # For non-spatial dims, use default tiling
        return self.compute_tiled_extent(tensor, context)


class ConvWeightTilingStrategy(TensorTilingStrategy):
    """Strategy for tiling WEIGHT tensors in convolution operations"""
    
    def can_tile(self, tensor: TensorWrapper, dim: LoopDim) -> bool:
        return dim == LoopDim.M
    
    def compute_tiled_extent(self, tensor: TensorWrapper, context: TilingContext) -> int:
        iterator = tensor.get_loop_dim(context.dim)
        return iterator.extent // context.n_outer_tiles


class ConvOutputTilingStrategy(TensorTilingStrategy):
    """Strategy for tiling OUTPUT tensors in convolution operations"""
    
    def can_tile(self, tensor: TensorWrapper, dim: LoopDim) -> bool:
        return True  # Can tile any dimension
    
    def compute_tiled_extent(self, tensor: TensorWrapper, context: TilingContext) -> int:
        iterator = tensor.get_loop_dim(context.dim)
        return iterator.extent // context.n_outer_tiles


class ConvBiasTilingStrategy(TensorTilingStrategy):
    """Strategy for tiling BIAS tensors in convolution operations (no-op)"""
    
    def can_tile(self, tensor: TensorWrapper, dim: LoopDim) -> bool:
        return False  # Bias is not tiled
    
    def compute_tiled_extent(self, tensor: TensorWrapper, context: TilingContext) -> int:
        # This should never be called since can_tile() returns False
        raise NotImplementedError("BIAS tensors cannot be tiled")


class GemmTilingStrategy(TensorTilingStrategy):
    """Unified strategy for all GEMM operands (uniform tiling)"""
    
    def can_tile(self, tensor: TensorWrapper, dim: LoopDim) -> bool:
        return True
    
    def compute_tiled_extent(self, tensor: TensorWrapper, context: TilingContext) -> int:
        iterator = tensor.get_loop_dim(context.dim)
        return iterator.extent // context.n_outer_tiles


class TensorTiler:
    """Handles tensor tiling with pluggable strategies"""
    
    def __init__(self):
        # Map (operation_type, operand_kind) -> strategy
        self.conv_strategies: Dict[OperandKind, TensorTilingStrategy] = {
            OperandKind.INPUT: ConvInputTilingStrategy(),
            OperandKind.WEIGHT: ConvWeightTilingStrategy(),
            OperandKind.OUTPUT: ConvOutputTilingStrategy(),
            OperandKind.BIAS: ConvBiasTilingStrategy(),
        }
        self.gemm_strategy = GemmTilingStrategy()
    
    def _get_strategy(self, tensor: TensorWrapper, params: GemmParameters | ConvParameters) -> Optional[TensorTilingStrategy]:
        """Select the appropriate tiling strategy based on operation and operand type"""
        if isinstance(params, ConvParameters):
            return self.conv_strategies.get(tensor.operand)
        elif isinstance(params, GemmParameters):
            return self.gemm_strategy
        return None
    
    def _create_conv_tiling_context(self, base_context: TilingContext, conv_params: ConvParameters, 
                                     tensor: TensorWrapper) -> ConvolutionTilingContext:
        """Create a ConvolutionTilingContext with kernel/stride/padding/dilation info for spatial dimensions"""
        kernel_size = 0
        stride = 1
        padding = 0
        dilation = 1
        
        # Extract kernel (r/s), stride, padding (wpad/hpad), and dilation for the tiling dimension
        # Note: ConvParameters uses r=kernel_h, s=kernel_w
        if base_context.dim == LoopDim.P:
            kernel_size = conv_params.s  # s = kernel_w
            stride = conv_params.wstride
            padding = conv_params.wpad
            dilation = conv_params.wdilation
        elif base_context.dim == LoopDim.Q:
            kernel_size = conv_params.r  # r = kernel_h
            stride = conv_params.hstride
            padding = conv_params.hpad
            dilation = conv_params.hdilation
        
        return ConvolutionTilingContext(
            dim=base_context.dim,
            n_outer_tiles=base_context.n_outer_tiles,
            dim_value=base_context.dim_value,
            params=base_context.params,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
    
    def tile_tensor(self, tensor: TensorWrapper, context: TilingContext) -> bool:
        """
        Tile a single tensor according to the context.
        Returns True if tiling was applied, False otherwise.
        """
        # Skip if dimension not present in tensor
        if context.dim not in tensor:
            return False
        
        # Get the appropriate strategy
        strategy = self._get_strategy(tensor, context.params)
        if strategy is None:
            raise RuntimeError(f"No tiling strategy for {type(context.params)}")
        
        # Check if this strategy can tile this tensor
        if not strategy.can_tile(tensor, context.dim):
            return False
        
        # Get loop dimension and verify divisibility
        iterator = tensor.get_loop_dim(context.dim)
        assert (iterator.extent % context.n_outer_tiles) == 0, \
            f"Extent {iterator.extent} not divisible by n_outer_tiles {context.n_outer_tiles}"
        
        # Compute and apply tiling
        # For conv input on spatial dims, use receptive-field aware tiling
        if (isinstance(context.params, ConvParameters) and 
            tensor.operand == OperandKind.INPUT and 
            context.dim in [LoopDim.P, LoopDim.Q] and
            isinstance(strategy, ConvInputTilingStrategy)):
            
            conv_context = self._create_conv_tiling_context(context, context.params, tensor)
            tiled_extent = strategy.compute_tiled_extent_with_padding(tensor, conv_context)
        else:
            tiled_extent = strategy.compute_tiled_extent(tensor, context)
        
        iterator.lower_bound = tiled_extent * context.dim_value
        iterator.upper_bound = (tiled_extent * (context.dim_value + 1)) - 1
        iterator.extent = tiled_extent
        
        return True
    
    def tile_tensors(self, tensors: list[TensorWrapper], dim: LoopDim, n_outer_tiles: int,
                     dim_value: int, params: GemmParameters | ConvParameters) -> list[TensorWrapper]:
        """Tile multiple tensors with the same context"""
        context = TilingContext(dim, n_outer_tiles, dim_value, params)
        
        for tensor in tensors:
            self.tile_tensor(tensor, context)
        
        return tensors


class LayerTiling:
    def __init__(self, config: ConfigHelper):
        # Tiling will insert new compute nodes into the graph and remove old ones with smaller size

        # Map network to num of tiles, these are "outer" tiles
        self.tiling = {wl.name: wl.tiling for wl in config.workload.workload_spec}
        self.tensor_tiler = TensorTiler()
        
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
    
    def _determine_num_tiles(self, layer: ComputationLayer, dim: LoopDim, num_tiles: int):
        # Try to create a smaller number of tiles, if the dimension is not evenly
        # tilable by the given factor
        if num_tiles is None:
            return num_tiles
        while num_tiles >= 1:
            ldim_extent = layer.get_loop_iterator(dim).extent

            if ldim_extent % num_tiles == 0 :
                # valid tiling: extent evenly divisible by num_tiles
                return num_tiles
            else:
                num_tiles -= 1

        return num_tiles
    
    def tile_layers_inplace(self, models: dict[str, OnnxModel]):
        for network, model in models.items():
            tiling_data = self.tiling[network]
            layers_to_remove = []
            layers_to_add: list[tuple[ComputationLayer, list[ComputationLayer], list[ComputationLayer]]] = []
            logger.debug(f"Begin tiling for network {network}")

            # map original layer to all generated tiles, needed to correctly map
            # data dependencies after tiling has been performed
            generated_tiles_of_layer: dict[ComputationLayer, list[ComputationLayer]] = {}

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
                    dim, num_tiles_cfg = tiling_data_op_type.tiling.popitem()
                    generated_tiles = []
                    logger.debug(f"Tiling along dimension \"{dim.name}\" with num_tiles={num_tiles_cfg}")

                    # The inner while loops iterates over all currently existing tiles and applies
                    # further tiling. In essence, this models a recursive tiling where each layer is
                    # successively tiled, one dimension after the other. The order is irrelevant here.
                    while len(tiles_to_process) > 0:
                        current_tile = tiles_to_process.pop(0)
                        
                        # Hack to get FCNResNet working with Simba
                        #if network == "fcnresnet50" and current_tile.has_loop_dim(dim) and current_tile.get_loop_iterator(dim).extent == 21:
                        #    num_tiles_cfg = 8
                        
                        num_tiles = self._determine_num_tiles(current_tile, dim, num_tiles_cfg)
                        
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
                    generated_tiles_of_layer[layer] = final_generated_tiles
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
            # We do this in two passes to correctly connect generated tiles
            for new_layer, preds, succs in layers_to_add:
                model.add_layer(new_layer)

            for new_layer, preds, succs in layers_to_add:
                for pred in preds:
                    if pred in generated_tiles_of_layer:
                        # The predecessor has been tiled and will be removed from the network.
                        # Instead of connecting the predecessor, connect with all generated tiles
                        for tiled_pred in generated_tiles_of_layer[pred]:
                            model.add_edge(tiled_pred, new_layer)
                    else:
                        model.add_edge(pred, new_layer)
                for succ in succs:
                    if succ in generated_tiles_of_layer:
                        # The successor has been tiled and will be removed from the network
                        # Instead of connecting the successor, connect with all generated tiles
                        for tiled_succ in generated_tiles_of_layer[succ]:
                            model.add_edge(new_layer, tiled_succ)
                    else:
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
    
    def _adjust_padding_for_tile(self, params: ConvParameters, dim: LoopDim, tile_idx: int, 
                                  n_outer_tiles: int) -> ConvParameters:
        """
        Adjust padding parameters for tiled convolution layers.
        
        When tiling spatial dimensions, only edge tiles need padding:
        - First tile (idx=0): Keep left-side padding
        - Middle tiles: No padding (fully contained within receptive field)
        - Last tile (idx=n_outer_tiles-1): Keep right-side padding
        """
        if dim == LoopDim.P:
            padding_attr = 'wpad'
        elif dim == LoopDim.Q:
            padding_attr = 'hpad'
        else:
            return params
        
        original_padding = getattr(params, padding_attr, 0)
        
        # Middle tiles don't need padding (receptive field is fully covered by input overlap)
        if tile_idx > 0 and tile_idx < n_outer_tiles - 1:
            setattr(params, padding_attr, 0)
        # First tile keeps original padding on left side
        elif tile_idx == 0:
            setattr(params, padding_attr, original_padding)
        # Last tile might need right-side padding (keep original for conservative approach)
        else:
            setattr(params, padding_attr, original_padding)
        
        return params

    
    def generate_tiles(self, layer: ComputationLayer, dim: LoopDim, n_outer_tiles: int) -> list[ComputationLayer]:
        generated_tiles: list[ComputationLayer] = []
        new_inputs = self._find_inputs_with_dim(layer, dim)

        for num in range(n_outer_tiles):
            tiled_inputs = self.tile_tensors(copy.deepcopy(new_inputs), dim, n_outer_tiles, num, layer.layer_params)
            tiled_output = self.tile_tensors([copy.deepcopy(layer.output)], dim, n_outer_tiles, num, layer.layer_params)[0]

            # Modify the layer parameter to reflect the tiling of the operation
            # I.e., we tile the loop with range N into two loop with ranges (outer, inner) respectively
            loop_var = dim.name.lower()
            new_params = copy.deepcopy(layer.layer_params)
            param_to_tile = getattr(new_params, loop_var)
            param_to_tile = param_to_tile // n_outer_tiles
            assert hasattr(new_params, loop_var), f"Did not find {loop_var} as a parameter of {type(new_params)}"
            setattr(new_params, loop_var, param_to_tile)
            
            # Adjust padding for tiled spatial dimensions in convolution layers
            if isinstance(new_params, ConvParameters) and dim in [LoopDim.P, LoopDim.Q]:
                new_params = self._adjust_padding_for_tile(new_params, dim, num, n_outer_tiles)

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
            tiled_layer.id = layer.id #ID must be the same for all tiles
            tiled_layer.num_tiles = layer.num_tiles * n_outer_tiles
            logger.debug(f"Generated tile {tiled_layer.layer_name}")

            generated_tiles.append(tiled_layer)
        
        return generated_tiles
    
    def tile_tensors(self, tensors: list[TensorWrapper], dim: LoopDim, n_outer_tiles: int,
                     dim_value: int, params: GemmParameters | ConvParameters) -> list[TensorWrapper]:
        """Tile multiple tensors using the unified TensorTiler strategy"""
        return self.tensor_tiler.tile_tensors(tensors, dim, n_outer_tiles, dim_value, params)
