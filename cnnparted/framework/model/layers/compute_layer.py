from .abstract_layer import LayerBase, SchedulingInfo
from framework.model.layer_parameters import ComputeParameters
from framework.model.tiling.iterators import ConcreteLoopIterator, LoopDim
from framework.model.tensor_wrapper import TensorWrapper


class LoopNest:
    def __init__(self, loop_dims: list[ConcreteLoopIterator]):
        self.loop_order = [ldim for ldim in loop_dims]
        self.axis = {ldim.iterator: ldim.extent for ldim in loop_dims}
        self._map = {ldim.iterator: [ldim] for ldim in loop_dims}
        #self.variables = [ldim.iterator.name.lower() for ldim in loop_dims]
    
    def tile(self, dim: LoopDim, n_outer_tiles: int):
        assert len(self._map[dim]) == 1, "Cannot tile a loop that was already tiled"

        loop_idx = self.loop_order.index(dim)
        loop_to_tile = self._map[dim]
        new_outer, new_inner = loop_to_tile.tile(factors=[n_outer_tiles, None])

        #Delete entries referencing the old index, add newly constructed loops
        del self.loop_order[loop_idx]
        self.loop_order = [new_outer, new_inner] + self.loop_order
        self._map[dim] = [new_outer, new_inner]

    
    def reorder(self, new_order: list[ConcreteLoopIterator]):
        assert len(new_order) == len(self.loop_order), "New loop order must be fully specified"


class ComputationLayer(LayerBase):
    layer_params: ComputeParameters
    scheduling: SchedulingInfo

    def __init__(self, 
                 layer_name: str, 
                 layer_params: ComputeParameters,
                 op_type: str,
                 inputs: list[TensorWrapper], 
                 output: TensorWrapper,
                 loop_dims: list[ConcreteLoopIterator]):
        super().__init__(
            layer_name=layer_name,
            layer_params=layer_params,
            op_type=op_type,
            inputs=inputs,
            output=output
        )
        self.loop_dims = loop_dims
        self.scheduling = SchedulingInfo()

    def __repr__(self):
        return f"layers.{self.__class__.__name__}(layer_name={self.layer_name}), op_type={self.op_type}, params={self.layer_params}"
    
    @property
    def layer_iterators(self):
        return [ldim.iterator for ldim in self.loop_dims]

    def has_loop_dim(self, dim: LoopDim):
        return dim in self.layer_iterators
    
    def get_loop_iterator(self, dim: LoopDim):
        #TODO This assumes uniqueness
        for ldim in self.loop_dims:
            if ldim.iterator == dim:
                return ldim
        raise RuntimeError("Could not find the desired loop dimension")
    
    def get_dependent_tensors(self, dim: LoopDim) -> tuple[list[TensorWrapper], TensorWrapper]:
        """Find all input and output tensors that have dim as an axis"""
        dependent_ins = []
        dependent_out = None
        for idx, input in enumerate(self.inputs):
            if dim in input:
                dependent_ins.append(input)
        if dim in self.output:
            dependent_out = self.output
        return dependent_ins, dependent_out

    def tile_dim(self, dim: LoopDim, n_outer_tiles:  int):
        layer_dim = list(filter(lambda d: d.iterator==dim, self.loop_dims))[0]
        dim_idx = self.loop_dims.index(layer_dim)
        outer_dim, inner_dim = layer_dim.tile(factors=[n_outer_tiles, None])
        del self.loop_dims[dim_idx]
        self.loop_dims = self.loop_dims + [outer_dim, inner_dim] 
        return ((outer_dim, inner_dim), dim_idx)

# Specialization Classes
class ConvLayer(ComputationLayer): 
    ...

class GemmLayer(ComputationLayer):
    ...