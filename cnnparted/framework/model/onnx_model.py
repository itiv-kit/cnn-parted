import math
import os
from typing import Any, TYPE_CHECKING

import onnx
from onnx import shape_inference, ModelProto, NodeProto, TypeProto, TensorProto  # noqa: F401
#from onnx2pytorch import ConvertModel
from onnx2torch import convert
import networkx as nx
from loguru import logger

from framework.helpers.dtypes import WorkloadInfo
from framework.model.tensor_wrapper import TensorWrapper
from framework.constants import MODEL_PATH
from framework.model.onnx_tensor import OnnxTensor#, OnnxTensorKind
from framework.model.tensor_like import TensorKind
from framework.model.layers.abstract_layer import LayerBase
from framework.model.layers.rename_layer import InputLayer, OutputLayer, RenameLayer
from framework.model.layers.compute_layer import ComputationLayer 

from framework.model.parser.onnx_layer import OnnxLayerParser
from framework.model.parser.conv import ConvParser
from framework.model.parser.conv_transpose import ConvTransposeParser
from framework.model.parser.elementwise import ElementwiseParser
from framework.model.parser.gemm import GemmParser
from framework.model.parser.pooling import PoolingParser
from framework.model.parser.activation_func import ActivationFuncParser
from framework.model.parser.transpose import TransposeParser
from framework.model.parser.layer_normalization import LayerNormalizationParser
from framework.model.parser.concat import ConcatParser
from framework.model.parser.clip import ClipParser
from framework.model.parser.shape import ShapeParser
from framework.model.parser.slice import SliceParser
from framework.model.parser.resize import ResizeParser


class OnnxModel(nx.DiGraph[ComputationLayer | RenameLayer] if TYPE_CHECKING else nx.DiGraph):
    """Result after parsing the onnx model into an internal representation"""
    def __init__(self, onnx_in_model: ModelProto, **attr: Any):
        super().__init__(**attr)
        self._current_id = 0
        self.onnx_in_model = onnx_in_model
        self.name_to_node_mapping: dict[str, LayerBase] = {}
        self.deadline_mul: float = None
        self.deadline_div: float = None
        self.frames: int = None
        self.deadlines: list[float] = None
        self.has_frame_constraint = False

    def set_runtime_constraints(self, wl_info: WorkloadInfo):
        self.deadline_mul = wl_info.deadline_mul
        self.deadline_div = wl_info.deadline_div

        if wl_info.deadline != math.inf:
            self.has_frame_constraint = False
            self.deadlines = [wl_info.deadline]
            self.frames = 1
        elif wl_info.frames is not None:
            self.has_frame_constraint = True
            self.frames = wl_info.frames
            time_per_frame = 1000 / wl_info.frames
            self.deadlines = [i*time_per_frame for i in range(1, wl_info.frames+1)]
            print(f"Model deadlines: {self.deadlines}")
        else:
            self.has_frame_constraint = False
            self.frames = 1
            self.deadlines = [math.inf]
    
    @property
    def has_auto_deadline(self) -> bool:
        return self.deadlines[0] == math.inf and len(self.deadlines) == 1 and  \
                (self.deadline_div is not None and self.deadline_mul is not None)
    
    @property
    def has_constraints(self) -> bool:
        return any([deadline != math.inf for deadline in self.deadlines]) or (self._has_frame_constraint)
    
    def add_layer(self, layer: RenameLayer | ComputationLayer):
        layer.id = self._current_id
        self._current_id += 1
        self.name_to_node_mapping[layer.layer_name] = layer
        self.add_node(layer)
    
    def remove_layer(self, layer: RenameLayer | ComputationLayer | str):
        if isinstance(layer, (LayerBase)):
            del self.name_to_node_mapping[layer.layer_name]
            self.remove_node(layer)
        elif isinstance(layer, str):
            layer_node = self.name_to_node_mapping[layer]
            del self.name_to_node_mapping[layer]
            self.remove_node(layer_node)
        else:
            raise RuntimeError(f"Invalid call to remove_layer, called on {type(layer)}")
        
    def _find_consumers(self, tensor: OnnxTensor | TensorWrapper, layer_inputs: dict[str, list[OnnxTensor]]) -> list[LayerBase]:
        consumers = []
        if isinstance(tensor, TensorWrapper):
            tensor = tensor._onnx_tensor
        if tensor is None:
            return consumers
        for layer_name, layer_input_tensors in layer_inputs.items():
            if tensor in layer_input_tensors:
                consumers.append(self.name_to_node_mapping[layer_name])
        return consumers
    
    def construct_layer_relations(self, layer_outputs: dict[str, list[OnnxTensor]], 
                                  layer_inputs: dict[str, list[OnnxTensor]], network: str):
        for node in self.nodes:
            consumers = self._find_consumers(node.output, layer_inputs)
            for consumer in consumers:
                self.add_edge(node, consumer)
            
    def successors_by_layer_name(self, layer: str) -> LayerBase:
        node = self.name_to_node_mapping[layer]
        return self.successors(node)

    def predecessors_by_layer_name(self, layer: str) -> LayerBase:
        node = self.name_to_node_mapping[layer]
        return self.predecessors(node)
    
    def node_by_layer_name(self, layer: str) -> LayerBase:
        return self.name_to_node_mapping[layer]
    
    def get_initial_layers(self):
        # Fetch all layers that can be scheduled directly at the start, 
        # i.e. those that only depend on model inputs and/or static data
        initial_layers = []
        for node in self.nodes:
            if isinstance(node, (InputLayer, OutputLayer)):
                continue
            tensor_initially_available = [inp.is_initially_available for inp in node.inputs]
            if all(tensor_initially_available):
                initial_layers.append(node)
        return initial_layers
    
    def get_input_layers(self) -> list[InputLayer]:
        input_layers = []
        for node in self.nodes:
            if isinstance(node, InputLayer):
                input_layers.append(node)
        return input_layers
    
    def get_compute_layers(self) -> list[ComputationLayer]:
        compute_layers = []
        for node in self.nodes:
            if not isinstance(node, ComputationLayer):
                continue
            compute_layers.append(node)
        return compute_layers
    
    def get_ready_layers(self, scheduled_layers: list[LayerBase]) -> list[LayerBase]:
        rdy: list[LayerBase] = []
        for node in self.nodes:
            # A layer can be scheduled if all predcessors are scheduled
            is_pred_scheduled = [(pred in scheduled_layers) 
                                 for pred in self.predecessors(node)]
            if all(is_pred_scheduled):
                rdy.append(node)
        return rdy
    
    def computation_successors(self, layer: LayerBase) -> list[ComputationLayer]:
        """Depth first search of all computation nodes that are immediate successors to a given layer"""
        found_computation_succs = []
        for succ in self.successors(layer):
            if isinstance(succ, ComputationLayer):
                found_computation_succs.append(succ)
            else:
                found_computation_succs = found_computation_succs + self.computation_successors(succ)
        return found_computation_succs


class OnnxParser: 
    OP_PARSER: dict = {
        "Conv": ConvParser,
        "MatMul": GemmParser,
        "Gemm": GemmParser,
        "ConvTranspose": ConvTransposeParser,
        # Elementwise operations
        "Mul": ElementwiseParser,
        "Div": ElementwiseParser,
        "Add": ElementwiseParser,
        # Pooling Operations
        "MaxPool": PoolingParser,
        "AveragePool": PoolingParser,
        "GlobalMaxPool": PoolingParser,
        "GlobalAveragePool": PoolingParser,
        # Activation Functions
        "Relu": ActivationFuncParser,
        "Sigmoid": ActivationFuncParser,
        "Erf": ActivationFuncParser,
        "HardSwish": ActivationFuncParser,
        "HardSigmoid": ActivationFuncParser,
        # Misc. operations
        "Transpose": TransposeParser,
        "Flatten": TransposeParser,
        "LayerNormalization": LayerNormalizationParser,
        "Concat": ConcatParser,
        "Clip": ClipParser,
        "Shape": ShapeParser,
        "Slice": SliceParser,
        "Resize": ResizeParser,
    }

    def __init__(self, run_name: str, network_name: str):
        self.network_name = network_name
        _model_path = os.path.join(MODEL_PATH, run_name, network_name + "_model.onnx")
        self.onnx_model: ModelProto = onnx.load(_model_path)
        onnx.checker.check_model(self.onnx_model)
        self.onnx_model: ModelProto = shape_inference.infer_shapes(self.onnx_model)

    def get_torch_model(self):
        """Convert ONNX model to PyTorch model using onnx2torch."""
        # Assuming self.model is the loaded ONNX model
        pytorch_model = convert(self.onnx_model)
        #pytorch_model = ConvertModel(model,experimental=True)
        return pytorch_model

    def _get_tensor_info(self, name: str):
        for i in self.onnx_model.graph.input:
            if i.name == name:
                return OnnxTensor.from_tensor(i.name, i.type.tensor_type, TensorKind.INPUT)

        for v in self.onnx_model.graph.value_info:
            if v.name == name:
                return OnnxTensor.from_tensor(v.name, v.type.tensor_type, TensorKind.HIDDEN)  
        
        for o in self.onnx_model.graph.output:
            if o.name == name:
                return OnnxTensor.from_tensor(o.name, o.type.tensor_type, TensorKind.OUTPUT)
        
        for init in self.onnx_model.graph.initializer:
            if init.name == name:
                return OnnxTensor.from_initializer(init.name, init)

    def _get_input_output_shape(self, node: NodeProto):
        in_name = node.input[0]
        in_shape = self._get_tensor_info(in_name).shape
        out_name = node.output[0]
        out_shape = self._get_tensor_info(out_name).shape
        return in_shape, out_shape
    
    def _get_tensors(self, names: list[str]):
        tensors: list[OnnxTensor] = []
        for name in names:
            tensors.append(self._get_tensor_info(name))
        return tensors
    
    def parse_graph(self) -> OnnxModel:
        model = OnnxModel(self.onnx_model)

        # Remember input and output tensors
        layer_inputs: dict[str, list[OnnxTensor]] = {}
        layer_outputs: dict[str, list[OnnxTensor]] = {}

        # First we take care of the input layer and add all tensors it produces
        inputs = [inp.name for inp in self.onnx_model.graph.input]
        assert len(inputs) == 1, "Currently only Onnx Models with one input are supported"
        input_tensors = self._get_tensors(inputs)
        for input, tensor in zip(inputs, input_tensors):
            layer_outputs[input] = [tensor]
            layer_repr = InputLayer(layer_name=input, output=tensor)
            model.add_layer(layer_repr)

        # Transform every hidden node of the onnx graph into an
        # internal representation for further processing. 
        for node in self.onnx_model.graph.node:
            if node.op_type == "Constant":
                continue
            logger.debug(f"Parsing layer {node.name} with op_type {node.op_type}")
            
            assert len(node.output) == 1, "Layers with more than one output are not supported"
            
            input_tensors = self._get_tensors(node.input)
            output_tensors = self._get_tensors(node.output)
            layer_inputs[node.name] = input_tensors
            layer_outputs[node.name] = output_tensors

            in_shape, out_shape = self._get_input_output_shape(node)
            parser_class = OnnxParser.OP_PARSER[node.op_type]
            parser: OnnxLayerParser = parser_class(node, 
                                                   in_shape, 
                                                   out_shape)
            layer: RenameLayer | ComputationLayer = parser.parse(input_tensors, output_tensors[0]) 

            model.add_layer(layer)
        
        
        # Finalize parsing by adding output layers
        outputs = [outp.name for outp in self.onnx_model.graph.output]
        model_output_tensors = self._get_tensors(outputs)
        for output, tensor in zip(outputs, model_output_tensors):
            layer_inputs[output] = [tensor]
            layer_repr = OutputLayer(layer_name=output, input=[tensor])
            model.add_layer(layer_repr)

        logger.debug(f"Done parsing Onnx model, there are {model.number_of_nodes()} layers")
        
        # Finally, determine edges between the nodes
        model.construct_layer_relations(layer_outputs, layer_inputs, self.network_name)
        return model
