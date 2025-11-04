import os
from enum import IntEnum, auto
from dataclasses import dataclass, field

import onnx
from onnx import shape_inference
import numpy as np

from framework.constants import MODEL_PATH
from framework.model.model_helper import ModelHelper

@dataclass
class ConvParameters:
    h: int
    w: int
    n: int
    m: int
    q: int
    p: int
    c: int
    s: int
    r: int
    wpad: int
    hpad: int
    wstride: int
    hstride: int
    ifms: int
    ofms: int
    weights: int
    groups: int
    depthwise: bool

    def __init__(self, cfg: dict):
        self.h = cfg["h"]
        self.w = cfg["w"]
        self.n = cfg["n"]
        self.m = cfg["m"]
        self.q = cfg["q"]
        self.p = cfg["p"]
        self.c = cfg["c"]
        self.s = cfg["s"]
        self.r = cfg["r"]
        self.wpad = cfg["wpad"]
        self.hpad = cfg["hpad"]
        self.wstride = cfg["wstride"]
        self.hstride = cfg["hstride"]
        self.ifms = cfg["ifms"]
        self.ofms = cfg["ofms"]
        self.weights = cfg["weights"]
        self.groups = cfg["groups"]
        self.depthwise = False #TODO Figure this out

    @property
    def dimensions(self):
        return ["N", "M", "P", "Q", "C", "R", "S"]
    
    def get_input_tensor_shapes(self):
        input_shape = [self.n, self.c, self.h, self.w]
        kernel_shape = [self.m, self.c, self.r, self.s]
        bias_shape = [self.m]
        return [input_shape, kernel_shape, bias_shape]

    def get_input_tensor_dimensions(self):
        #(N,Cin​,H,W) and output (N,Cout,Hout,Wout)
        # Kernel: (M, C/groups, R, S)
        input_dimensions = ["N", "C", "H", "W"]
        kernel_dimensions = ["M", "C", "R", "S"]
        bias_dimensions = ["M"]
        return [input_dimensions, kernel_dimensions, bias_dimensions]
    
    def get_output_tensor_dimensions(self):
        return [["N", "M", "P", "Q"]]


@dataclass
class GemmParameters:
    b: int | None
    n: int
    m: int
    c: int
    has_bias: bool
    weights: int

    def __init__(self, cfg: dict):
        self.b = cfg.get("b")
        self.n = cfg["n"]
        self.m = cfg["m"]
        self.c = cfg["c"]
        self.weights = cfg["weights"]
        self.has_bias = cfg["has_bias"]
    
    @property
    def dimensions(self):
        dims = ["N", "M", "C"]
        if self.b is not None:
            dims = ["B"] + dims
        return dims
    
    def get_input_tensor_shapes(self):
        input_shape = [self.n, self.c]
        weight_shape = [self.c, self.m]
        bias_shape = [self.n, self.m]
        out = [input_shape, weight_shape, ] #bias_shape
        if self.has_bias:
            out.append(bias_shape)
        return out
    
    def get_input_tensor_dimensions(self):
        out = [["N", "C"], ["C", "M"], ]
        if self.has_bias:
            out.append(["N", "M"])
        return out

    def get_output_tensor_dimensions(self):
        return [["N", "M"]]
    
@dataclass
class PoolParameters:
    kernel_shape: np.ndarray
    padding: np.ndarray | None
    stride: np.ndarray

    def __init__(self, cfg: dict):
        self.kernel_shape = cfg["kernel"]
        self.padding = cfg["padding"]
        self.stride = cfg["stride"]
    
    @property
    def dimensions(self):
        return []
    
    def get_input_tensor_shapes(self):
        return [[]]

    def get_input_tensor_dimensions(self):
        return [[]]

    def get_output_tensor_dimensions(self):
        return [[]]

class ActivationKind(IntEnum):
    Relu = auto()
    Sigmoid = auto()
    Erf = auto()
    unknown = auto()

@dataclass
class ActivationParameters:
    shape: list[int]
    kind: ActivationKind

    def __init__(self, cfg: dict):
        self.shape = cfg["shape"]
        self.kind = ActivationKind[cfg["kind"]]
    
    @property
    def dimensions(self):
        return []
    
    def get_input_tensor_shapes(self):
        return [self.shape]
    
    def get_input_tensor_dimensions(self):
        return [[]]

    def get_output_tensor_dimensions(self):
        return [[]]

class ElementwiseKind(IntEnum):
    Add = auto()
    Mul = auto()
    Div = auto()
    unknown = auto()

@dataclass
class ElementwiseParameters:
    shape: list[int]
    kind: ElementwiseKind

    def __init__(self, cfg: dict):
        self.shape = cfg["shape"]
        self.kind = ElementwiseKind[cfg["kind"]]
    
    @property
    def dimensions(self):
        return []
    
    def get_input_tensor_shapes(self):
        return [self.shape,  self.shape]
    
    def get_input_tensor_dimensions(self):
        return [[], []]

    def get_output_tensor_dimensions(self):
        return [[]]

class MiscOperationsKind(IntEnum):
    Flatten = auto()
    GlobalAveragePool = auto()
    Concat = auto()
    Transpose = auto()
    LayerNormalization = auto()
    Constant = auto()
    unknown = auto()

@dataclass
class UnaryMiscOperationParameters:
    in_shape: list[int]
    out_shape : list[int]
    kind: MiscOperationsKind

    def __init__(self, cfg: dict):
        self.in_shape = cfg["in_shape"]
        self.out_shape = cfg["out_shape"]
        self.kind = MiscOperationsKind[cfg["kind"]]

    @property
    def dimensions(self):
        return []
    
    def get_input_tensor_shapes(self):
        return [self.in_shape]
    
    def get_input_tensor_dimensions(self):
        return [[]]

    def get_output_tensor_dimensions(self):
        return [[]]

@dataclass
class BinaryMiscOperationParameters:
    a_shape: list[int]
    b_shape: list[int]
    out_shape = list[int]
    kind: MiscOperationsKind

    def __init__(self, cfg: dict):
        self.a_shape = cfg["a_shape"]
        self.b_shape = cfg["b_shape"]
        self.out_shape = cfg["out_shape"]
        self.kind = MiscOperationsKind[cfg["kind"]]

    @property
    def dimensions(self):
        return []

    def get_input_tensor_shapes(self):
        return [self.a_shape, self.b_shape]
    
    def get_input_tensor_dimensions(self):
        return [[], []]

    def get_output_tensor_dimensions(self):
        return [[]]

@dataclass
class ConstantOperationParameters:
    kind: MiscOperationsKind

    def __init__(self):
        self.kind = MiscOperationsKind.Constant

    @property
    def dimensions(self):
        return []

    def get_input_tensor_shapes(self):
        return []
    
    def get_input_tensor_dimensions(self):
        return []

    def get_output_tensor_dimensions(self):
        return [[]]


@dataclass
class TensorParameters:
    name: str
    shape: list[str]
    #elem_type: np.integer | np.floating
    definition: str = None
    used_by: list[str] = field(default_factory=list)
    dimensions: list[str] = None

type ComputeParameters = ConvParameters | GemmParameters
type GeneralLayerParameters = ComputeParameters | PoolParameters | ActivationParameters | ElementwiseParameters | BinaryMiscOperationParameters | UnaryMiscOperationParameters

@dataclass
class LayerParameters:
    # A layer consists of the computational parameters and the input and output tensors
    name: str
    inputs: list[str]
    op_type: str
    outputs: list[str]
    output_size: np.ndarray
    params: GeneralLayerParameters | None
    dimensions: list[str]
    children: list[str]

    def __init__(self, cfg: dict, params: ComputeParameters | None):
        self.name = cfg["name"]
        self.op_type = cfg["op_type"]

        if isinstance(cfg["input"], list):
            self.inputs = cfg["input"]
        else:
            self.inputs = [cfg["input"]]

        if isinstance(cfg["output"], list):
            self.outputs = cfg["output"]
        else:
            self.outputs = [cfg["output"]]

        if len(cfg["output_size"]) == 0 and cfg["op_type"] == "Constant":
            self.output_size = [[]]
        elif isinstance(cfg["output_size"][0], list):
            self.output_size = cfg["output_size"]
        else:
            self.output_size = [cfg["output_size"]]

        self.params = params
        if params is not None:
            self.dimensions = params.dimensions
        else:
            self.dimensions = None
        
        self.children = None


class TreeModel:
    def __init__(self, run_name, network_name, input_size):
        _model_path = os.path.join(MODEL_PATH, run_name, network_name + "_model.onnx")
        self.input_size = input_size
        self.model_helper = ModelHelper()
        self._model = onnx.load(_model_path)
        onnx.checker.check_model(self._model)
        self._model = shape_inference.infer_shapes(self._model)
        self.output_sizes = self._get_output_sizes()
        self._layer_tree = self._get_layers_data()
        self.new_model_path = os.path.join(MODEL_PATH, run_name, network_name + "_new_model.onnx")
        onnx.save(self._model, self.new_model_path)

    def get_torch_model(self):
        return self.model_helper.convert_to_pytorch(self._model)

    def get_tree(self):
        return self._layer_tree

    def _get_layers_data(self) -> list[LayerParameters]:
        graph_def = self._model.graph

        def add_name_hint(layer: dict):
            # We need the weight and bias as string here for later processing, some onnx models 
            # just call them onnx::Conv_193 and similar
            if "weight" not in layer["input"][1]:
                layer["input"][1] = layer["input"][1] + "_weight"
            
            # MatMul has not bias tensor here, so we need to guard against that
            if len(layer["input"]) > 2:
                if "bias" not in layer["input"][2]:
                    layer["input"][2] = layer["input"][2] + "_bias"
            return layer

        self.in_layer, self.out_layer = self._get_in_out_layers(graph_def)

        nodes = graph_def.node
        output = []
        output.append(LayerParameters(self.in_layer, None))

        for node in nodes:
            if node.op_type == 'Identity': #or node.op_type == "Constant":
                continue

            layer = {
                "name": node.name,
                "op_type": node.op_type,
                "input": list(node.input),
            }
            
            # Check if any of the node's outputs match the keys in output_sizes
            matched_outputs = [output for output in node.output if output in self.output_sizes]

            if matched_outputs:
                layer["output_size"] = np.array([self.output_sizes[i] for i in matched_outputs]).flatten()
                layer["output"] = matched_outputs
            else:
                layer["output_size"] = self.out_layer['output_size']
                layer['output'] = self.out_layer['name']
            
            if node.op_type =='Conv':
                conv_params = self._get_conv_params(node)
                params = ConvParameters(conv_params)
                layer = add_name_hint(layer)
            elif node.op_type in ['MaxPool', 'AveragePool']:
                pool_params = self._get_pool_params(node)
                params = PoolParameters(pool_params)
            elif node.op_type =='Gemm' or node.op_type == "MatMul":
                gemm_params = self._get_gemm_params(node)
                params = GemmParameters(gemm_params)
                layer = add_name_hint(layer)
            elif node.op_type in ["Relu", "Sigmoid", "Erf"]:
                activation_params = self._get_activation_params(node)
                params = ActivationParameters(activation_params)
            elif node.op_type in ["Mul", "Add", "Div", ""]:
                eltwise_params = self._get_elementwise_params(node)
                params = ElementwiseParameters(eltwise_params)
            elif node.op_type in ["Flatten", "GlobalAveragePool", "Transpose", "LayerNormalization"]:
                # LayerNorm has weight and bias as variables, which we ignore here
                # and assume that they are simply statically available
                if node.op_type == "LayerNormalization":
                    layer["input"] = [layer["input"][0]]
                misc_params = self._get_unary_misc_params(node)
                params = UnaryMiscOperationParameters(misc_params)
            elif node.op_type in ["Concat"]:
                misc_params = self._get_binary_misc_params(node)
                params = BinaryMiscOperationParameters(misc_params)
            elif node.op_type == "Constant":
                params = ConstantOperationParameters()
            else:
                print(f"Unknown Op: {node}")
                params = None
            
            layer_params = LayerParameters(layer, params)

            output.append(layer_params)

        output.append(LayerParameters(self.out_layer, None))

        self._create_layer_relationships(output)

        return output

    def _get_pool_params(self,node):
        attributes = {}
        for attr in node.attribute:
            if attr.name == 'kernel_shape':
                attributes['kernel_shape'] = list(attr.ints)
            elif attr.name == 'strides':
                attributes['strides'] = list(attr.ints)
            elif attr.name == 'pads':
                attributes['pads'] = list(attr.ints)
        
        kernel_shape = attributes["kernel_shape"]
        pads = attributes.get("pad", [0 for _ in kernel_shape])
        strides = attributes.get("strides", [1 for _ in kernel_shape])

        output = {
            'kernel': kernel_shape,
            'padding': pads,
            'stride': strides,
        }

        return output

    def _get_gemm_params(self,node):
        matched_outputs = [output for output in node.output if output in self.output_sizes]
        o_shape = [self.output_sizes[i] for i in matched_outputs]
        matched_inputs = [input for input in node.input if input in self.output_sizes]
        i_shape = [self.output_sizes[i] for i in matched_inputs]
        has_bias = len(node.input) == 3 #input, weight, bias

        if matched_outputs == [] and node.output[0] == "output":
            o_shape = [self.out_layer['output_size']]
        if matched_inputs == [] and node.input[0]=="input":
            c = self.in_layer['output_size'][1]
        else :
            c = i_shape[0][1]

        output = {
            'n': o_shape[0][0],
            'm': o_shape[0][1],
            'c': c,
            'weights': np.prod(i_shape + o_shape),
            'has_bias': has_bias 
        }

        # Check if this is a batch matmul
        if len(o_shape[0]) == 4 and len(i_shape[0]) == 4:
            # Shape layout: [1, b, x, y]
            assert o_shape[0][0] == 1 and i_shape[0][0] == 1, "4D matmul is currently not supported" 
            output = {
                'b': o_shape[0][1],
                'n': o_shape[0][2],
                'm': o_shape[0][3],
                'c': i_shape[0][3],
                'weights': np.prod(i_shape + o_shape),
                'has_bias': has_bias 
            }

        return output

    def _get_conv_params(self,node):
        #(N,Cin​,H,W) and output (N,Cout,Hout,Wout)
        # Kernel: (M, C/groups, R, S)
        attributes = {}
        attributes['groups'] = 1
        for attr in node.attribute:
            if attr.name == 'kernel_shape':
                attributes['kernel_shape'] = list(attr.ints)
            elif attr.name == 'strides':
                attributes['strides'] = list(attr.ints)
            elif attr.name == 'pads':
                attributes['pads'] = list(attr.ints)
            elif attr.name == 'dilations':
                attributes['dilations'] = list(attr.ints)
            elif attr.name == 'groups' or attr.name == 'group':
                attributes['groups'] = attr.i

        matched_outputs = [output for output in node.output if output in self.output_sizes]
        o_shape = [self.output_sizes[i] for i in matched_outputs]
        matched_inputs = [input for input in node.input if input in self.output_sizes]
        i_shape = [self.output_sizes[i] for i in matched_inputs]


        if matched_outputs == [] and node.output[0] == "output":
            o_shape = [self.out_layer['output_size']]
        if matched_inputs == [] and node.input[0]=="input":
            c = self.in_layer['output_size'][1]
            i_shape = [self.in_layer["output_size"]]
        else :
            c = i_shape[0][1]

        ofms    = o_shape[0][0]*o_shape[0][1]*o_shape[0][2]*o_shape[0][3]
        weights = o_shape[0][1]*c*attributes['kernel_shape'][0]*attributes['kernel_shape'][1]
        ifms    = o_shape[0][0]*c*((o_shape[0][3]-1)*attributes['strides'][1]+attributes['kernel_shape'][1])*((o_shape[0][2]-1)*attributes['strides'][0]+attributes['kernel_shape'][0])
        output = {
            'h': i_shape[0][2],
            'w': i_shape[0][3],
            'n': o_shape[0][0],
            'm': o_shape[0][1] // attributes['groups'],
            'q': o_shape[0][2],
            'p': o_shape[0][3],
            'c': c // attributes['groups'],
            's': attributes['kernel_shape'][0],
            'r': attributes['kernel_shape'][1],
            'wpad': attributes['pads'][0],
            'hpad': attributes['pads'][1],
            'wstride': attributes['strides'][0],
            'hstride': attributes['strides'][1],
            'ifms':ifms,
            'ofms':ofms,
            'weights':weights,
            'groups': attributes.get("groups", 1)
        }

        return output

    def _get_activation_params(self, node) -> dict:
        matched_outputs = [output for output in node.output if output in self.output_sizes]
        o_shape = [self.output_sizes[i] for i in matched_outputs]
        matched_inputs = [input for input in node.input if input in self.output_sizes]
        i_shape = [self.output_sizes[i] for i in matched_inputs]


        if matched_outputs == [] and node.output[0] == "output":
            o_shape = [self.out_layer['output_size']]
        if matched_inputs == [] and node.input[0]=="input":
            i_shape = [self.in_layer["output_size"]]

        output = {
            "shape": i_shape[0],
            "kind": node.op_type
        }
        return output
        
    def _get_elementwise_params(self, node) -> dict:
        matched_outputs = [output for output in node.output if output in self.output_sizes]
        o_shape = [self.output_sizes[i] for i in matched_outputs]
        matched_inputs = [input for input in node.input if input in self.output_sizes]
        i_shape = [self.output_sizes[i] for i in matched_inputs]


        if matched_outputs == [] and node.output[0] == "output":
            o_shape = [self.out_layer['output_size']]
        if matched_inputs == [] and node.input[0]=="input":
            i_shape = [self.in_layer["output_size"]]

        output = {
            "shape": i_shape[0],
            "kind": node.op_type
        }
        return output
        
    def _get_unary_misc_params(self, node) -> dict:
        matched_outputs = [output for output in node.output if output in self.output_sizes]
        o_shape = [self.output_sizes[i] for i in matched_outputs]
        matched_inputs = [input for input in node.input if input in self.output_sizes]
        i_shape = [self.output_sizes[i] for i in matched_inputs]


        if matched_outputs == [] and node.output[0] == "output":
            o_shape = [self.out_layer['output_size']]
        if matched_inputs == [] and node.input[0]=="input":
            i_shape = [self.in_layer["output_size"]]

        output = {
            "in_shape": i_shape[0],
            "out_shape": o_shape[0],
            "kind": node.op_type
        }
        return output

    def _get_binary_misc_params(self, node) -> dict:
        matched_outputs = [output for output in node.output if output in self.output_sizes]
        o_shape = [self.output_sizes[i] for i in matched_outputs]
        matched_inputs = [input for input in node.input if input in self.output_sizes]
        i_shape = [self.output_sizes[i] for i in matched_inputs]


        if matched_outputs == [] and node.output[0] == "output":
            o_shape = [self.out_layer['output_size']]
        if matched_inputs == [] and node.input[0]=="input":
            i_shape = [self.in_layer["output_size"]]

        output = {
            "a_shape": i_shape[0],
            "b_shape": i_shape[1],
            "out_shape": o_shape[0],
            "kind": node.op_type
        }
        return output


    def _create_layer_relationships(self, layers_data: list[LayerParameters]):
        for layer in layers_data:

            layer.children = []
            for other_layer in layers_data:
                #  input.output.name == output.input.name leads to error in the the tree
                # (needs to be solved in another way for graphs with direct connection between input and output(if any))
                if ( layer.name == 'input' and other_layer.name == 'output'):
                    continue
                try:
                    if layer.name != other_layer.name and set(layer.outputs) & set(other_layer.inputs):
                        layer.children.append(other_layer.name)
                except:
                    breakpoint()
                    print(other_layer)

    def _get_output_sizes(self) -> dict[str, list[int]]:
        output_sizes = {}
        graph_def = self._model.graph
        for info in graph_def.value_info:
            #if "Constant" in info.name:
            #    continue
            dims = [dim.dim_value for dim in info.type.tensor_type.shape.dim]
            output_sizes[info.name] = dims
        return output_sizes

    def _give_unique_node_names(self, model, prefix=""):
        optype_count = {}
        for n in model.graph.node:
            if n.op_type not in optype_count.keys():
                optype_count[n.op_type] = 0
            n.name = f"{prefix}{n.op_type}_{optype_count[n.op_type]}"
            optype_count[n.op_type] += 1
        return model

    def _get_in_out_layers(self,graph_def):

        graph_output = graph_def.output[0]
        out_shape=[]

        for d in graph_output.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                out_shape.append(None)
            else:
                out_shape.append(d.dim_value)

        # This loop assumes that the output shapes can be concatendated along axis=0
        # The dummy zero array is only used to get the shape of the concatenated output
        if len(graph_def.output) > 1:
            out_shapes = []
            for out in graph_def.output:
                shape = []
                for d in out.type.tensor_type.shape.dim:
                    if d.dim_value == 0:
                        shape.append(None)
                    else:
                        shape.append(d.dim_value)
                out_shapes.append(shape)
            tmp = [np.zeros(s) for s in out_shapes]
            out_shape = np.concatenate((tmp[0], tmp[1]), axis=0).shape

        output_layer = {
            "name": 'output',
            "op_type": graph_output.type.tensor_type.elem_type,
            "output_size": out_shape,
            "input": 'output',
            "output":[]
            }


        graph_input = graph_def.input[0]
        in_shape = []
        for d in graph_input.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                in_shape.append(None)
            else:
                in_shape.append(d.dim_value)

        input_layer = {
            "name": 'input',
            "output" : [graph_input.name],
            "op_type": graph_input.type.tensor_type.elem_type,
            "output_size": in_shape,
            "input":[]
            }
        return input_layer,output_layer
