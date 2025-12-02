import os

import onnx
from onnx import shape_inference
import numpy as np

#from onnx2pytorch import ConvertModel
from onnx2torch import convert

from framework.constants import MODEL_PATH
from framework.model.layer_parameters import (LayerParameters, 
                                              ConvParameters, 
                                              GemmParameters, 
                                              PoolingParameters, 
                                              ActivationParameters, 
                                              ElementwiseParameters,
                                              BinaryMiscOperationParameters, 
                                              UnaryMiscOperationParameters, 
                                              ConstantOperationParameters)


class TreeModel:
    def __init__(self, run_name, network_name, input_size):
        _model_path = os.path.join(MODEL_PATH, run_name, network_name + "_model.onnx")
        self.input_size = input_size
        self._model = onnx.load(_model_path)
        onnx.checker.check_model(self._model)
        self._model = shape_inference.infer_shapes(self._model)
        self.output_sizes = self._get_output_sizes()
        self._layer_tree = self._get_layers_data()
        self.new_model_path = os.path.join(MODEL_PATH, run_name, network_name + "_new_model.onnx")
        onnx.save(self._model, self.new_model_path)

    def get_torch_model(self):
        """Convert ONNX model to PyTorch model using onnx2torch."""
        # Assuming self.model is the loaded ONNX model
        pytorch_model = convert(self._model)
        #pytorch_model = ConvertModel(model,experimental=True)
        return pytorch_model

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
                params = PoolingParameters(pool_params)
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
