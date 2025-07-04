import os
import onnx
from onnx import shape_inference
import numpy as np
from framework.constants import MODEL_PATH
from framework.model.model_helper import ModelHelper


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

    def _get_layers_data(self):
        graph_def = self._model.graph


        self.in_layer, self.out_layer = self._get_in_out_layers(graph_def)

        nodes = graph_def.node
        output = []
        output.append(self.in_layer)

        for node in nodes:
            if node.op_type == 'Identity':
                continue

            layer = {
                "name": node.name,
                "op_type": node.op_type,
                "input": node.input,
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
                layer['conv_params'] = conv_params
            elif node.op_type =='MaxPool':
                pool_params = self._get_pool_params(node)
                layer['pool_params'] = pool_params
            elif node.op_type =='AveragePool':
                pool_params = self._get_pool_params(node)
                layer['pool_params'] = pool_params
            elif node.op_type =='Gemm' or node.op_type == "MatMul":
                gemm_params = self._get_gemm_params(node)
                layer['gemm_params'] = gemm_params

            output.append(layer)

        output.append(self.out_layer)
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

        output = {
            'kernel': attributes.get('kernel_shape'),
            'padding': attributes.get('pads'),
            'stride': attributes.get('strides')
        }

        return output

    def _get_gemm_params(self,node):
        matched_outputs = [output for output in node.output if output in self.output_sizes]
        o_shape = [self.output_sizes[i] for i in matched_outputs]
        matched_inputs = [input for input in node.input if input in self.output_sizes]
        i_shape = [self.output_sizes[i] for i in matched_inputs]

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
            'weights': np.prod(i_shape + o_shape)
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
                'weights': np.prod(i_shape + o_shape)
            }

        return output

    def _get_conv_params(self,node):
        #(N,Cin​,H,W) and output (N,Cout,Hout,Wout)(N,Cout​,Hout​,Wout​)
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
        else :
            c = i_shape[0][1]

        ofms    = o_shape[0][0]*o_shape[0][1]*o_shape[0][2]*o_shape[0][3]
        weights = o_shape[0][1]*c*attributes['kernel_shape'][0]*attributes['kernel_shape'][1]
        ifms    = o_shape[0][0]*c*((o_shape[0][3]-1)*attributes['strides'][1]+attributes['kernel_shape'][1])*((o_shape[0][2]-1)*attributes['strides'][0]+attributes['kernel_shape'][0])
        output = {
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

    def _create_layer_relationships(self, layers_data):
        for layer in layers_data:

            layer['children'] = []
            for other_layer in layers_data:
                #  input.output.name == output.input.name leads to error in the the tree
                # (needs to be solved in another way for graphs with direct connection between input and output(if any))
                if ( layer['name']=='input' and other_layer['name']=='output'):
                    continue
                if layer['name'] != other_layer['name'] and set(layer['output']) & set(other_layer['input']):
                    layer['children'].append(other_layer)

    def _get_output_sizes(self):
        output_sizes = {}
        graph_def = self._model.graph
        for info in graph_def.value_info:
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
