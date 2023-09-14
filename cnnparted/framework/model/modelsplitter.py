import onnx
from onnx import shape_inference
import os
from framework.constants import NEW_MODEL_PATH, ROOT_DIR


class ModelSplitter:
    def __init__(self, input_path):
        self.input_path = input_path
        # model = onnx.load(input_path)

    def GiveUniqueNodeNames(self, model, prefix=""):
        optype_count = {}
        for n in model.graph.node:
            if n.op_type not in optype_count.keys():
                optype_count[n.op_type] = 0
            n.name = "%s%s_%d" % (prefix, n.op_type, optype_count[n.op_type])
            optype_count[n.op_type] += 1
        return model

    def clean_unused_initializers(self, model):
        used_initializers = set()

        for node in model.graph.node:
            used_initializers.update(node.input)

        unused_initializers = [
            init
            for init in model.graph.initializer
            if init.name not in used_initializers
        ]

        # Remove unused initializers from the model's initializer list
        for init in unused_initializers:
            model.graph.initializer.remove(init)

    def get_node_index_by_name(self, model_def, node_name):
        for index, node in enumerate(model_def.graph.node):
            if node.name == node_name:
                return index
        print(" Index Not found for :",node_name)
        return None

    def get_node_by_name(self, model_def, node_name):
        for node in model_def.graph.node:
            if node.name == node_name:
                return node
        return None

    def get_value_info_shape_and_type(self, model_def, value_info_name):
        value_info = None
        for vi in model_def.graph.value_info:
            if vi.name == value_info_name:
                value_info = vi
                break

        if value_info is None:
            if value_info_name == model_def.graph.output[0].name:
                out_shape = []
                for d in model_def.graph.output[0].type.tensor_type.shape.dim:
                    if d.dim_value == 0:
                        out_shape.append(None)
                    else:
                        out_shape.append(d.dim_value)
                type = model_def.graph.output[0].type.tensor_type.elem_type
                return out_shape, type

            else:
                print(f"ValueInfoProto '{value_info_name}' not found in graph.")
                return None

        shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
        data_type = value_info.type.tensor_type.elem_type

        return shape, data_type

    def remove_identity_nodes(self, model_def, modified_model_path):
        # Create a mapping of node outputs to their corresponding nodes
        output_to_node = {
            output: node for node in model_def.graph.node for output in node.output
        }

        # Find all identity nodes and their outputs
        identity_nodes = [
            node for node in model_def.graph.node if node.op_type == "Identity"
        ]
        identity_outputs = [output for node in identity_nodes for output in node.output]

        # Replace identity node outputs with their original inputs
        for node in model_def.graph.node:
            node.input[:] = [
                input
                if input not in identity_outputs
                else output_to_node[input].input[0]
                for input in node.input
            ]

        # Remove identity nodes from the graph
        non_identity_nodes = [
            node for node in model_def.graph.node if node not in identity_nodes
        ]
        model_def.graph.ClearField("node")
        model_def.graph.node.extend(non_identity_nodes)

        # Save the modified model
        onnx.save(model_def, modified_model_path)

        # print(f"Identity nodes removed. Modified model saved as '{modified_model_path}'.")

    def split_model(self, node_name, output_path_head, output_path_tail):
        if node_name == 'output':
            return True
         
        new_model = "new_model.onnx"
        new_model_path = os.path.join(ROOT_DIR, node_name + new_model)

        model = onnx.load(self.input_path)

        model = self.GiveUniqueNodeNames(model)
        model = shape_inference.infer_shapes(model)
        self.remove_identity_nodes(model, new_model_path)

        model = onnx.load(new_model_path)
        onnx.checker.check_model(model)
        onnx.save(model, new_model_path)

        newmodelhead = onnx.load(new_model_path)
        newmodeltail = onnx.load(new_model_path)
        
        if node_name == 'input':
            onnx.save(newmodeltail, output_path_tail)
            return False
        
        to_be_deleted_nodes_count = self.get_node_index_by_name(model, node_name) + 1
        node = self.get_node_by_name(model, node_name)
        shape, type = self.get_value_info_shape_and_type(model, node.output[0])

        oldnodes = [n for n in model.graph.node]

        if to_be_deleted_nodes_count >= len(oldnodes):
            return True

        newheadnodes = oldnodes[0:to_be_deleted_nodes_count]
        newtailnodes = oldnodes[to_be_deleted_nodes_count:]

        # delete all nodes after after the node[to_be_deleted_nodes_count] and build the head model
        head_output_name = newmodelhead.graph.node[to_be_deleted_nodes_count].input[0]
        del newmodelhead.graph.node[:]  # clear old nodes
        newmodelhead.graph.node.extend(newheadnodes)
        newmodelhead.graph.output.pop()
        out = [onnx.helper.make_tensor_value_info(head_output_name, type, shape)]
        newmodelhead.graph.output.extend(out)
        self.clean_unused_initializers(newmodelhead)
        onnx.checker.check_model(newmodelhead)
        onnx.save(newmodelhead, output_path_head)

        model_output = model.graph.output[0]
        out_shape = []
        for d in model_output.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                out_shape.append(None)
            else:
                out_shape.append(d.dim_value)

        tail_input_name = newmodeltail.graph.node[to_be_deleted_nodes_count].input[0]
        del newmodeltail.graph.node[:]
        newmodeltail.graph.node.extend(newtailnodes)
        newmodeltail.graph.input.pop()
        inp = [onnx.helper.make_tensor_value_info(tail_input_name, type, shape)]
        newmodeltail.graph.input.extend(inp)
        self.clean_unused_initializers(newmodeltail)
        onnx.checker.check_model(newmodeltail)
        onnx.save(newmodeltail, output_path_tail)

        return False
