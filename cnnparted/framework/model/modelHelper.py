import onnx
from onnx import helper as onnx_helper
from onnx2pytorch import ConvertModel
from copy import deepcopy

class modelHelper:

    def __init__(self):
        pass


    def add_identity_layers(self,_model):
        model = deepcopy(_model)
        """Add identity layers after every node in the ONNX model."""
        for i, node in enumerate(list(model.graph.node)):
            # Create an identity node
            identity_node = onnx_helper.make_node(
                'Identity',
                inputs=[node.output[0]],   # connect input of identity to output of the current node
                outputs=[node.output[0]+'_' + node.name +'_identity'],  # create a new output name
                name=node.name + '_identity'  # name the node
            )

            # Replace next node's input with the output of this new identity node
            if i + 1 < len(model.graph.node):
                next_node = model.graph.node[i + 1]
                for j, input_name in enumerate(next_node.input):
                    if input_name == node.output[0]:
                        next_node.input[j] = identity_node.output[0]

            # Append the identity node to the model graph
            model.graph.node.insert(i + 1, identity_node)

        return model

    def save_model(self,model, output_path):
        """Save the modified model."""
        onnx.save(model, output_path)

    def convert_to_pytorch(self,model):
        """Convert ONNX model to PyTorch model using onnx2pytorch."""
        # Assuming self.model is the loaded ONNX model
        pytorch_model = ConvertModel(model,experimental=True)
        return pytorch_model
