from onnx2pytorch import ConvertModel

class ModelHelper:

    def __init__(self):
        pass

    def convert_to_pytorch(self,model):
        """Convert ONNX model to PyTorch model using onnx2pytorch."""
        # Assuming self.model is the loaded ONNX model
        pytorch_model = ConvertModel(model,experimental=True)
        return pytorch_model
