import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, layer_infos):
        super(CustomModel, self).__init__()


    def forward(self, x):
        for layer in self.infos:
            x = layer.module(x)
            print(layer.var_name)
        return x

# Create the model
# model = CustomModel(layer_infos)

# # Test the model
# dummy_input = torch.randn(1, 3, 227, 227)  # Example input tensor
# output = model(dummy_input)
# print(output.shape)
