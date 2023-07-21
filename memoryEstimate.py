import torch
from torchvision.models import vgg16 , squeezenet1_1
from torchsummary import summary

# Load the VGG16 model
model = squeezenet1_1(pretrained=False)

# Specify the input size (224x224 RGB images)
input_size = (3, 224, 224)

# Print the model summary
summary(model, input_size=input_size)

# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the VGG16 model
model = squeezenet1_1(pretrained=False).to(device)

# Generate random input data
input_data = torch.randn(1, 3, 224, 224).to(device)

# Perform a forward pass to measure memory usage
with torch.no_grad():
    output = model(input_data)

def get_layerwise_memory_stats(model, dummy_input):


    layerwise_memory = {}

    def hook(module, input, output):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            layer_name = str(module)
            layer_memory = output.element_size() * output.nelement() / 1024 ** 2
            layerwise_memory[layer_name] = layer_memory

    handles = []
    for module in model.modules():
        handle = module.register_forward_hook(hook)
        handles.append(handle)

    # Perform a forward pass to trigger the hooks
    model(dummy_input)

    # Remove the hooks
    for handle in handles:
        handle.remove()

    return layerwise_memory



# Get the layerwise memory stats
memory_stats = get_layerwise_memory_stats(model, input_data)

tot_mem = 0

# Print the memory usage for each layer
for layer_name in memory_stats:
    print(f"Layer: {layer_name}\tMemory Usage: {memory_stats[layer_name]} MB")
    tot_mem += memory_stats[layer_name]
print(tot_mem)

