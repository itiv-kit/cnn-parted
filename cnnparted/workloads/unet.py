import torch
from torchvision import models
from model_explorer.accuracy_functions.segmentation_accuracy import compute_sematic_segmentation_accuracy

# Simply take the model available in torchvision
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
accuracy_function = compute_sematic_segmentation_accuracy
