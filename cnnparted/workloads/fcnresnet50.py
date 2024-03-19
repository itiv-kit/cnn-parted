from torchvision import models

from model_explorer.accuracy_functions.segmentation_accuracy import compute_sematic_segmentation_accuracy

# Simply take the FCN-ResNet-50 available in torchvision
model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)

accuracy_function = compute_sematic_segmentation_accuracy
