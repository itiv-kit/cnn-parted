from torchvision import models

from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy


# Simply take the ResNet-50 available in torchvision
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

accuracy_function = compute_classification_accuracy
