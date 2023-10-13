from torchvision import models

from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy


# Simply take the VGG16 available in torchvision
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

accuracy_function = compute_classification_accuracy
