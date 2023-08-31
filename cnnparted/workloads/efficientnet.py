from torchvision import models

from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy


# Simply take the EfficientNet available in torchvision
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

accuracy_function = compute_classification_accuracy
