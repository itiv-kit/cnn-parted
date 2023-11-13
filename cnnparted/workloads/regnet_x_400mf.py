from torchvision import models

from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy


# Simply take the EfficientNet available in torchvision
model = models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V2)

accuracy_function = compute_classification_accuracy
