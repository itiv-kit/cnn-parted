from torchvision import models

from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy


# Simply take the SqueezeNet V1.1 available in torchvision
model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)

accuracy_function = compute_classification_accuracy
