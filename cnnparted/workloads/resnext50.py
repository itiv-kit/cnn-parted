from torchvision import models
from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy

# Simply take the model available in torchvision
model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
accuracy_function = compute_classification_accuracy
