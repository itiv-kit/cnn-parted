from torchvision import models
from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy

# Simply take the model available in torchvision
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
accuracy_function = compute_classification_accuracy
