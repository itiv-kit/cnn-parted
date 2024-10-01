from torchvision import models
from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy

# Simply take the model available in torchvision
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
accuracy_function = compute_classification_accuracy
