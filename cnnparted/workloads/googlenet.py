from torchvision import models
from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy

# Simply take the GoogLeNet available in torchvision but disable input transform
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1, transform_input = False)
accuracy_function = compute_classification_accuracy
