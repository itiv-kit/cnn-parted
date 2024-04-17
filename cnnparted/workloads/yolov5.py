import torch
from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy


# Simply take the model available in torchvision
model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s-cls.pt").cpu()
accuracy_function = compute_classification_accuracy
