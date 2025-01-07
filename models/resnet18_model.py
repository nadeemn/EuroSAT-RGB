import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18:
    def __init__(self, num_classes):
        self.model = resnet18(weights = ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_features = num_classes)

    def get_model(self):
        return self.model