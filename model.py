import torch
import torch.nn as nn
from torchvision import models

def get_model():
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for cat/dog (2 classes)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

    return model