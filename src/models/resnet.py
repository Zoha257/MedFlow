from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


def create_resnet18(num_classes: int = 9) -> nn.Module:
    """Create a ResNet18 classifier without external pretrained weights."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
