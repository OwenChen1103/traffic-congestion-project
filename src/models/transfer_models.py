"""
transfer_models.py
------------------
Transfer learning models — Models 2 & 3.
Wraps torchvision pretrained architectures with a new classification head.

Supported:
  - mobilenet_v2
  - resnet50
  - efficientnet_b0  (optional third model)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_mobilenet_v2(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_resnet50(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


MODEL_BUILDERS = {
    "mobilenet_v2": build_mobilenet_v2,
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
}


def build_model(name: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](num_classes=num_classes, pretrained=pretrained)
