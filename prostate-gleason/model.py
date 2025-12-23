"""
ResNet50 model for Prostate Cancer Gleason Grading
"""
import torch.nn as nn
from torchvision import models


def create_model(num_classes, pretrained=True):
    """
    Create ResNet50 model with custom classification head.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        Modified ResNet50 model
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        print("Loaded ImageNet pretrained weights (ResNet50_Weights.IMAGENET1K_V2)")
    else:
        model = models.resnet50(weights=None)
        print("Training from scratch (no pretrained weights)")

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, num_classes)
    )

    return model
