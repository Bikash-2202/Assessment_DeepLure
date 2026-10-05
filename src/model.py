import torch
import torch.nn as nn

from torchvision import models


def create_model(num_classes):

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze initial layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace classification layer
    input_features = model.fc.in_features

    model.fc = nn.Sequential(

        nn.Dropout(
            p=0.4
        ),

        nn.Linear(
            input_features,
            num_classes
        )
    )

    return model