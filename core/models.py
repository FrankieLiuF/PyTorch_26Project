"""
Model management module
This module provides model building and configuration for all CNN architectures
used in the study.

Functions:
    1. get_model
    2. get_model_config
    3. get_available_models
    4. _get_feature_dim
    5. _replace_classifier
"""
import sys
# print(sys.executable)
import torch.nn as nn
from torchvision.models import (
    efficientnet_v2_m, EfficientNet_V2_M_Weights, # V1
    resnext50_32x4d, ResNeXt50_32X4D_Weights, # V2
    mobilenet_v3_large, MobileNet_V3_Large_Weights, # V2
    densenet161, DenseNet161_Weights # V1
)

from .config import Config

# Registry of pretrained CNN models with their weight sources and classifier names
# The feature_dim (input to classifier) is detected dynamically in get_model()
MODEL_CONFIGS = {
    'efficientnet_v2_m': {
        'weights': EfficientNet_V2_M_Weights.IMAGENET1K_V1,
        'builder': efficientnet_v2_m,
        'classifier_name': 'classifier',
        'input_size': 224  # 480 from weights.transforms().crop_size
    },
    'resnext50_32x4d': {
        'weights': ResNeXt50_32X4D_Weights.IMAGENET1K_V2,
        'builder': resnext50_32x4d,
        'classifier_name': 'fc',  # ResNet uses 'fc' instead of 'classifier'
        'input_size': 224  # from weights.transforms().crop_size
    },
    'mobilenet_v3_large': {
        'weights': MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        'builder': mobilenet_v3_large,
        'classifier_name': 'classifier',
        'input_size': 224  # from weights.transforms().crop_size
    },
    'densenet161': {
        'weights': DenseNet161_Weights.IMAGENET1K_V1,
        'builder': densenet161,
        'classifier_name': 'classifier',
        'input_size': 224  # from weights.transforms().crop_size
    },
}

# Build model by name and replace classifier for target number of classes
def get_model(model_name, num_classes=None, device='cpu'):
    """
    Create a model by name and replace the classifier layer

    Args:
        model_name: str, model name, must be a key in MODEL_CONFIGS
        num_classes: int, number of output classes, defaults to Config.NUM_CLASSES
        device: device

    Return:
        torch.nn.Module: Model instance with replaced classifier

    """
    if num_classes is None:
        num_classes = Config.NUM_CLASSES

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}."
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_name]

    # Build model with pretrained ImageNet weights (transfer learning)
    model = config['builder'](weights=config['weights'])

    # Detect the input dimension of the classifier head to be replaced
    classifier_name = config['classifier_name']
    feature_dim = _get_feature_dim(model, classifier_name)

    # Replace the classifier layer
    model = _replace_classifier(model, classifier_name, feature_dim, num_classes)

    return model.to(device)


# Get model configuration dict by name
def get_model_config(model_name):
    """
    Get model configuration

    Args:
        model_name: str, model_name

    Returns:
        dict: Model configuration dictionary
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}."
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name]


# Get list of all available model names
def get_available_models():
    return list(MODEL_CONFIGS.keys())


# Dynamically get the input feature dimension of the classifier layer
def _get_feature_dim(model, classifier_name):
    if classifier_name == 'classifier':
        if isinstance(model.classifier, nn.Sequential):
            # Find the FIRST Linear layer's in_features, not the last.
            # MobileNet's classifier is Linear(960,1280)+Linear(1280,1000):
            # using the last layer would give 1280, but the backbone outputs 960.
            for layer in model.classifier:
                if isinstance(layer, nn.Linear):
                    return layer.in_features
            raise ValueError('No Linear layer found in classifier Sequential')
        else:
            return model.classifier.in_features
    elif classifier_name == 'fc':
        return model.fc.in_features
    else:
        raise ValueError(f'Unknown classifier_name: {classifier_name}')

# Replace the classifier layer with a new one for the target number of classes
def _replace_classifier(model, classifier_name, feature_dim, num_classes):
    if classifier_name == 'classifier':
        # Preserve the original dropout rate from the pretrained model
        dropout_rate = 0.2
        if isinstance(model.classifier, nn.Sequential):
            for layer in model.classifier:
                if isinstance(layer, nn.Dropout):
                    dropout_rate = layer.p
                    break

        # replace classifier with new Sequential
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(feature_dim, num_classes)
        )
    elif classifier_name == 'fc':
        # Replace fc layer directly
        model.fc = nn.Linear(feature_dim, num_classes)

    return model
