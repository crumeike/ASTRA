#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model architecture definitions for post-tornado damage recognition experiments.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_activation_function(activation_name):
    """
    Get activation function by name.
    
    Args:
        activation_name (str): Name of activation function
    
    Returns:
        nn.Module: Activation function
    """
    if activation_name.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_name.lower() == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif activation_name.lower() == 'gelu':
        return nn.GELU()
    elif activation_name.lower() == 'swish' or activation_name.lower() == 'silu':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")


def create_model(model_name, num_classes, pretrained=True, activation='relu'):
    """
    Create a model with the specified architecture.
    
    Args:
        model_name (str): Model architecture name
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        activation (str): Activation function name
    
    Returns:
        nn.Module: Neural network model
    """
    # Get activation function
    act_fn = get_activation_function(activation)

    # ResNet family
    if model_name.startswith('resnet'):
        if model_name.startswith('resnet'):
            if model_name == 'resnet18':
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1, pretrained=pretrained)
            elif model_name == 'resnet34':
                model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1, pretrained=pretrained)
            elif model_name == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1, pretrained=pretrained)
            elif model_name == 'resnet101':
                model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1, pretrained=pretrained)              
            elif model_name == 'resnet152':
                model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1, pretrained=pretrained)
        
        # Replace ReLU with specified activation
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(model, name, act_fn)
        
        # Modify final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # VGG family
    elif model_name.startswith('vgg'):
        if model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_BN_Weights.IMAGENET1K_V1, pretrained=pretrained)
        elif model_name == 'vgg19':
            model = models.vgg19(weights=models.VGG19_BN_Weights.IMAGENET1K_V1, pretrained=pretrained)
        
        # Replace ReLU with specified activation
        for i, module in enumerate(model.features):
            if isinstance(module, nn.ReLU):
                model.features[i] = act_fn
        
        for i, module in enumerate(model.classifier):
            if isinstance(module, nn.ReLU):
                model.classifier[i] = act_fn
        
        # Modify classifier
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    # EfficientNet family
    elif model_name.startswith('efficientnet'):
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1, pretrained=pretrained)
        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1, pretrained=pretrained) # has Imagenet1K_V2 weights
        elif model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1, pretrained=pretrained)
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1, pretrained=pretrained)
        elif model_name == 'efficientnet_b4':
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1, pretrained=pretrained)
        elif model_name == 'efficientnet_b5':
            model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1, pretrained=pretrained)
        elif model_name == 'efficientnet_b6':
            model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1, pretrained=pretrained)
        elif model_name == 'efficientnet_b7':
            model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1, pretrained=pretrained)
        elif model_name == 'efficientnet_v2_s': 
            model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, pretrained=pretrained)
        elif model_name == 'efficientnet_v2_m':
            model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1, pretrained=pretrained)
        elif model_name == 'efficientnet_v2_l':
            model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1, pretrained=pretrained)

        # For EfficientNet, activation is built into the architecture
        # Modify classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # DenseNet family
    elif model_name.startswith('densenet'):
        if model_name == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
        elif model_name == 'densenet169':
            model = models.densenet169(pretrained=pretrained)
        elif model_name == 'densenet201':
            model = models.densenet201(pretrained=pretrained)
        
        # Replace activations
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(model, name, act_fn)
        
        # Modify classifier
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # MobileNet family
    elif model_name.startswith('mobilenet'):
        if model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
        elif model_name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(pretrained=pretrained)
        elif model_name == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(pretrained=pretrained)
        
        # Modify classifier
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    
    # ConvNeXt family
    elif model_name.startswith('convnext'):
        if model_name == 'convnext_tiny':
            model = models.convnext_tiny(pretrained=pretrained)
        elif model_name == 'convnext_small':
            model = models.convnext_small(pretrained=pretrained)
        elif model_name == 'convnext_base':
            model = models.convnext_base(pretrained=pretrained)
        
        # Modify classifier
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    
    # Vision Transformer family
    elif model_name.startswith('vit'):
        if model_name == 'vit_b_16':
            model = models.vit_b_16(pretrained=pretrained)
        elif model_name == 'vit_b_32':
            model = models.vit_b_32(pretrained=pretrained)
        elif model_name == 'vit_l_16':
            model = models.vit_l_16(pretrained=pretrained)
        
        # Modify head
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # Swin Transformer family
    elif model_name.startswith('swin'):
        if model_name == 'swin_t':
            model = models.swin_t(pretrained=pretrained)
        elif model_name == 'swin_s':
            model = models.swin_s(pretrained=pretrained)
        elif model_name == 'swin_b':
            model = models.swin_b(pretrained=pretrained)
        
        # Modify head
        model.head = nn.Linear(model.head.in_features, num_classes)
    
    # ResNeXt family
    elif model_name.startswith('resnext'):
        if model_name == 'resnext50_32x4d':
            model = models.resnext50_32x4d(pretrained=pretrained)
        elif model_name == 'resnext101_32x8d':
            model = models.resnext101_32x8d(pretrained=pretrained)
        
        # Replace activations
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(model, name, act_fn)
        
        # Modify fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model