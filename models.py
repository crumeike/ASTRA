#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model architecture definitions with dropout support for post-tornado damage recognition experiments.
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


def apply_dropout(model, dropout_rate=0.0, dropout_type='standard'):
    """
    Apply dropout to specific layers in the model.
    
    Args:
        model (nn.Module): Neural network model
        dropout_rate (float): Dropout probability (0.0 to 1.0)
        dropout_type (str): Type of dropout ('standard', 'spatial', 'feature')
    
    Returns:
        nn.Module: Model with dropout applied
    """
    if dropout_rate <= 0.0:
        return model  # No dropout to apply
    
    # ResNet, VGG, EfficientNet support
    if hasattr(model, 'classifier'):
        # For VGG, DenseNet, MobileNet, EfficientNet, etc.
        if isinstance(model.classifier, nn.Sequential):
            # Find fully connected layers and add dropout
            for i in range(len(model.classifier) - 1):
                if isinstance(model.classifier[i], nn.Linear) and i < len(model.classifier) - 1:
                    # Add dropout after FC layer but before the final layer
                    new_classifier = list(model.classifier.children())
                    new_classifier.insert(i + 1, nn.Dropout(dropout_rate))
                    model.classifier = nn.Sequential(*new_classifier)
                    break
        elif isinstance(model.classifier, nn.Linear):
            # For simpler models with a single classifier layer
            fc_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(fc_features, model.classifier.out_features)
            )
    
    # For ResNet models
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        fc_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fc_features, model.fc.out_features)
        )
    
    # For Vision Transformers
    if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        head_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(head_features, model.heads.head.out_features)
        )
    
    # Spatial dropout for convolutional layers (if requested)
    if dropout_type in ['spatial', 'feature'] and hasattr(model, 'features'):
        # Apply dropout after certain convolutional blocks
        feature_modules = list(model.features.children())
        modified_features = []
        
        for i, module in enumerate(feature_modules):
            modified_features.append(module)
            
            # Add dropout after convolutional blocks (but not too frequently)
            if (i > 0 and i % 3 == 0 and i < len(feature_modules) - 1):
                if dropout_type == 'spatial':
                    # Spatial dropout (drops entire feature maps)
                    if hasattr(module, 'out_channels'):
                        modified_features.append(nn.Dropout2d(dropout_rate/2))  # Use lower rate for spatial
                elif dropout_type == 'feature':
                    # Feature dropout (drops individual features)
                    modified_features.append(nn.Dropout(dropout_rate/2))
        
        model.features = nn.Sequential(*modified_features)
    
    return model


def create_model(model_name, num_classes, pretrained=True, activation='relu', dropout_rate=0.0, dropout_type='standard'):
    """
    Create a model with the specified architecture and dropout.
    
    Args:
        model_name (str): Model architecture name
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        activation (str): Activation function name
        dropout_rate (float): Dropout probability (0.0 to 1.0)
        dropout_type (str): Type of dropout ('standard', 'spatial', 'feature')
    
    Returns:
        nn.Module: Neural network model
    """
    # Get activation function
    act_fn = get_activation_function(activation)

    # ResNet family
    if model_name.startswith('resnet'):
        if model_name == 'resnet18':
            print("Using ResNet18...")
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet34':
            print("Using ResNet34...")
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet50' or model_name == 'resnet50_baseline':
            print("Using ResNet50...")
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet50v2':
            print("Using ResNet50v2...")
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif model_name == 'resnet101':
            print("Using ResNet101...")
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet101v2':
            print("Using ResNet101v2...")
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)              
        elif model_name == 'resnet152':
            print("Using ResNet152...")
            model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet152v2':
            print("Using ResNet152v2...")
            model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)

        # Replace ReLU with specified activation
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.ReLU):
                setattr(model, name, act_fn)
        
        # Modify final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # VGG family
    elif model_name.startswith('vgg'):
        if model_name == 'vgg16' or model_name == 'vgg16_baseline':
            print("Using VGG16...")
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg16_bn':
            print("Using VGG16 with batch normalization...")
            model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg19':
            print("Using VGG19...")
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg19_bn':
            print("Using VGG19 with batch normalization...")
            model = models.vgg19(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg11':
            print("Using VGG11...")
            model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg11_bn':
            print("Using VGG11 with batch normalization...")
            model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg13':
            print("Using VGG13...")
            model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg13_bn':
            print("Using VGG13 with batch normalization...")
            model = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1)
        
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
            print("Using EfficientNet-B0...")
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b1':
            print("Using EfficientNet-B1...")   
            # EfficientNet-B1 has two sets of weights: IMAGENET1K_V1 and IMAGENET1K_V2
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1,) # has Imagenet1K_V2 weights
        elif model_name == 'efficientnet_b1v2':
            print("Using EfficientNet-B1v2...")
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2)
        elif model_name == 'efficientnet_b2':
            print("Using EfficientNet-B2...")
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b3' or model_name == 'efficientnet_b3_baseline':
            print("Using EfficientNet-B3...")
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b4':
            print("Using EfficientNet-B4...")
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b5':
            print("Using EfficientNet-B5...")
            model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b6':
            print("Using EfficientNet-B6...")
            model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_b7':
            print("Using EfficientNet-B7...")
            model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_v2_s':
            print("Using EfficientNet-V2-S...")
            model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_v2_m':
            print("Using EfficientNet-V2-M...")
            model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        elif model_name == 'efficientnet_v2_l':
            print("Using EfficientNet-V2-L...")
            model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)

        # For EfficientNet, activation is built into the architecture
        # Modify classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # DenseNet family
    elif model_name.startswith('densenet'):
        if model_name == 'densenet121':
            print("Using DenseNet121...")
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        elif model_name == 'densenet169':
            print("Using DenseNet169...")
            model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        elif model_name == 'densenet201':
            print("Using DenseNet201...")
            model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        
        # Replace activations
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(model, name, act_fn)
        
        # Modify classifier
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # MobileNet family
    elif model_name.startswith('mobilenet'):
        if model_name == 'mobilenet_v2':
            print("Using MobileNetV2...")
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif model_name == 'mobilenet_v3_small':
            print("Using MobileNetV3-Small...")
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        elif model_name == 'mobilenet_v3_large':
            print("Using MobileNetV3-Large...")
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # Modify classifier
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    # ShuffleNet family
    elif model_name.startswith('shufflenet'):
        if model_name == 'shufflenet_v2_x0_5':
            print("Using ShuffleNetV2 x0.5...")
            model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
        elif model_name == 'shufflenet_v2_x1_0':
            print("Using ShuffleNetV2 x1.0...")
            model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        elif model_name == 'shufflenet_v2_x1_5':
            print("Using ShuffleNetV2 x1.5...")
            model = models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1)
        elif model_name == 'shufflenet_v2_x2_0':
            print("Using ShuffleNetV2 x2.0...")
            model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
        
        # Replace activations
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(model, name, act_fn)
                
        # Modify classifier
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    #RegNet family
    elif model_name.startswith('regnet'):
        if model_name == 'regnet_x_400mf':
            print("Using RegNetX-400MF...")
            model = models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.IMAGENET1K_V1)
        elif model_name == 'regnet_x_800mf':
            print("Using RegNetX-800MF...")
            model = models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V1)
        elif model_name == 'regnet_x_1_6gf':
            print("Using RegNetX-1.6GF...")
            model = models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V1)
        elif model_name == 'regnet_x_3_2gf':
            print("Using RegNetX-3.2GF...")
            model = models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.IMAGENET1K_V1)
        elif model_name == 'regnet_x_8gf':
            print("Using RegNetX-8GF...")
            model = models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.IMAGENET1K_V1)
        elif model_name == 'regnet_x_16gf':
            print("Using RegNetX-16GF...")
            model = models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.IMAGENET1K_V1)
        elif model_name == 'regnet_y_400mf':
            print("Using RegNetY-400MF...")
            model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1)
        elif model_name == 'regnet_y_800mf':
            print("Using RegNetY-800MF...")
            model = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V1)
        elif model_name == 'regnet_y_8gf':
            print("Using RegNetY-8GF...")
            model = models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V1)
        elif model_name == 'regnet_y_16gf':
            print("Using RegNetY-16GF...")
            model = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V1)
        elif model_name == 'regnet_y_32gf':
            print("Using RegNetY-32GF...")
            model = models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_V1)
        elif model_name == 'regnet_y_128gf':
            print("Using RegNetY-128GF...")
            model = models.regnet_y_128gf(weights=models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1)


        # Replace activations
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(model, name, act_fn)
        # Modify classifier
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    
    # ConvNeXt family
    elif model_name.startswith('convnext'):
        if model_name == 'convnext_tiny':
            print("Using ConvNeXt-Tiny...")
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        elif model_name == 'convnext_small':
            print("Using ConvNeXt-Small...")
            model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        elif model_name == 'convnext_base':
            print("Using ConvNeXt-Base...")
            model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        elif model_name == 'convnext_large':
            print("Using ConvNeXt-Large...")
            model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)

        # Replace activations
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(model, name, act_fn)
                
        # Modify classifier
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    
    # Vision Transformer family
    elif model_name.startswith('vit'):
        if model_name == 'vit_b_16':
            print("Using ViT-B/16...")
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif model_name == 'vit_b_32':
            print("Using ViT-B/32...")
            model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        elif model_name == 'vit_l_16':
            print("Using ViT-L/16...")
            model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        elif model_name == 'vit_l_32':
            print("Using ViT-L/32...")
            model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
        elif model_name == 'vit_h_14_e2e':
            print("Using ViT-H/14 with end-to-end training...")
            model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
        elif model_name == 'vit_h_14_linear':
            print("Using ViT-H/14 with linear head...") 
            model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)

        
        # Modify head
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # Swin Transformer family
    elif model_name.startswith('swin'):
        if model_name == 'swin_t':
            print("Using Swin-T...")
            model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        elif model_name == 'swin_s':
            print("Using Swin-S...")
            model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)  
        elif model_name == 'swin_b':
            print("Using Swin-B...")
            model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        elif model_name == 'swin_v2_b':
            print("Using Swin-V2-B...")
            model = models.swin_l(weights=models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1))
        
        # Modify head
        model.head = nn.Linear(model.head.in_features, num_classes)
    
    # ResNeXt family
    elif model_name.startswith('resnext'):
        if model_name == 'resnext50_32x4d':
            print("Using ResNeXt50 32x4d...")
            model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        elif model_name == 'resnext101_32x8d':
            print("Using ResNeXt101 32x8d...")
            model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        elif model_name == 'resnext101_64x4d':
            print("Using ResNeXt101 64x4d...")
            model = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
        
        # Replace activations
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(model, name, act_fn)
        
        # Modify fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    

    # Apply dropout to the model
    if dropout_rate > 0.0:
        print(f"Applying {dropout_type} dropout with rate {dropout_rate}...")
        model = apply_dropout(model, dropout_rate, dropout_type)
    
    return model