#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Class Activation Mapping utilities for post-tornado damage recognition experiments.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization.
    
    Paper: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/abs/1610.02391
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model (nn.Module): Neural network model
            target_layer: Layer to compute CAM for (usually last convolutional layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        
        # Forward hook
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Backward hook
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        
        # Store handles for removal
        self.hooks = [forward_handle, backward_handle]
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
    
    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, channels, height, width)
            class_idx (int, optional): Class index to generate CAM for.
                                      If None, uses the predicted class.
        
        Returns:
            numpy.ndarray: Grad-CAM heatmap (batch_size, height, width)
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        # If class_idx is None, use the model's prediction
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        
        # One-hot encode target
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, class_idx.view(-1, 1), 1.0)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to focus on positive influence
        
        # Normalize CAM
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


def visualize_cam(image, cam, output_path=None, alpha=0.5):
    """
    Visualize Grad-CAM heatmap overlaid on image.
    
    Args:
        image (torch.Tensor or numpy.ndarray): Input image
        cam (numpy.ndarray): Grad-CAM heatmap
        output_path (str, optional): Path to save visualization. If None, the image is displayed.
        alpha (float): Transparency level for heatmap overlay (0-1)
    
    Returns:
        numpy.ndarray: Visualization image with CAM overlay
    """
    # Convert torch tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Reverse normalization (approximation)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Create visualization
    visualization = (1 - alpha) * image + alpha * heatmap
    visualization = np.clip(visualization, 0, 1)
    
    # Convert to displayable format
    visualization = (visualization * 255).astype(np.uint8)
    
    # Save or display visualization
    if output_path:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(visualization)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    return visualization


class GradCAMpp(GradCAM):
    """
    Grad-CAM++ implementation for CNN visualization.
    
    Paper: "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
    https://arxiv.org/abs/1710.11063
    """
    
    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, channels, height, width)
            class_idx (int, optional): Class index to generate CAM for.
                                      If None, uses the predicted class.
        
        Returns:
            numpy.ndarray: Grad-CAM++ heatmap (batch_size, height, width)
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        # If class_idx is None, use the model's prediction
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        
        # One-hot encode target
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, class_idx.view(-1, 1), 1.0)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Calculate alpha_k^c for each pixel
        eps = 1e-8
        alpha_numerator = gradients.pow(2)
        alpha_denominator = 2 * gradients.pow(2) + (activations * gradients.pow(3)).sum(dim=[2, 3], keepdim=True)
        alpha = alpha_numerator / (alpha_denominator + eps)
        
        # Weight the activations
        weights = (alpha * torch.relu(gradients)).sum(dim=[2, 3], keepdim=True)
        
        # Generate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to focus on positive influence
        
        # Normalize CAM
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + eps)
        
        return cam.squeeze().cpu().numpy()


def generate_cam_visualization(model, layer, images, labels, class_names, output_dir, method='gradcam'):
    """
    Generate CAM visualizations for a batch of images.
    
    Args:
        model (nn.Module): Neural network model
        layer: Target layer for CAM (usually last convolutional layer)
        images (torch.Tensor): Batch of images (batch_size, channels, height, width)
        labels (torch.Tensor): Ground truth labels
        class_names (list): List of class names
        output_dir (str): Directory to save visualizations
        method (str): CAM method ('gradcam' or 'gradcam++')
    """
    # Create CAM instance
    if method.lower() == 'gradcam':
        cam_extractor = GradCAM(model, layer)
    elif method.lower() == 'gradcam++':
        cam_extractor = GradCAMpp(model, layer)
    else:
        raise ValueError(f"Unsupported CAM method: {method}")
    
    # Ensure output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for i in range(images.shape[0]):
        # Extract single image
        img = images[i:i+1]
        true_label = labels[i].item()
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            predicted_label = predicted.item()
        
        # Generate CAM for predicted class
        cam = cam_extractor(img, class_idx=predicted)
        
        # Create file name with true and predicted labels
        file_name = f"cam_{i}_true-{class_names[true_label]}_pred-{class_names[predicted_label]}.png"
        output_path = os.path.join(output_dir, file_name)
        
        # Visualize and save
        visualize_cam(img, cam, output_path)
    
    # Remove hooks
    cam_extractor.remove_hooks()


def find_target_layer(model):
    """
    Automatically find the last convolutional layer in the model.
    
    Args:
        model (nn.Module): Neural network model
    
    Returns:
        nn.Module: Last convolutional layer
    """
    import torch.nn as nn
    
    # For ResNet models
    if hasattr(model, 'layer4') and hasattr(model.layer4, '1') and hasattr(model.layer4[1], 'conv2'):
        return model.layer4[1].conv2
    
    # For VGG models
    elif hasattr(model, 'features'):
        for i in range(len(model.features) - 1, -1, -1):
            if isinstance(model.features[i], nn.Conv2d):
                return model.features[i]
    
    # For EfficientNet models
    elif hasattr(model, 'features') and hasattr(model.features, 'blocks'):
        return model.features[-1]
    
    # For other models, find the last conv layer
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    
    if last_conv is not None:
        return last_conv
    else:
        raise ValueError("Could not find a suitable target layer for CAM visualization")