#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script showing how to use ordinal metrics in the training and evaluation pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

# Import from the framework
from data_utils import get_data_loaders
from models import create_model
from utils import set_seed
from ordinal_metrics import (
    ordinal_top_k_accuracy,
    ordinal_top_k_error,
    weighted_ordinal_error,
    evaluate_damage_state_predictions,
    plot_ordinal_confusion_matrix,
    plot_error_distribution,
    get_default_damage_state_weights
)

# Set random seed for reproducibility
set_seed(24)

# Define damage state names including all six categories
DAMAGE_STATE_NAMES = ["DS0-Undamaged", "DS1-Slight", "DS2-Moderate", 
                      "DS3-Extensive", "DS4-Complete", "DS5-Debris"]

def main():
    # Configuration
    data_dir = "./data"
    model_name = "resnet50"
    batch_size = 128
    input_size = 224
    num_epochs = 10
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"./ordinal_results_{timestamp}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader, _, class_names = get_data_loaders(
        data_dir=data_dir,
        input_size=input_size,
        batch_size=batch_size,
        data_augmentation='standard'
    )
    
    num_classes = len(class_names)
    if num_classes != len(DAMAGE_STATE_NAMES):
        print(f"Warning: Expected {len(DAMAGE_STATE_NAMES)} classes but found {num_classes}.")
        print(f"Using class names from dataset: {class_names}")
        class_labels = class_names
    else:
        class_labels = DAMAGE_STATE_NAMES
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True
    )
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"Training {model_name} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_acc:.4f}")
    
    print("\nEvaluating model with ordinal metrics...")
    
    # Evaluate with ordinal metrics
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            all_outputs.append(outputs)
            all_targets.append(labels)
    
    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, 0)
    all_targets = torch.cat(all_targets, 0)
    
    # Calculate standard metrics
    _, predicted = all_outputs.max(1)
    standard_accuracy = (predicted == all_targets).sum().item() / all_targets.size(0)
    
    # Calculate traditional top-k metrics
    print("\nTraditional Metrics:")
    print(f"Standard Accuracy: {standard_accuracy:.4f}")
    
    # Calculate ordinal metrics
    print("\nOrdinal Metrics:")
    for k in [1, 2]:
        for dist in [0, 1]:
            acc = ordinal_top_k_accuracy(all_outputs, all_targets, k=k, ordinal_distance=dist)
            err = ordinal_top_k_error(all_outputs, all_targets, k=k, ordinal_distance=dist)
            print(f"Ordinal Top-{k} Accuracy (off by ±{dist} class): {acc:.4f}")
            # print(f"Ordinal Top-{k} Error (off by ±{dist} class): {err:.4f}")
    
    # Calculate weighted ordinal error
    w_error = weighted_ordinal_error(all_outputs, all_targets)
    print(f"\nWeighted Ordinal Error: {w_error:.4f}")
    
    # Get default damage state weights
    damage_weights = get_default_damage_state_weights()
    print("\nDamage State Weights (For reference - higher values = more severe errors):")
    for true_class in range(len(class_labels)):
        if true_class >= 6:  # Skip if out of bounds
            continue
        weights = []
        for pred_class in range(len(class_labels)):
            if pred_class >= 6:  # Skip if out of bounds
                continue
            weights.append(f"{damage_weights[true_class][pred_class]:.1f}")
        print(f"True={class_labels[true_class]}: [{', '.join(weights)}]")
    
    # Comprehensive evaluation
    print("\nRunning comprehensive damage state evaluation...")
    damage_metrics = evaluate_damage_state_predictions(
        all_outputs, all_targets, class_labels
    )
    
    # Print error distribution
    print("\nOrdinal Error Distribution:")
    for key in sorted([k for k in damage_metrics.keys() if k.startswith('off_by_') and k.endswith('_percent')]):
        error_level = key.split('_')[2]
        print(f"Off by {error_level}: {damage_metrics[key]:.2f}%")
    
    # Class-wise accuracy
    print("\nClass-wise Accuracy:")
    for class_name, acc in damage_metrics['class_accuracy'].items():
        print(f"{class_name}: {acc:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot ordinal confusion matrix
    conf_matrix_path = os.path.join(output_dir, "ordinal_confusion_matrix.png")
    plot_ordinal_confusion_matrix(
        damage_metrics['confusion_matrix'],
        class_labels,
        output_path=conf_matrix_path
    )
    
    # Plot error distribution
    error_dist_path = os.path.join(output_dir, "error_distribution.png")
    plot_error_distribution(
        damage_metrics,
        output_path=error_dist_path
    )
    
    # Create a synthetic confusion matrix for demonstration if the real one is too sparse
    if np.count_nonzero(damage_metrics['confusion_matrix']) < 15:
        print("\nCreating synthetic confusion matrix for demonstration...")
        
        synthetic_conf_matrix = np.array([
            [50, 5, 2, 0, 0, 0],
            [8, 40, 7, 1, 0, 0],
            [3, 10, 35, 6, 2, 0],
            [1, 2, 8, 30, 5, 1],
            [0, 1, 3, 7, 25, 4],
            [0, 0, 1, 2, 3, 20]
        ])
        
        synthetic_path = os.path.join(output_dir, "synthetic_ordinal_confusion_matrix.png")
        plot_ordinal_confusion_matrix(
            synthetic_conf_matrix,
            DAMAGE_STATE_NAMES,
            output_path=synthetic_path
        )
        
        # Calculate error distribution from synthetic matrix
        synthetic_metrics = {}
        total = synthetic_conf_matrix.sum()
        
        # Count correct and off-by-n predictions
        for n in range(6):
            count = 0
            for i in range(6):
                for j in range(6):
                    if abs(i - j) == n:
                        count += synthetic_conf_matrix[i, j]
            synthetic_metrics[f'off_by_{n}_percent'] = (count / total) * 100
        
        synthetic_error_path = os.path.join(output_dir, "synthetic_error_distribution.png")
        plot_error_distribution(
            synthetic_metrics,
            output_path=synthetic_error_path
        )
        
        print(f"- Synthetic Confusion Matrix: {synthetic_path}")
        print(f"- Synthetic Error Distribution: {synthetic_error_path}")
    
    print("\nVisualization files saved:")
    print(f"- Confusion Matrix: {conf_matrix_path}")
    print(f"- Error Distribution: {error_dist_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()