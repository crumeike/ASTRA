#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ordinal error metrics for damage state classification.
These metrics account for the progressive nature of damage states.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def ordinal_top_k_accuracy(outputs, targets, k=2, ordinal_distance=1):
    """
    Calculate ordinal top-k accuracy where a prediction is considered correct
    if any of the top-k classes are within ordinal_distance of the true class.
    
    Args:
        outputs (torch.Tensor): Model output logits of shape (batch_size, num_classes)
        targets (torch.Tensor): Target class indices of shape (batch_size)
        k (int): Number of top predictions to consider
        ordinal_distance (int): Maximum acceptable distance between predicted and true class (e.g., ordinal_distance=1 means "off by at most one class")
    
    Returns:
        float: Ordinal top-k accuracy
    """
    if hasattr(targets, 'size') and callable(targets.size):
        batch_size = targets.size(0)  # PyTorch tensor
    elif hasattr(targets, 'shape'):
        batch_size = targets.shape[0]  # NumPy array
    else:
        batch_size = len(targets)  # List or other sequence
    
    # Get top-k predictions (indices)``
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()  # Shape: (k, batch_size)
    
    # Count correct predictions based on ordinal distance
    correct = 0
    for i in range(batch_size):
        target_class = targets[i].item()
        
        # Check if any of the top-k predictions are within ordinal_distance of the target
        for j in range(k):
            pred_class = pred[j][i].item()
            if abs(pred_class - target_class) <= ordinal_distance:
                correct += 1
                break
    
    return correct / batch_size


def ordinal_top_k_error(outputs, targets, k=2, ordinal_distance=1):
    """
    Calculate ordinal top-k error rate where a prediction is considered correct
    if any of the top-k classes are within ordinal_distance of the true class.
    
    Args:
        outputs (torch.Tensor): Model output logits of shape (batch_size, num_classes)
        targets (torch.Tensor): Target class indices of shape (batch_size)
        k (int): Number of top predictions to consider
        ordinal_distance (int): Maximum acceptable distance between predicted and true class
    
    Returns:
        float: Ordinal top-k error rate
    """
    return 1.0 - ordinal_top_k_accuracy(outputs, targets, k, ordinal_distance)


def get_default_damage_state_weights():
    """
    Get default weights for damage state error assessment.
    Weights represent error severity for predictions.
    Higher values indicate more severe errors.
    
    Returns:
        dict: Nested dictionary of damage state weights
    """
    # Format: damage_state_weights[true_class][pred_class] = weight
    # Higher weights = more severe errors
    return {
        # DS0 (Undamaged)
        0: {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5},
        
        # DS1 (Slight)
        1: {0: 1.0, 1: 0.0, 2: 0.5, 3: 1.0, 4: 1.5, 5: 2.0},
        
        # DS2 (Moderate)
        2: {0: 2.0, 1: 1.0, 2: 0.0, 3: 0.5, 4: 1.0, 5: 1.5},
        
        # DS3 (Extensive)
        3: {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0, 4: 0.5, 5: 1.0},
        
        # DS4 (Complete)
        4: {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0, 5: 0.5},
        
        # DS5 (Debris)
        5: {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.5, 4: 1.0, 5: 0.0}
    }


def weighted_ordinal_error(outputs, targets, damage_state_weights=None):
    """
    Calculate weighted ordinal error where mistakes are penalized based on 
    how far the prediction is from the true class, with custom weights for damage states.
    
    Args:
        outputs (torch.Tensor): Model output logits of shape (batch_size, num_classes)
        targets (torch.Tensor): Target class indices of shape (batch_size)
        damage_state_weights (dict, optional): Custom weight matrix for damage states
        
    Returns:
        float: Weighted ordinal error score (lower is better)
    """
    batch_size = targets.size(0)
    
    # Default weights if not provided
    if damage_state_weights is None:
        damage_state_weights = get_default_damage_state_weights()
    
    # Get predicted classes
    if isinstance(outputs, np.ndarray):
        outputs_tensor = torch.from_numpy(outputs)
    else:
        outputs_tensor = outputs
    _, predicted = torch.max(outputs_tensor, 1)
    
    # Calculate weighted error
    total_error = 0.0
    for i in range(batch_size):
        target_class = targets[i].item()
        pred_class = predicted[i].item()
        
        # Get error weight from the damage state weights matrix
        error_weight = damage_state_weights[target_class][pred_class]
        total_error += error_weight
    
    # Normalize by batch size
    return total_error / batch_size


def compute_ordinal_confusion_matrix(outputs, targets):
    """
    Generate a confusion matrix with counts for model evaluation.
    
    Args:
        outputs (torch.Tensor): Model output logits
        targets (torch.Tensor): Target class indices
        
    Returns:
        numpy.ndarray: Confusion matrix with raw counts
    """
    # Get predicted classes
    if isinstance(outputs, np.ndarray):
        outputs_tensor = torch.from_numpy(outputs)
    else:
        outputs_tensor = outputs
    _, predicted = torch.max(outputs_tensor, 1)
    
    # Convert to numpy arrays
    predicted = predicted.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Number of classes is the maximum class index plus one
    num_classes = max(predicted.max(), targets.max()) + 1
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(
        targets, 
        predicted, 
        labels=range(num_classes)
    )
    
    return conf_matrix


def evaluate_damage_state_predictions(outputs, targets, damage_state_names=None):
    """
    Comprehensive evaluation of damage state predictions accounting for ordinal relationships.
    
    Args:
        outputs (torch.Tensor): Model output logits of shape (batch_size, num_classes)
        targets (torch.Tensor): Target class indices of shape (batch_size)
        damage_state_names (list, optional): Names of damage states for reporting
        
    Returns:
        dict: Dictionary with various ordinal metrics
    """
    if damage_state_names is None:
        damage_state_names = ["DS0-Undamaged", "DS1-Slight", "DS2-Moderate", 
                             "DS3-Extensive", "DS4-Complete", "DS5-Debris"]
    
    num_classes = len(damage_state_names)
    if hasattr(targets, 'size') and callable(targets.size):
        # print("Targets is a PyTorch tensor")
        # Get batch size from PyTorch tensor
        batch_size = targets.size(0)  # PyTorch tensor
    elif hasattr(targets, 'shape'):
        # print("Targets is a NumPy array")
        # Get batch size from NumPy array
        batch_size = targets.shape[0]  # NumPy array
        targets = torch.from_numpy(targets) # Convert to PyTorch tensor if needed
    else:
        # print("Targets is a list or other sequence")
        # Get batch size from list or other sequence
        batch_size = len(targets)  # List or other sequence
    
    # Get predicted classes
    if isinstance(outputs, np.ndarray):
        outputs = torch.from_numpy(outputs)

    _, predicted = torch.max(outputs, 1)


    # Standard accuracy
    if isinstance(predicted, np.ndarray) and isinstance(targets, np.ndarray):
        # NumPy implementation
        standard_accuracy = np.mean(predicted == targets)
    elif isinstance(predicted, (int, float)) or isinstance(targets, (int, float)):
        # Handle scalar case
        standard_accuracy = float(predicted == targets)  # Will be 1.0 if equal, 0.0 if not
    else:
        # PyTorch implementation
        standard_accuracy = (predicted == targets).float().mean().item()
    
    # Top-2 accuracy (traditional)
    _, top2_preds = outputs.topk(2, 1, True, True)

    top2_correct = 0
    for i in range(batch_size):
        if targets[i] in top2_preds[i]:
            top2_correct += 1
    top2_accuracy = top2_correct / batch_size
    
    # Ordinal top-1 and top-2 accuracy (allowing off-by-one)
    ordinal_top1_acc = ordinal_top_k_accuracy(outputs, targets, k=1, ordinal_distance=1)
    ordinal_top2_acc = ordinal_top_k_accuracy(outputs, targets, k=2, ordinal_distance=1)
    
    # Weighte0d ordinal error using the damage state weights
    weighted_error = weighted_ordinal_error(outputs, targets)
    
    # Calculate ordinal metrics for each error distance
    error_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for i in range(batch_size):
        target_class = targets[i].item()
        pred_class = predicted[i].item()
        distance = abs(pred_class - target_class)
        
        if distance <= 5:  # Max distance is 5 (DS0 to DS5)
            error_counts[distance] += 1
    
    # Calculate percentages
    error_percentages = {
        f"off_by_{i}_percent": (count / batch_size) * 100 
        for i, count in error_counts.items()
    }
    
    # Confusion matrix
    conf_matrix = compute_ordinal_confusion_matrix(outputs, targets)
    
    # Get class-wise accuracy
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    for i in range(batch_size):
        target_class = targets[i].item()
        if target_class < num_classes:  # Ensure we don't go out of bounds
            if predicted[i].item() == target_class:
                class_correct[target_class] += 1
            class_total[target_class] += 1
    
    class_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[damage_state_names[i]] = class_correct[i] / class_total[i]
        else:
            class_accuracy[damage_state_names[i]] = 0.0
    
    # Return comprehensive metrics
    return {
        "standard_top1_accuracy": standard_accuracy,
        "standard_top2_accuracy": top2_accuracy,
        "ordinal_top1_accuracy": ordinal_top1_acc,
        "ordinal_top2_accuracy": ordinal_top2_acc,
        "weighted_ordinal_error": weighted_error,
        "confusion_matrix": conf_matrix,
        "standard_class_accuracy": class_accuracy,
        **error_percentages  # Unpack error percentages
    }


def plot_ordinal_confusion_matrix(conf_matrix, class_names, figsize=(12, 10), output_path=None):
    """
    Plot a color-coded confusion matrix that highlights ordinal errors differently.
    
    Args:
        conf_matrix (numpy.ndarray): Standard confusion matrix
        class_names (list): List of class names
        figsize (tuple): Figure size (width, height)
        output_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    # Create a normalized version for coloring
    with np.errstate(divide='ignore', invalid='ignore'):
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Replace NaNs with zeros
    
    # Create ordinal distance matrix for coloring
    num_classes = len(class_names)
    ordinal_dist_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            ordinal_dist_matrix[i, j] = abs(i - j) / (num_classes - 1)
    '''
        Example ordinal distance matrix for 6 classes (0-5):
        0 = Undamaged, 1 = Slight, 2 = Moderate, 3 = Extensive, 4 = Complete, 5 = Debris
        The Ordinal distance matrix would look like this:
    
            [[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            [0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
            [0.4, 0.2, 0.0, 0.2, 0.4, 0.6],
            [0.6, 0.4, 0.2, 0.0, 0.2, 0.4],
            [0.8, 0.6, 0.4, 0.2, 0.0, 0.2],
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]]

    '''
    # Create RdYlGn_r colormap for the heatmap (red=far, green=close)
    cmap = plt.cm.get_cmap('RdYlGn_r')
    
    # Plot the heatmap with ordinal distance colors
    ax = plt.gca()
    im = ax.imshow(ordinal_dist_matrix, cmap=cmap, vmin=0, vmax=1)
    
    # Add annotations with raw counts
    for i in range(num_classes):
        for j in range(num_classes):
            # Use white text for dark background, black for light
            text_color = 'white' if ordinal_dist_matrix[i, j] > 0.5 else 'black'
            ax.text(j, i, str(conf_matrix[i, j]), 
                   ha="center", va="center", color=text_color, fontsize=12, fontweight='bold')
    
    # Configure axes
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add labels and title
    plt.ylabel('True Damage State')
    plt.xlabel('Predicted Damage State')
    plt.title('Ordinal Confusion Matrix for Damage States')
    
    # Add a colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Ordinal Distance (Normalized)')
    
    # Create legend for ordinal distances
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap(0.0), label='Correct'),
        Patch(facecolor=cmap(0.2), label='Off by 1'),
        Patch(facecolor=cmap(0.4), label='Off by 2'),
        Patch(facecolor=cmap(0.6), label='Off by 3'),
        Patch(facecolor=cmap(0.8), label='Off by 4'),
        Patch(facecolor=cmap(1.0), label='Off by 5')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1), loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_distribution(metrics, figsize=(10, 6), output_path=None):
    """
    Plot the distribution of ordinal prediction errors.
    
    Args:
        metrics (dict): Metrics dictionary from evaluate_damage_state_predictions
        figsize (tuple): Figure size
        output_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    # Extract error percentages
    error_keys = [key for key in metrics.keys() if key.startswith('off_by_') and key.endswith('_percent')]
    error_keys.sort(key=lambda x: int(x.split('_')[2]))  # Sort by error distance
    
    error_distances = [int(key.split('_')[2]) for key in error_keys]
    error_values = [metrics[key] for key in error_keys]
    
    # Create bar chart
    bars = plt.bar(error_distances, error_values, color='skyblue', alpha=0.7, width=0.6)
    
    # Color the first bar (correct predictions) differently
    if len(bars) > 0:
        bars[0].set_color('green')
    
    # Add annotations above bars
    for i, v in enumerate(error_values):
        plt.text(error_distances[i], v + 1, f"{v:.1f}%", ha='center')
    
    plt.xlabel('Error Distance (Ordinal Steps)')
    plt.ylabel('Percentage of Predictions (%)')
    plt.title('Distribution of Ordinal Prediction Errors')
    plt.xticks(error_distances)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a note explaining the interpretation
    plt.figtext(0.5, 0.01, 
                "Error distance 0 = correct predictions\n"
                "Error distance 1 = predictions off by one damage state, etc.",
                ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()