#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance metrics calculation for post-tornado damage recognition experiments.
"""

import time
import torch
import numpy as np


def calculate_flops(model, input_size=(3, 224, 224), device="cuda"):
    """
    Calculate FLOPs for a PyTorch model.
    
    Args:
        model (nn.Module): Neural network model
        input_size (tuple): Input image dimensions (channels, height, width)
        device (str): Device to run calculation on ('cuda' or 'cpu')
    
    Returns:
        tuple: (flops, params) - Number of floating point operations and parameters
    """
    try:
        from thop import profile
        dummy_input = torch.randn(1, *input_size).to(device)
        flops, params = profile(model, inputs=(dummy_input,))
        return flops, params
    
    except ImportError:
        print("thop package not installed. Install with 'pip install thop' to calculate FLOPs.")
        # Rough estimate based on parameters
        print("Calculating FLOPs based on parameters only.")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Rough estimate: ~2 FLOPs per parameter per forward pass
        flops = params * 2
        return flops, params


def measure_inference_time(model, input_size=(3, 224, 224), device="cuda", num_runs=100):
    """
    Measure inference time for a PyTorch model.
    
    Args:
        model (nn.Module): Neural network model
        input_size (tuple): Input image dimensions (channels, height, width)
        device (str): Device to run measurement on ('cuda' or 'cpu')
        num_runs (int): Number of inference runs to average over
    
    Returns:
        float: Average inference time in milliseconds
    """
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Warm-up runs
    for _ in range(10):
        _ = model(dummy_input)
    
    # Measure timing
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)
    total_time = time.time() - start_time
    
    # Average time in milliseconds
    avg_time_ms = (total_time / num_runs) * 1000
    return avg_time_ms


def calculate_top_k_error(outputs, targets, k=1):
    """
    Calculate top-k error rate.
    
    Args:
        outputs (torch.Tensor): Model output logits
        targets (torch.Tensor): Target class indices
        k (int): k value for top-k error
    
    Returns:
        float: Top-k error rate
    """
    batch_size = targets.size(0)
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return (1.0 - correct_k.item() / batch_size)


def compute_per_class_metrics(conf_matrix):
    """
    Compute per-class precision, recall, and F1 score from confusion matrix.
    
    Args:
        conf_matrix (numpy.ndarray): Confusion matrix
    
    Returns:
        dict: Dictionary with per-class metrics
    """
    # Number of classes
    n_classes = conf_matrix.shape[0]
    
    # Initialize arrays for metrics
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)
    
    # Calculate metrics for each class
    for i in range(n_classes):
        # True positives
        tp = conf_matrix[i, i]
        
        # False positives (sum of column i - true positives)
        fp = np.sum(conf_matrix[:, i]) - tp
        
        # False negatives (sum of row i - true positives)
        fn = np.sum(conf_matrix[i, :]) - tp
        
        # Calculate precision: tp / (tp + fp)
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculate recall: tp / (tp + fn)
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }