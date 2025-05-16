#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration of ordinal metrics into the main training pipeline.
This file demonstrates how to modify the training.py module to include ordinal metrics.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, classification_report

# Import ordinal metrics
from ordinal_metrics import (
    ordinal_top_k_accuracy,
    evaluate_damage_state_predictions,
    plot_ordinal_confusion_matrix,
    plot_error_distribution
)

# Define damage state names
DAMAGE_STATE_NAMES = ["DS0-Undamaged", "DS1-Slight", "DS2-Moderate", 
                      "DS3-Extensive", "DS4-Complete", "DS5-Debris"]


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch (unchanged).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
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
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch_with_ordinal_metrics(model, val_loader, criterion, device, num_classes, class_names=None):
    """
    Validate model on validation set with ordinal metrics.
    
    Args:
        model (nn.Module): Neural network model
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device (cuda or cpu)
        num_classes (int): Number of output classes
        class_names (list, optional): Names of classes
    
    Returns:
        dict: Dictionary with metrics
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_outputs = []
    
    # Use dataset class names if not provided
    if class_names is None:
        if num_classes == len(DAMAGE_STATE_NAMES):
            class_names = DAMAGE_STATE_NAMES
        else:
            class_names = [f"Class-{i}" for i in range(num_classes)]
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Store predictions, outputs, and targets
            _, predicted = outputs.max(1)
            all_predictions.append(predicted.cpu())
            all_targets.append(labels.cpu())
            all_outputs.append(outputs.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, 0)
    all_targets = torch.cat(all_targets, 0)
    all_outputs = torch.cat(all_outputs, 0)
    
    # Calculate standard metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    standard_accuracy = (all_predictions == all_targets).float().mean().item()
    
    # Calculate precision, recall, and F1 score (weighted by support)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets.numpy(), all_predictions.numpy(), average='weighted'
    )
    
    # Calculate standard top-k errors
    _, top2_preds = all_outputs.topk(2, 1, True, True)
    top2_correct = 0
    for i in range(len(all_targets)):
        if all_targets[i] in top2_preds[i]:
            top2_correct += 1
    top2_accuracy = top2_correct / len(all_targets)
    
    top1_error = 1.0 - standard_accuracy
    top2_error = 1.0 - top2_accuracy
    
    # Calculate ordinal metrics
    ordinal_top1_accuracy = ordinal_top_k_accuracy(all_outputs, all_targets, k=1, ordinal_distance=1)
    ordinal_top2_accuracy = ordinal_top_k_accuracy(all_outputs, all_targets, k=2, ordinal_distance=1)
    
    # Compute standard confusion matrix
    conf_matrix = confusion_matrix(
        all_targets.numpy(), all_predictions.numpy(), labels=range(num_classes)
    )
    
    # Compute ordinal metrics
    ordinal_metrics = evaluate_damage_state_predictions(all_outputs, all_targets, class_names)
    
    # Combine all metrics
    metrics = {
        'val_loss': epoch_loss,
        'val_acc': standard_accuracy,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1,
        'val_top1_error': top1_error,
        'val_top2_error': top2_error,
        'confusion_matrix': conf_matrix,
        'ordinal_top1_accuracy': ordinal_top1_accuracy,
        'ordinal_top2_accuracy': ordinal_top2_accuracy,
        'classification_report': classification_report(
            all_targets.numpy(), all_predictions.numpy(), output_dict=True
        )
    }
    
    # Add ordinal metrics
    metrics.update({
        f"ordinal_{k}": v for k, v in ordinal_metrics.items() 
        if k not in ['confusion_matrix', 'class_accuracy']
    })
    
    return metrics


def train_model_with_ordinal_metrics(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
    num_classes, num_epochs, patience=10, experiment_tracker=None, class_names=None
):
    """
    Train and validate model with ordinal metrics.
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device (cuda or cpu)
        num_classes (int): Number of output classes
        num_epochs (int): Maximum number of epochs to train
        patience (int): Early stopping patience
        experiment_tracker (ExperimentTracker, optional): Experiment tracker
        class_names (list, optional): Names of classes
    
    Returns:
        tuple: (best_model, best_metrics)
    """
    # Initialize variables for early stopping
    best_epoch = 0
    best_val_f1 = 0.0
    no_improve_epochs = 0
    
    # Use dataset class names if not provided
    if class_names is None:
        if num_classes == len(DAMAGE_STATE_NAMES):
            class_names = DAMAGE_STATE_NAMES
        else:
            class_names = [f"Class-{i}" for i in range(num_classes)]
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase with ordinal metrics
        metrics = validate_epoch_with_ordinal_metrics(
            model, val_loader, criterion, device, num_classes, class_names
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics['val_loss'])
            else:
                scheduler.step()
        
        # Update metrics in tracker if provided
        is_best = False
        if experiment_tracker is not None:
            # Add ordinal metrics to the tracker
            is_best = experiment_tracker.update_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=metrics['val_loss'],
                val_acc=metrics['val_acc'],
                val_precision=metrics['val_precision'],
                val_recall=metrics['val_recall'],
                val_f1=metrics['val_f1'],
                val_top1_error=metrics['val_top1_error'],
                val_top2_error=metrics['val_top2_error'],
                ordinal_top1_accuracy=metrics['ordinal_top1_accuracy'],
                ordinal_top2_accuracy=metrics['ordinal_top2_accuracy'],
                learning_rate=current_lr
            )
            
            # Save ordinal metrics visualizations
            if epoch % 5 == 0 or epoch == num_epochs - 1:  # Every 5 epochs and last epoch
                plots_dir = os.path.join(experiment_tracker.save_dir, 'plots')
                
                # Plot ordinal confusion matrix
                plot_ordinal_confusion_matrix(
                    metrics['confusion_matrix'],
                    class_names,
                    output_path=os.path.join(plots_dir, f'ordinal_confusion_matrix_epoch_{epoch}.png')
                )
                
                # Plot error distribution
                plot_error_distribution(
                    metrics,
                    output_path=os.path.join(plots_dir, f'error_distribution_epoch_{epoch}.png')
                )
            
            # Save model checkpoint
            experiment_tracker.save_model(model, optimizer, epoch, is_best=is_best)
        else:
            # If no tracker is provided, use F1 score to determine best model
            is_best = metrics['val_f1'] > best_val_f1
            if is_best:
                best_val_f1 = metrics['val_f1']
                best_metrics = metrics
                best_model_state = model.state_dict()
                best_optimizer_state = optimizer.state_dict()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {metrics['val_loss']:.4f} | Val Acc: {metrics['val_acc']:.4f} | F1: {metrics['val_f1']:.4f}")
        
        # Also print ordinal metrics
        print(f"Ordinal Top-1 Acc (off by ≤1): {metrics['ordinal_top1_accuracy']:.4f} | "
              f"Ordinal Top-2 Acc (off by ≤1): {metrics['ordinal_top2_accuracy']:.4f}")
        
        # Early stopping check
        if is_best:
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                break
    
    # After training, load the best model weights
    if experiment_tracker is not None:
        best_model_path = os.path.join(experiment_tracker.save_dir, 'checkpoints', 'best_model.pth')
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final validation with best model
        best_metrics = validate_epoch_with_ordinal_metrics(
            model, val_loader, criterion, device, num_classes, class_names
        )
    else:
        model.load_state_dict(best_model_state)
    
    print("\nBest model performance:")
    print(f"Validation Accuracy: {best_metrics['val_acc']:.4f}")
    print(f"Validation F1 Score: {best_metrics['val_f1']:.4f}")
    print(f"Ordinal Top-1 Accuracy (off by ≤1): {best_metrics['ordinal_top1_accuracy']:.4f}")
    print(f"Ordinal Top-2 Accuracy (off by ≤1): {best_metrics['ordinal_top2_accuracy']:.4f}")
    
    return model, best_metrics


# Enhanced ExperimentTracker methods (to be added to utils.py)

def save_ordinal_metrics(self, epoch, ordinal_metrics, class_names):
    """
    Save ordinal metrics visualizations.
    
    Args:
        epoch (int): Current epoch
        ordinal_metrics (dict): Ordinal metrics dictionary
        class_names (list): Names of classes
    """
    plots_dir = os.path.join(self.save_dir, 'plots')
    
    # Plot ordinal confusion matrix
    plot_ordinal_confusion_matrix(
        ordinal_metrics['confusion_matrix'],
        class_names,
        output_path=os.path.join(plots_dir, f'ordinal_confusion_matrix_epoch_{epoch}.png')
    )
    
    # Plot error distribution
    plot_error_distribution(
        ordinal_metrics,
        output_path=os.path.join(plots_dir, f'error_distribution_epoch_{epoch}.png')
    )