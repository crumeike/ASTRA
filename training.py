#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model training and validation utilities for post-tornado damage recognition experiments.
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, classification_report
from datetime import datetime

# Import ordinal metrics
from ordinal_metrics import (
    evaluate_damage_state_predictions,
)

# Define damage state names
DAMAGE_STATE_NAMES = ["DS0-Undamaged", "DS1-Slight", "DS2-Moderate", 
                      "DS3-Extensive", "DS4-Complete", "DS5-Debris"]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda or cpu)
    
    Returns:
        tuple: (epoch_loss, epoch_acc)
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


def validate_epoch(model, val_loader, criterion, device, num_classes, class_names=None):
    """
    Validate model on validation set.
    
    Args:
        model (nn.Module): Neural network model
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device (cuda or cpu)
        num_classes (int): Number of output classes
        class_names (list, optional): Names of classes
    
    Returns:
        dict: Dictionary containing validation metrics such as loss, accuracy, precision, recall, F1 score, etc.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probs = []

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
            
            # Store predictions and targets
            _, predicted = outputs.max(1)
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    # Calculate standard metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_predictions)
    
    # Calculate precision, recall, and F1 score (weighted by support)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    
    # Calculate standard top-1 and top-2 error
    top1_error = 1.0 - epoch_acc
    
    # For top-2 error, check if the true class is in the top 2 predictions
    top2_correct = 0
    for i, target in enumerate(all_targets):
        # Get indices of top 2 predictions
        top2_indices = np.argsort(all_probs[i])[-2:]
        if target in top2_indices:
            top2_correct += 1
    
    top2_error = 1.0 - (top2_correct / len(all_targets))
    
    # Compute standard confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions, labels=range(num_classes))
    
    # Generate classification report
    report = classification_report(all_targets, all_predictions, output_dict=True, zero_division=0)

    # Compute ordinal metrics
    ordinal_metrics = evaluate_damage_state_predictions(all_probs, all_targets, class_names)    
    
    # Combine all metrics
    metrics = {
        'val_loss': epoch_loss,
        'val_acc': epoch_acc, #same as standard accuracy from ordinal_metrics
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1,
        'val_top1_error': top1_error,
        'val_top2_error': top2_error,
    }

    # Add ordinal metrics
    metrics.update({
        f"{k}": v for k, v in ordinal_metrics.items() 
        if k not in ['confusion_matrix']
    })
    
    metrics['confusion_matrix'] = conf_matrix
    metrics['class_report'] = report
    
    return metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                num_classes, num_epochs, patience=10, experiment_tracker=None, class_names=None):
    """
    Train and validate model for specified number of epochs.
    
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
        experiment_tracker (ExperimentTracker): Experiment tracker object
        class_names (list, optional): Names of classes
    
    Returns:
        tuple: (best_model, best_optimizer, best_metrics, best_epoch)
    """
    # Initialize variables for early stopping
    best_epoch = 0
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
        
        # Validation phase
        metrics = validate_epoch(model, val_loader, criterion, device, num_classes, class_names)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler if using ReduceLROnPlateau
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics['val_loss'])
            else:
                scheduler.step()
        
        # Update metrics in tracker if provided
        is_best = False
        if experiment_tracker is not None:
            is_best = experiment_tracker.update_metrics(
                epoch=epoch+1,          #Epoch number (1-indexed)
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=metrics['val_loss'],
                val_acc=metrics['val_acc'],
                val_precision=metrics['val_precision'],
                val_recall=metrics['val_recall'],
                val_f1=metrics['val_f1'],
                val_top1_error=metrics['val_top1_error'],
                val_top2_error=metrics['val_top2_error'],
                weighted_ordinal_error=metrics['weighted_ordinal_error'],
                standard_top1_acc=metrics['standard_top1_accuracy'],
                standard_top2_acc=metrics['standard_top2_accuracy'],
                ordinal_top1_acc=metrics['ordinal_top1_accuracy'],
                ordinal_top2_acc=metrics['ordinal_top2_accuracy'],
                standard_class_acc=metrics['standard_class_accuracy'],
                learning_rate=current_lr
            )
            
            # Save model checkpoint
            experiment_tracker.save_model(model, optimizer, epoch, is_best=is_best)
            
            # Save best metrics if is_best is True
            if is_best:
                best_metrics = metrics.copy()
        else:

            # Save model checkpoint after every 5 epochs
            if not os.path.exists(f"playground/checkpoints/{timestamp}"):
                os.makedirs(f"playground/checkpoints/{timestamp}")
                
            # Save model checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': metrics
            }, f"playground/checkpoints/{timestamp}/last_checkpoint.pth")

            # If no tracker is provided, use F1 score to determine best model
            is_best = epoch == 0 or metrics['val_f1'] > best_val_f1
            if is_best:
                best_val_f1 = metrics['val_f1']
                best_metrics = metrics.copy()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {metrics['val_loss']:.4f} | Val Acc: {metrics['val_acc']:.4f} | F1: {metrics['val_f1']:.4f}")
    
        # Also print ordinal metrics
        print(f"Ordinal Top-1 Acc (off by ±1 class): {metrics['ordinal_top1_accuracy']:.4f} | "
              f"Ordinal Top-2 Acc (off by ±1 class): {metrics['ordinal_top2_accuracy']:.4f}")
        
        # Early stopping check
        if is_best:
            best_epoch = epoch + 1  # Store best epoch (1-indexed)
            no_improve_epochs = 0
            
            # Save best model and optimizer state if no tracker is provided
            if experiment_tracker is None:
                best_model_state = model.state_dict()
                best_optimizer_state = optimizer.state_dict()
                print(f"Best model saved at epoch {best_epoch} with F1 score: {metrics['val_f1']:.4f}")

                torch.save({
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': best_optimizer_state,
                    'epoch': best_epoch,
                    'metrics': metrics
                }, f"playground/checkpoints/{timestamp}/best_model.pth")

        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                break
    
    # After training, load the best model weights
    if experiment_tracker is not None:
        # Load the best model from the tracker
        print(f"{'='*50}")
        print(f"\nLoading best model from epoch {best_epoch} for evaluation on test dataset...")
        best_model_path = f"{experiment_tracker.save_dir}/checkpoints/best_model.pth"
        checkpoint = torch.load(best_model_path)
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        if experiment_tracker.config["activation"] != "reLU" or experiment_tracker.config["activation"] != "leaky_relu":
            # If the model architecture has changed, load only the matching parameters
            # This is useful if the model architecture has been modified (e.g., different activation function)
            # Remove the ReLU-specific keys from the state dict
            model_state_dict = {k: v for k, v in model_state_dict.items() if 'relu' not in k}
            # optimizer_state_dict = {k: v for k, v in optimizer_state_dict.items() if 'relu' not in k}
            
            # Load the model state dicts
            # Note: strict=False allows loading only the matching parameters
            model.load_state_dict(model_state_dict, strict=False)
        else:
            # Load the model state dicts    
            model.load_state_dict(model_state_dict)

        # Load the optimizer state dicts
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        # Load the best model from the saved checkpoint
        print(f"{'='*50}")
        print(f"\nLoading best model from epoch {best_epoch} for evaluation on test dataset...")
        best_model_path = f"playground/checkpoints/{timestamp}/best_model.pth"
        checkpoint = torch.load(best_model_path)
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        if experiment_tracker.config["activation"] != "reLU" or experiment_tracker.config["activation"] != "leaky_relu":
            # If the model architecture has changed, load only the matching parameters
            # This is useful if the model architecture has been modified (e.g., different activation function)
            # Remove the ReLU-specific keys from the state dict
            model_state_dict = {k: v for k, v in model_state_dict.items() if 'relu' not in k}
            # optimizer_state_dict = {k: v for k, v in optimizer_state_dict.items() if 'relu' not in k}
            
            # Load the model state dicts
            # Note: strict=False allows loading only the matching parameters
            model.load_state_dict(model_state_dict, strict=False)
        else:
            # Load the model state dicts    
            model.load_state_dict(model_state_dict)

        # Load the optimizer state dicts
        optimizer.load_state_dict(optimizer_state_dict)
    
    # # Final validation with best model
    # val_loss, val_acc, val_precision, val_recall, val_f1, top1_error, top2_error, conf_matrix, class_report = validate_epoch(
    #     model, val_loader, criterion, device, num_classes)
    
    # print("\nBest model performance:")
    # print(f"Validation Accuracy: {val_acc:.4f}")
    # print(f"Validation F1 Score: {val_f1:.4f}")
    # print(f"Top-1 Error: {top1_error:.4f}")
    # print(f"Top-2 Error: {top2_error:.4f}")
    
    return model, optimizer, best_metrics, best_epoch


def validate_model(model, val_loader, criterion, device, num_classes):
    """
    Validate a model on validation set and return metrics.
    
    Args:
        model (nn.Module): Neural network model
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device (cuda or cpu)
        num_classes (int): Number of output classes
    
    Returns:
        dict: Dictionary containing validation metrics such as loss, accuracy, precision, recall, F1 score, etc.
    """
    val_metrics = validate_epoch(model, val_loader, criterion, device, num_classes)
    
    return val_metrics