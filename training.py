#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model training and validation utilities for post-tornado damage recognition experiments.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, classification_report


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


def validate_epoch(model, val_loader, criterion, device, num_classes):
    """
    Validate model on validation set.
    
    Args:
        model (nn.Module): Neural network model
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device (cuda or cpu)
        num_classes (int): Number of output classes
    
    Returns:
        tuple: (epoch_loss, epoch_acc, precision, recall, f1, top1_error, top2_error, conf_matrix, class_report)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probs = []
    
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
    
    # Calculate metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_predictions)
    
    # Calculate precision, recall, and F1 score (weighted by support)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    
    # Calculate top-1 and top-2 error
    top1_error = 1.0 - epoch_acc
    
    # For top-2 error, check if the true class is in the top 2 predictions
    top2_correct = 0
    for i, target in enumerate(all_targets):
        # Get indices of top 2 predictions
        top2_indices = np.argsort(all_probs[i])[-2:]
        if target in top2_indices:
            top2_correct += 1
    
    top2_error = 1.0 - (top2_correct / len(all_targets))
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions, labels=range(num_classes))
    
    # Generate classification report
    report = classification_report(all_targets, all_predictions, output_dict=True)
    
    return epoch_loss, epoch_acc, precision, recall, f1, top1_error, top2_error, conf_matrix, report


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                num_classes, num_epochs, patience=10, experiment_tracker=None):
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
    
    Returns:
        tuple: (best_model, best_optimizer, conf_matrix, class_report)
    """
    # Initialize variables for early stopping
    best_epoch = 0
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc, val_precision, val_recall, val_f1, top1_error, top2_error, conf_matrix, class_report = validate_epoch(
            model, val_loader, criterion, device, num_classes)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler if using ReduceLROnPlateau
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Update metrics in tracker if provided
        is_best = False
        if experiment_tracker is not None:
            is_best = experiment_tracker.update_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                val_precision=val_precision,
                val_recall=val_recall,
                val_f1=val_f1,
                val_top1_error=top1_error,
                val_top2_error=top2_error,
                learning_rate=current_lr
            )
            
            # Save model checkpoint
            experiment_tracker.save_model(model, optimizer, epoch, is_best=is_best)
        else:
            # If no tracker is provided, use F1 score to determine best model
            is_best = epoch == 0 or val_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_f1
                best_epoch = epoch
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # Early stopping check
        if is_best:
            best_epoch = epoch
            no_improve_epochs = 0
            
            # Save best metrics and outputs if no tracker is provided
            if experiment_tracker is None:
                best_model_state = model.state_dict()
                best_optimizer_state = optimizer.state_dict()
                best_conf_matrix = conf_matrix
                best_class_report = class_report
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                break
    
    # After training, load the best model weights
    if experiment_tracker is not None:
        best_model_path = f"{experiment_tracker.save_dir}/checkpoints/best_model.pth"
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)
    
    # # Final validation with best model
    # val_loss, val_acc, val_precision, val_recall, val_f1, top1_error, top2_error, conf_matrix, class_report = validate_epoch(
    #     model, val_loader, criterion, device, num_classes)
    
    # print("\nBest model performance:")
    # print(f"Validation Accuracy: {val_acc:.4f}")
    # print(f"Validation F1 Score: {val_f1:.4f}")
    # print(f"Top-1 Error: {top1_error:.4f}")
    # print(f"Top-2 Error: {top2_error:.4f}")
    
    return model, optimizer, conf_matrix, class_report


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
        dict: Dictionary containing all validation metrics
    """
    val_loss, val_acc, val_precision, val_recall, val_f1, top1_error, top2_error, conf_matrix, class_report = validate_epoch(
        model, val_loader, criterion, device, num_classes)
    
    return {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'top1_error': top1_error,
        'top2_error': top2_error,
        'conf_matrix': conf_matrix,
        'class_report': class_report
    }