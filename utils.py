#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
General utilities and experiment tracking for post-tornado damage recognition experiments.
"""

import os
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model, optimizer_name, learning_rate, weight_decay=0):
    """
    Create optimizer based on configuration.
    
    Args:
        model (nn.Module): Neural network model
        optimizer_name (str): Name of optimizer
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
    
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    if optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name, num_epochs):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler_name (str): Name of scheduler
        num_epochs (int): Number of training epochs
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler or None
    """
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None
    elif scheduler_name.lower() == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
    elif scheduler_name.lower() == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name.lower() == 'reduce_on_plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def get_loss_function(loss_name, class_weights=None):
    """
    Create loss function based on configuration.
    
    Args:
        loss_name (str): Name of loss function
        class_weights (torch.Tensor, optional): Weights for each class
    
    Returns:
        nn.Module: Loss function
    """
    if loss_name.lower() == 'cross_entropy':
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        else:
            return nn.CrossEntropyLoss()
    
    elif loss_name.lower() == 'focal':
        try:
            # Try using kornia's implementation
            from kornia.losses import FocalLoss
            return FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        
        except ImportError:
            # Custom implementation of Focal Loss
            class FocalLoss(nn.Module):
                def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
                    super(FocalLoss, self).__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.reduction = reduction
                    self.ce_loss = nn.CrossEntropyLoss(reduction='none')
                
                def forward(self, input, target):
                    logp = self.ce_loss(input, target)
                    p = torch.exp(-logp)
                    loss = self.alpha * (1 - p) ** self.gamma * logp
                    
                    if self.reduction == 'mean':
                        return loss.mean()
                    elif self.reduction == 'sum':
                        return loss.sum()
                    else:
                        return loss
            
            return FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
    
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


class ExperimentTracker:
    """Class to track experiment metrics and save results."""
    
    def __init__(self, experiment_name, model_name, config, save_dir='experiments'):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name (str): Name of experiment
            model_name (str): Name of model architecture
            config (dict): Configuration parameters
            save_dir (str): Directory to save results
        """
        timestamp = datetime.now().strftime('%Y%m%d %H%M%S')
        self.experiment_id = f"{model_name}_{timestamp}"
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.config = config
        self.base_save_dir = save_dir
        self.save_dir = os.path.join(save_dir, self.experiment_id)
        
        # Create directories
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            os.makedirs(os.path.join(self.save_dir, 'plots'))
            os.makedirs(os.path.join(self.save_dir, 'checkpoints'))
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'tensorboard'))
        
        # Initialize metrics tracking
        self.best_metrics = {'val_f1': 0.0}
        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_top1_error': [],
            'val_top2_error': [],
            'learning_rate': []
        }
        
        # Save experiment configuration
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump({
                'experiment_name': experiment_name,
                'model_name': model_name,
                'config': config,
                'timestamp': timestamp
            }, f, indent=2)
    
    def update_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, 
                       val_precision, val_recall, val_f1, val_top1_error, val_top2_error,
                       learning_rate):
        """
        Update metrics history and determine if current model is best.
        
        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            train_acc (float): Training accuracy
            val_loss (float): Validation loss
            val_acc (float): Validation accuracy
            val_precision (float): Validation precision
            val_recall (float): Validation recall
            val_f1 (float): Validation F1 score
            val_top1_error (float): Validation top-1 error
            val_top2_error (float): Validation top-2 error
            learning_rate (float): Current learning rate
        
        Returns:
            bool: True if current model is best so far
        """
        # Update metrics history
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_acc'].append(train_acc)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_acc'].append(val_acc)
        self.metrics_history['val_precision'].append(val_precision)
        self.metrics_history['val_recall'].append(val_recall)
        self.metrics_history['val_f1'].append(val_f1)
        self.metrics_history['val_top1_error'].append(val_top1_error)
        self.metrics_history['val_top2_error'].append(val_top2_error)
        self.metrics_history['learning_rate'].append(learning_rate)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        self.writer.add_scalar('Precision/val', val_precision, epoch)
        self.writer.add_scalar('Recall/val', val_recall, epoch)
        self.writer.add_scalar('F1/val', val_f1, epoch)
        self.writer.add_scalar('Error/top1', val_top1_error, epoch)
        self.writer.add_scalar('Error/top2', val_top2_error, epoch)
        self.writer.add_scalar('LR', learning_rate, epoch)
        
        # Check if this is the best model so far
        if val_f1 > self.best_metrics['val_f1']:
            self.best_metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'val_top1_error': val_top1_error,
                'val_top2_error': val_top2_error
            }
            return True  # Return True if this is the best model
        return False
    
    def save_model(self, model, optimizer, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            model (nn.Module): Neural network model
            optimizer (torch.optim.Optimizer): Optimizer
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metrics': self.best_metrics,
            'config': self.config
        }
        
        # Save the latest checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoints', 'last_checkpoint.pth'))
        
        # If this is the best model, save a separate copy
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoints', 'best_model.pth'))
    
    def plot_metrics(self):
        """Plot training and validation metrics."""
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)
        
        # 1. Loss and Accuracy curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.metrics_history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.metrics_history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.metrics_history['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(epochs, self.metrics_history['val_acc'], 'r-', label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 'loss_accuracy.png'))
        plt.close()
        
        # 2. Precision, Recall, F1 curves
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, self.metrics_history['val_precision'], 'g-', label='Precision')
        plt.plot(epochs, self.metrics_history['val_recall'], 'b-', label='Recall')
        plt.plot(epochs, self.metrics_history['val_f1'], 'r-', label='F1 Score')
        plt.title('Precision, Recall, and F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 'precision_recall_f1.png'))
        plt.close()
        
        # 3. Top-1 and Top-2 Error
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.metrics_history['val_top1_error'], 'b-', label='Top-1 Error')
        plt.plot(epochs, self.metrics_history['val_top2_error'], 'r-', label='Top-2 Error')
        plt.title('Top-1 and Top-2 Error Rates')
        plt.xlabel('Epochs')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 'top_k_error.png'))
        plt.close()
        
        # 4. Learning rate
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.metrics_history['learning_rate'], 'g-')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 'learning_rate.png'))
        plt.close()
    
    def plot_confusion_matrix(self, conf_matrix, class_names):
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix (numpy.ndarray): Confusion matrix
            class_names (list): List of class names
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', 'confusion_matrix.png'))
        plt.close()
    
    def save_metrics_to_csv(self):
        """Save metrics history to CSV."""
        df = pd.DataFrame(self.metrics_history)
        df.index = df.index + 1  # Start epochs at 1
        df.index.name = 'epoch'
        df.to_csv(os.path.join(self.save_dir, 'metrics_history.csv'))
        
        # Save best metrics separately
        with open(os.path.join(self.save_dir, 'best_metrics.json'), 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
    
    def save_experiment_summary(self, model, params_count, flops, inference_time, conf_matrix, class_names, classification_report_dict):
        """
        Save comprehensive experiment summary.
        
        Args:
            model (nn.Module): Neural network model
            params_count (int): Number of parameters
            flops (int): Number of FLOPs
            inference_time (float): Inference time (ms)
            conf_matrix (numpy.ndarray): Confusion matrix
            class_names (list): List of class names
            classification_report_dict (dict): Classification report
        """
        # Create summary dictionary
        summary = {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'model_name': self.model_name,
            'config': self.config,
            'best_metrics': self.best_metrics,
            'model_stats': {
                'params_count': int(params_count),
                'flops': int(flops),
                'inference_time_ms': float(inference_time),
                'model_size_mb': os.path.getsize(os.path.join(self.save_dir, 'checkpoints', 'best_model.pth')) / (1024 * 1024)
            },
            'classification_report': classification_report_dict
        }
        
        # Save the summary to JSON
        with open(os.path.join(self.save_dir, 'experiment_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(conf_matrix, class_names)
        
        # Save other metrics plots
        self.plot_metrics()
        
        # Save metrics to CSV
        self.save_metrics_to_csv()
        
        # Save model architecture summary as text
        try:
            from torchsummary import summary as model_summary
            with open(os.path.join(self.save_dir, 'model_summary.txt'), 'w') as f:
                # Redirect stdout to file temporarily
                import sys
                old_stdout = sys.stdout
                sys.stdout = f
                
                # Print model summary
                if torch.cuda.is_available():
                    input_size = (3, self.config['input_resolution'], self.config['input_resolution'])
                    model_summary(model, input_size=input_size)
                
                # Restore stdout
                sys.stdout = old_stdout
        except ImportError:
            print("torchsummary not installed. Install with 'pip install torchsummary' for detailed model summaries.")
        
        print(f"Experiment summary saved to {self.save_dir}")
        