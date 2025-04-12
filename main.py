#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for running post-tornado damage recognition experiments.
"""

import os
import argparse
import json
import torch
from datetime import datetime

from data_utils import get_data_loaders
from models import create_model
from training import train_model, validate_model
from metrics import calculate_flops, measure_inference_time
from utils import set_seed, ExperimentTracker, get_optimizer, get_scheduler, get_loss_function

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Post-Tornado Damage Recognition Experiments')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default='experiment', help='Name of experiment')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model settings
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    
    # Data settings
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_augmentation', type=str, default='basic', 
                        choices=['none', 'basic', 'standard', 'advanced'], 
                        help='Data augmentation strategy')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default=None, 
                        choices=[None, 'step', 'cosine', 'reduce_on_plateau'])
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'focal'])
    
    # Miscellaneous
    parser.add_argument('--device', type=str, default=None, help='Device (cuda or cpu)')
    
    return parser.parse_args()

def main():
    """Main function to run experiments."""
    args = parse_args()

    # # Create a timestamped output directory
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # args.save_dir = os.path.join(args.save_dir, timestamp)

    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir=args.data_dir,
        input_size=args.input_size,
        batch_size=args.batch_size,
        data_augmentation=args.data_augmentation
    )
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Create model
    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        activation=args.activation
    )
    model.to(device)
    
    # Calculate model size and FLOPs
    input_size = (3, args.input_size, args.input_size)
    flops, params = calculate_flops(model, input_size=input_size, device=device)
    inference_time = measure_inference_time(model, input_size=input_size, device=device)
    
    print(f"Model parameters: {params:,}")
    print(f"Model FLOPs: {flops:,}")
    print(f"Inference time: {inference_time:.2f} ms")
    
    # Create optimizer, loss function, and scheduler
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    criterion = get_loss_function(args.loss)
    scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)
    
    # Create configuration dictionary
    config = vars(args)
    config['num_classes'] = num_classes
    config['class_names'] = class_names
    config['input_resolution'] = args.input_size
    config['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create experiment tracker
    experiment_tracker = ExperimentTracker(
        experiment_name=args.experiment_name,
        model_name=args.model,
        config=config,
        save_dir=args.save_dir
    )
    
    # Train model
    best_model, optimizer, val_conf_matrix, val_class_report = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=num_classes,
        num_epochs=args.epochs,
        patience=args.patience,
        experiment_tracker=experiment_tracker
    )

    #Validation metrics
    print(f"{'='*50}")
    print(f"Validation metrics:")
    print(f"Validation Matrix: {val_conf_matrix}")
    print(f"Validation Class Report: {val_class_report}")

    # Load best model for evaluation
    print(f"{'='*50}")
    print(f"Loading best model for evaluation...")
    best_model_path = f"{experiment_tracker.save_dir}/checkpoints/best_model.pth"
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_model.eval()
    best_model.load_state_dict(torch.load(experiment_tracker.best_model_path))
    
    try:
        if test_loader is not None:
            # Validate model
            test_results = validate_model(
                model=best_model,
                val_loader=test_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes
            )
            val_loss, val_acc, val_precision, val_recall, val_f1, top1_error, top2_error, conf_matrix, class_report = test_results.values()
            print(f"{'='*50}")
            print(f"Test results: ")  
            print(f"Loss: {val_loss:.4f} || Accuracy: {100 * val_acc:.4f}%")
            print(f"Precision: {val_precision:.4f}")
            print(f"Recall: {val_recall:.4f}")
            print(f"F1 Score: {val_f1:.4f}")
            print(f"Top-1 Error: {top1_error:.4f}")
            print(f"Top-2 Error: {top2_error:.4f}")
        else:
            print("No test loader provided. Skipping validation.")
    except Exception as e:
        print(f"Error during validation: {e}")

    # Save experiment summary
    print(f"{'='*50}")
    print(f"Saving experiment summary...")
    experiment_tracker.save_experiment_summary(
        model=best_model,
        params_count=params,
        flops=flops,
        inference_time=inference_time,
        conf_matrix=conf_matrix,
        class_names=class_names,
        classification_report_dict=class_report
    )
    
    print(f"Experiment completed. Results saved to {experiment_tracker.save_dir}")

if __name__ == '__main__':
    main()