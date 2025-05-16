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
from utils import set_seed, ExperimentTracker, get_optimizer, get_scheduler, get_loss_function, update_experiment_directory

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
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout probability (0.0 to disable)')
    parser.add_argument('--dropout_type', type=str, default='standard', 
                        choices=['standard', 'spatial', 'feature'], help='Type of dropout to apply')
    
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
                        choices=[None, 'None', 'step', 'cosine', 'reduce_on_plateau'])
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'focal'])

    # Evaluation settings
    parser.add_argument('--use_ordinal_metrics', action='store_true', 
                        help='Use ordinal metrics for evaluation')
    
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
    
    # Create model with dropout
    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        dropout_type=args.dropout_type
    )
    model.to(device)
    
    # Calculate model size and FLOPs
    input_size = (3, args.input_size, args.input_size)
    flops, params = calculate_flops(model, input_size=input_size, device=device)
    inference_time = measure_inference_time(model, input_size=input_size, device=device)
    
    print(f"Model parameters: {params:,}")
    print(f"Model GFLOPs: {flops / 1e9:.2f}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} MB")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Dropout rate: {args.dropout_rate}, Type: {args.dropout_type}")
    
    # Create optimizer, loss function, and scheduler
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    criterion = get_loss_function(args.loss)
    scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)
    
    # Create configuration dictionary
    config = vars(args)
    config['num_classes'] = num_classes
    config['class_names'] = class_names
    config['input_resolution'] = args.input_size
    config['date'] = datetime.now().strftime('%Y-%m-%d')

    # Update experiment directory based on model name prefix
    args.save_dir = update_experiment_directory(args.model, args.save_dir)
    print(f"Saving each {args.model} experiment to {args.save_dir}")
    
    # Create experiment tracker
    experiment_tracker = ExperimentTracker(
        experiment_name=args.experiment_name,
        model_name=args.model,
        config=config,
        save_dir=args.save_dir
    )

    if 'baseline' not in args.model:
        print(f"Training {args.model} for {args.epochs} epochs...")
        # Train model
        best_model, opimizer, val_results, best_epoch = train_model(
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

        print(f"Best validation accuracy: {100 * val_results['val_acc']:.2f}%")
        print(f"Best validation loss: {val_results['val_loss']:.4f}")
        val_results['best_epoch'] = best_epoch
    else:
        print(f"Zero-shot evaluation for {args.model} model. Skipping training.")
        best_model = model

    # #Save trained model report if experiment tracker is used
    # try:
    #     if experiment_tracker is not None:  
    #         print(f"Saving trained model report based on valid dataset...")
    #         with open(os.path.join(experiment_tracker.save_dir, 'training_phase_report.json'), 'w') as f:
    #             best_metrics['best_epoch'] = best_epoch
    #             best_metrics['model_stats'] = {
    #                 'model_name': args.model, 
    #                 'flops': flops, 
    #                 'params': params,
    #                 'inference_time': inference_time,
    #                 'model_size': best_model.state_dict().get('model_size', None)
    #             }
    #             # Convert metrics to serializable format before saving
    #             serializable_metrics = convert_to_serializable(best_metrics)
    #             json.dump(serializable_metrics, f, indent=4)    
    #     else:
    #         print(f"Experiment tracker not provided. Skipping model report saving.")
    # except Exception as e:
    #     print(f"Error saving model report: {e}")
    
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

            print(f"\n{'='*50}")
            print(f"Test results: ")  
            print(f"Loss: {test_results['val_loss']:.4f} || Accuracy: {100 * test_results['val_acc']:.2f}%")
            print(f"Precision: {test_results['val_precision']:.4f}")
            print(f"Recall: {test_results['val_recall']:.4f}")
            print(f"F1 Score: {test_results['val_f1']:.4f}")
            print(f"Top-1 Error: {test_results['val_top1_error']:.4f}")
            print(f"Top-2 Error: {test_results['val_top2_error']:.4f}")
            print(f"Weighted Ordinal Error: {test_results['weighted_ordinal_error']:.4f}")
            print(f"Standard Top-1 Accuracy: {100 * test_results['standard_top1_accuracy']:.2f}%")
            print(f"Standard Top-2 Accuracy: {100 * test_results['standard_top2_accuracy']:.2f}%")
            print(f"Ordinal Top-1 Accuracy: {100 * test_results['ordinal_top1_accuracy']:.2f}%")
            print(f"Ordinal Top-2 Accuracy: {100 * test_results['ordinal_top2_accuracy']:.2f}%")
            for class_name, acc in test_results['standard_class_accuracy'].items():
                print(f"{class_name} Accuracy: {100 * acc:.2f}%")
        else:
            print("No test loader provided. Skipping validation.")
    except Exception as e:
        print(f"Error during validation: {e}")

    if 'baseline' in args.model:
        val_results = test_results

    # Save experiment summary
    print(f"\n{'='*50}")
    print(f"Saving experiment summary...")
    experiment_tracker.save_experiment_summary(
        model=best_model,
        params_count=params,
        flops=flops,
        inference_time=inference_time,
        class_names=class_names,
        val_metrics=val_results,
        test_metrics=test_results,
    )
    
    print(f"Experiment completed. Results saved to {experiment_tracker.save_dir}")

if __name__ == '__main__':
    main()