
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run multiple experiments with different configurations,
with the ability to resume from interrupted runs.
"""

import os
import argparse
import json
import pandas as pd
from itertools import product
import subprocess
from datetime import datetime
import pickle
import hashlib
from utils import update_experiment_directory

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multiple post-tornado damage recognition experiments')
    
    # Experiment settings
    parser.add_argument('--config_file', type=str, default='experiment_configs.json',
                        help='JSON file with experiment configurations')
    parser.add_argument('--results_dir', type=str, default='experiment_results',
                        help='Directory to save results summary')
    parser.add_argument('--parallel', action='store_true',
                        help='Run experiments in parallel (requires GNU Parallel)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume experiments from the last checkpoint')
    parser.add_argument('--max_experiments', type=int, default=None,
                        help='Maximum number of experiments to run (useful for testing)')
    
    return parser.parse_args()

def load_experiment_configs(config_file):
    """
    Load experiment configurations from JSON file.
    
    Args:
        config_file (str): Path to JSON configuration file
    
    Returns:
        list: List of experiment configurations
        str: Directory to save individual experiment results
        str: Model name to be used to update the results directory for each experiment run
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Check if config is a list of experiments or a parameter grid
    if isinstance(config, list):
        # Config is already a list of experiments
        return config
    else:
        # Config is a parameter grid, generate all combinations
        param_grid = config.get('param_grid', {})
        base_config = config.get('base_config', {})
        
        # Generate all combinations of parameters
        keys = param_grid.keys()
        values = param_grid.values()
        
        experiments = []
        for combination in product(*values):
            # Create experiment config by combining base config with specific params
            experiment = base_config.copy()
            experiment.update(dict(zip(keys, combination)))
            
            # Generate experiment name based on parameters
            name_parts = []
            for key, value in zip(keys, combination):
                name_parts.append(f"{key}-{value}")
            
            experiment['experiment_name'] = f"{base_config.get('experiment_name', 'exp')}_{'-'.join(name_parts)}"
            print(f"Generated experiment: {experiment['experiment_name']}")
            experiments.append(experiment)
            save_dir = base_config.get('save_dir', 'experiments')
            model_list = param_grid.get('model', ['default_model'])
            model_name = model_list[0]  # Select the first model as default
        
        return experiments, save_dir, model_name

def generate_experiment_hash(experiment):
    """
    Generate a unique hash for an experiment configuration.
    
    Args:
        experiment (dict): Experiment configuration
    
    Returns:
        str: Unique hash for the experiment
    """
    # Convert experiment dict to a sorted string representation
    exp_str = json.dumps(experiment, sort_keys=True)
    
    # Generate hash
    return hashlib.md5(exp_str.encode()).hexdigest()

def save_checkpoint(results_dir, completed_experiments, pending_experiments):
    """
    Save the current state of experiment execution.
    
    Args:
        results_dir (str): Directory to save results
        completed_experiments (list): List of completed experiment hashes
        pending_experiments (list): List of pending experiment configurations
    """
    checkpoint_path = os.path.join(results_dir, "experiment_checkpoint.pkl")
    
    checkpoint = {
        'completed': completed_experiments,
        'pending': pending_experiments,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved: {len(completed_experiments)} completed, {len(pending_experiments)} pending")


def load_checkpoint(results_dir):
    """
    Load experiment checkpoint.
    
    Args:
        results_dir (str): Directory with results
    
    Returns:
        tuple: (completed_experiments, pending_experiments)
    """
    checkpoint_path = os.path.join(results_dir, "experiment_checkpoint.pkl")
    
    if not os.path.exists(checkpoint_path):
        return [], []
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"Loaded checkpoint from {checkpoint['timestamp']}")
    print(f"Completed experiments: {len(checkpoint['completed'])}")
    print(f"Pending experiments: {len(checkpoint['pending'])}")
    
    return checkpoint['completed'], checkpoint['pending']


def run_experiments(experiments, save_dir, model_name, results_dir, parallel=False, resume=False, max_experiments=None):
    """
    Run multiple experiments with different configurations.
    
    Args:
        experiments (list): List of experiment configurations
        model_name (str): Name of the model to be used
        results_dir (str): Directory to save results summary
        parallel (bool): Whether to run experiments in parallel
        resume (bool): Whether to resume from the last checkpoint
        max_experiments (int, optional): Maximum number of experiments to run
    """
    #update_result directory
    results_dir = update_experiment_directory(model_name, results_dir)
    print(f"Summary of Results will be saved in: {results_dir}")

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Load checkpoint if resuming
    completed_experiment_hashes = []
    if resume:
        completed_experiment_hashes, pending_experiments = load_checkpoint(results_dir)
        if pending_experiments:
            # Use pending experiments from checkpoint if available
            experiments = pending_experiments
    
    # Limit number of experiments if specified
    if max_experiments is not None:
        experiments = experiments[:max_experiments]
    
    # Create commands for each experiment
    commands = []
    filtered_experiments = []
    
    for i, experiment in enumerate(experiments):
        # Generate hash for the experiment
        exp_hash = generate_experiment_hash(experiment)
        
        # Skip if already completed
        if exp_hash in completed_experiment_hashes:
            print(f"Skipping experiment {i+1}/{len(experiments)} (already completed)")
            continue
        
        # Create command with all parameters
        cmd = ["python", "main.py"]
        
        for key, value in experiment.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
        commands.append(cmd)
        filtered_experiments.append(experiment)
    
    print(f"Running {len(commands)} experiments ({len(experiments) - len(commands)} already completed)")
    
    # Run experiments
    if parallel:
        # Write commands to file for GNU Parallel
        with open(os.path.join(results_dir, "commands.txt"), "w") as f:
            for cmd in commands:
                f.write(" ".join(cmd) + "\n")
        
        # Run with GNU Parallel
        parallel_cmd = [
            "parallel", "--bar", "--joblog", f"{results_dir}/parallel.log",
            "--", "<", f"{results_dir}/commands.txt"
        ]
        subprocess.run(" ".join(parallel_cmd), shell=True)
        
        # Mark all as completed
        for experiment in filtered_experiments:
            completed_experiment_hashes.append(generate_experiment_hash(experiment))
        
        # Save checkpoint
        save_checkpoint(results_dir, completed_experiment_hashes, [])
    else:
        # Run sequentially with checkpoint saving
        for i, (cmd, experiment) in enumerate(zip(commands, filtered_experiments)):
            exp_hash = generate_experiment_hash(experiment)
            
            print(f"\n[{i+1}/{len(commands)}] Running experiment: {' '.join(cmd)}")
            try:
                subprocess.run(cmd)
                # Mark as completed
                completed_experiment_hashes.append(exp_hash)
                # Save checkpoint after each experiment
                save_checkpoint(
                    results_dir, 
                    completed_experiment_hashes, 
                    filtered_experiments[i+1:] if i+1 < len(filtered_experiments) else []
                )
            except KeyboardInterrupt:
                print("\nExperiment interrupted by user. Saving checkpoint...")
                # Save checkpoint on interrupt
                save_checkpoint(
                    results_dir, 
                    completed_experiment_hashes, 
                    filtered_experiments[i:] if i < len(filtered_experiments) else []
                )
                raise
    
    print(f"\nAll experiments completed. Collecting results...")
    collect_results(results_dir, save_dir, model_name)


def collect_results(results_dir, save_dir, model_name):
    """
    Collect and summarize results from all experiments.
    
    Args:
        results_dir (str): Directory containing experiment results
        save_dir (str): Directory to save individual experiment results
        model_name (str): Name of the model to be used to update the results directory
    """
    # Find all experiment directories
    experiment_dirs = []
    base_dir = update_experiment_directory(model_name, save_dir)  # Default save directory from config file    
    
    for experiment in os.listdir(base_dir):
        exp_dir = os.path.join(base_dir, experiment)
        if os.path.isdir(exp_dir):
            # Check if this directory has experiment results
            if os.path.exists(os.path.join(exp_dir, "experiment_summary.json")):
                experiment_dirs.append(exp_dir)
    
    # Collect results from each experiment
    results = []
    for exp_dir in experiment_dirs:
        try:
            # Load experiment summary
            with open(os.path.join(exp_dir, "experiment_summary.json"), "r") as f:
                summary = json.load(f)
            
            # Extract relevant metrics
            result = {
                "experiment_id": summary["experiment_id"],
                "experiment_name": summary["experiment_name"],
                "model_name": summary["model_name"],
                "val_accuracy": summary["best_metrics(test)"]["val_acc"],
                "val_f1": summary["best_metrics(test)"]["val_f1"],
                "val_precision": summary["best_metrics(test)"]["val_precision"],
                "val_recall": summary["best_metrics(test)"]["val_recall"],
                "top1_error": summary["best_metrics(test)"]["val_top1_error"],
                "top2_error": summary["best_metrics(test)"]["val_top2_error"],
                "standard_top1_accuracy": summary["best_metrics(test)"].get("standard_top1_accuracy", None),
                "standard_top2_accuracy": summary["best_metrics(test)"].get("standard_top2_accuracy", None),
                "ordinal_top1_accuracy": summary["best_metrics(test)"].get("ordinal_top1_accuracy", None),
                "ordinal_top2_accuracy": summary["best_metrics(test)"].get("ordinal_top2_accuracy", None),
                "weighted_ordinal_error": summary["best_metrics(test)"].get("weighted_ordinal_error", None),
                "standard_class_accuracy": summary["best_metrics(test)"].get("standard_class_accuracy", None),
                "off_by_0_percent": summary["best_metrics(test)"].get("off_by_0_percent", None),
                "off_by_1_percent": summary["best_metrics(test)"].get("off_by_1_percent", None),
                "off_by_2_percent": summary["best_metrics(test)"].get("off_by_2_percent", None),
                "off_by_3_percent": summary["best_metrics(test)"].get("off_by_3_percent", None),
                "off_by_4_percent": summary["best_metrics(test)"].get("off_by_4_percent", None),
                "off_by_5_percent": summary["best_metrics(test)"].get("off_by_5_percent", None),
                "params_count": summary["model_stats"]["params_count_(M)"],
                "gflops": summary["model_stats"]["gflops"],
                "inference_time_ms": summary["model_stats"]["inference_time_ms"],
                "model_size_mb": summary["model_stats"]["model_size_mb"],
                "batch_size": summary["config"].get("batch_size", None),
                "input_resolution": summary["config"].get("input_resolution", None),
                "optimizer": summary["config"].get("optimizer", None),
                "learning_rate": summary["config"].get("lr", None),
                "dropout_rate": summary["config"].get("dropout_rate", None),
                "dropout_type": summary["config"].get("dropout_type", None),
                "weight_decay": summary["config"].get("weight_decay", None),
                "data_augmentation": summary["config"].get("data_augmentation", None),
                "activation": summary["config"].get("activation", None),
                "result_dir": exp_dir
            }
            
            results.append(result)
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
    
    # Create DataFrame and sort by F1 score
    df = pd.DataFrame(results)
    df = df.sort_values("val_f1", ascending=False)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"experiment_results_{timestamp}.csv")
    df.to_csv(results_file, index=False)
    
    # Generate summary report
    summary_file = os.path.join(results_dir, f"experiment_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("========== POST-TORNADO DAMAGE RECOGNITION EXPERIMENTS SUMMARY ==========\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {len(df)}\n\n")
        
        f.write("TOP 5 MODELS BY F1 SCORE:\n")
        for i, row in df.head(5).iterrows():
            f.write(f"{i+1}. {row['experiment_id']} - {row['experiment_name']}\n")
            f.write(f"   F1 Score: {row['val_f1']:.4f}, Accuracy: {row['val_accuracy']:.4f}\n")
            f.write(f"   Top-1 Error: {row['top1_error']:.4f}, Top-2 Error: {row['top2_error']:.4f}\n")
            f.write(f"   STD_Top-1_Acc: {row['standard_top1_accuracy']:.4f}, STD_Top-2_Acc: {row['standard_top2_accuracy']:.4f}\n")
            # Add ordinal metrics if available
            if 'ordinal_top1_accuracy' in row and not pd.isna(row['ordinal_top1_accuracy']):
                f.write(f"   Ordinal Top-1 Accuracy (off by ≤1 class): {row['ordinal_top1_accuracy']:.4f}\n")
            if 'ordinal_top2_accuracy' in row and not pd.isna(row['ordinal_top2_accuracy']):
                f.write(f"   Ordinal Top-2 Accuracy (off by ≤1 class): {row['ordinal_top2_accuracy']:.4f}\n")
                
            f.write(f"   Params: {row['params_count']:,}, Inference Time: {row['inference_time_ms']:.2f} ms\n")
            f.write(f"   Config: BS={row['batch_size']}, IR={row['input_resolution']}, Opt={row['optimizer']}, LR={row['learning_rate']}\n")
            f.write(f"   WD: {row['weight_decay']}, DO : {row['dropout_rate']}, DO Type: {row['dropout_type']}\n")
            f.write(f"   Model Size: {row['model_size_mb']:.2f} MB, GFLOPS: {row['gflops']:.2f}\n")
            f.write(f"   Data Aug: {row['data_augmentation']}, Activation: {row['activation']}\n\n")
        
        f.write("\nFASTEST MODELS (TOP 3):\n")
        for i, row in df.sort_values("inference_time_ms").head(3).iterrows():
            f.write(f"{i+1}. {row['experiment_id']} - {row['experiment_name']}\n")
            f.write(f"   Inference Time: {row['inference_time_ms']:.2f} ms, F1 Score: {row['val_f1']:.4f}\n\n")
        
        f.write("\nSMALLEST MODELS (TOP 3):\n")
        for i, row in df.sort_values("params_count").head(3).iterrows():
            f.write(f"{i+1}. {row['experiment_id']} - {row['experiment_name']}\n")
            f.write(f"   Params: {row['params_count']:,}, F1 Score: {row['val_f1']:.4f}\n\n")
    
    print(f"Results saved to {results_file}")
    print(f"Summary report saved to {summary_file}")
    
    # Clear checkpoint since all experiments are complete
    checkpoint_path = os.path.join(results_dir, "experiment_checkpoint.pkl")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Cleared experiment checkpoint file")

def main():
    """Main function to run experiments."""
    args = parse_args()

    # # Create a timestamped output directory
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # args.results_dir = os.path.join(args.results_dir, timestamp)
    # os.makedirs(args.results_dir, exist_ok=True)
    
    # Load experiment configurations
    experiments, save_dir, model_name = load_experiment_configs(args.config_file)
    print(f"Loaded {len(experiments)} experiment configurations")
    
    # Run experiments
    run_experiments(
        experiments, 
        save_dir,
        model_name,
        args.results_dir, 
        args.parallel, 
        args.resume,
        args.max_experiments,
    )

if __name__ == "__main__":
    main()