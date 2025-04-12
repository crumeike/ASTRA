#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run multiple experiments with different configurations.
"""

import os
import argparse
import json
import pandas as pd
from itertools import product
import subprocess
from datetime import datetime

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
    
    return parser.parse_args()

def load_experiment_configs(config_file):
    """
    Load experiment configurations from JSON file.
    
    Args:
        config_file (str): Path to JSON configuration file
    
    Returns:
        list: List of experiment configurations
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
            experiments.append(experiment)
        
        return experiments

def run_experiments(experiments, results_dir, parallel=False):
    """
    Run multiple experiments with different configurations.
    
    Args:
        experiments (list): List of experiment configurations
        results_dir (str): Directory to save results summary
        parallel (bool): Whether to run experiments in parallel
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create commands for each experiment
    commands = []
    for i, experiment in enumerate(experiments):
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
    else:
        # Run sequentially
        for i, cmd in enumerate(commands):
            print(f"\n[{i+1}/{len(commands)}] Running experiment: {' '.join(cmd)}")
            subprocess.run(cmd)
    
    print(f"\nAll experiments completed. Collecting results...")
    collect_results(results_dir)

def collect_results(results_dir):
    """
    Collect and summarize results from all experiments.
    
    Args:
        results_dir (str): Directory containing experiment results
    """
    # Find all experiment directories
    experiment_dirs = []
    base_dir = "experiments"  # Default save directory in main.py
    
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
                "val_accuracy": summary["best_metrics"]["val_acc"],
                "val_f1": summary["best_metrics"]["val_f1"],
                "val_precision": summary["best_metrics"]["val_precision"],
                "val_recall": summary["best_metrics"]["val_recall"],
                "top1_error": summary["best_metrics"]["val_top1_error"],
                "top2_error": summary["best_metrics"]["val_top2_error"],
                "params_count": summary["model_stats"]["params_count"],
                "flops": summary["model_stats"]["flops"],
                "inference_time_ms": summary["model_stats"]["inference_time_ms"],
                "model_size_mb": summary["model_stats"]["model_size_mb"],
                "batch_size": summary["config"].get("batch_size", None),
                "input_resolution": summary["config"].get("input_resolution", None),
                "optimizer": summary["config"].get("optimizer", None),
                "learning_rate": summary["config"].get("lr", None),
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
    with open(summary_file, "w") as f:
        f.write("========== POST-TORNADO DAMAGE RECOGNITION EXPERIMENTS SUMMARY ==========\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {len(df)}\n\n")
        
        f.write("TOP 5 MODELS BY F1 SCORE:\n")
        for i, row in df.head(5).iterrows():
            f.write(f"{i+1}. {row['model_name']} - {row['experiment_name']}\n")
            f.write(f"   F1 Score: {row['val_f1']:.4f}, Accuracy: {row['val_accuracy']:.4f}\n")
            f.write(f"   Params: {row['params_count']:,}, Inference Time: {row['inference_time_ms']:.2f} ms\n")
            f.write(f"   Config: BS={row['batch_size']}, IR={row['input_resolution']}, Opt={row['optimizer']}, LR={row['learning_rate']}\n")
            f.write(f"   Data Aug: {row['data_augmentation']}, Activation: {row['activation']}\n\n")
        
        f.write("\nFASTEST MODELS (TOP 3):\n")
        for i, row in df.sort_values("inference_time_ms").head(3).iterrows():
            f.write(f"{i+1}. {row['model_name']} - {row['experiment_name']}\n")
            f.write(f"   Inference Time: {row['inference_time_ms']:.2f} ms, F1 Score: {row['val_f1']:.4f}\n\n")
        
        f.write("\nSMALLEST MODELS (TOP 3):\n")
        for i, row in df.sort_values("params_count").head(3).iterrows():
            f.write(f"{i+1}. {row['model_name']} - {row['experiment_name']}\n")
            f.write(f"   Params: {row['params_count']:,}, F1 Score: {row['val_f1']:.4f}\n\n")
    
    print(f"Results saved to {results_file}")
    print(f"Summary report saved to {summary_file}")

def main():
    """Main function to run experiments."""
    args = parse_args()
    
    # Load experiment configurations
    experiments = load_experiment_configs(args.config_file)
    print(f"Loaded {len(experiments)} experiment configurations")
    
    # Run experiments
    run_experiments(experiments, args.results_dir, args.parallel)

if __name__ == "__main__":
    main()