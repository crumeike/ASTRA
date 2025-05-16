#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for conducting ablation studies on post-tornado damage recognition models.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
from collections import OrderedDict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ablation studies for post-tornado damage recognition')
    
    # Base experiment settings
    parser.add_argument('--base_config', type=str, required=True,
                        help='Path to JSON file with base experiment configuration')
    parser.add_argument('--ablation_factors', type=str, nargs='+', required=True,
                        help='Factors to ablate (e.g. data_augmentation activation regularization)')
    parser.add_argument('--results_dir', type=str, default='ablation_results',
                        help='Directory to save ablation results')
    
    return parser.parse_args()


def load_base_config(config_file):
    """
    Load base experiment configuration from JSON file.
    
    Args:
        config_file (str): Path to JSON configuration file
    
    Returns:
        dict: Base experiment configuration
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def generate_ablation_configs(base_config, ablation_factors):
    """
    Generate configurations for ablation experiments.
    
    Args:
        base_config (dict): Base experiment configuration
        ablation_factors (list): Factors to ablate
    
    Returns:
        list: List of ablation experiment configurations
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_exp_name = f"{base_config.get('model', 'model')}_{timestamp}"
    
    # Create full configuration (baseline)
    full_config = base_config.copy()
    full_config['experiment_name'] = f"{base_exp_name}_full"
    
    # Create configurations with one factor ablated at a time
    ablation_configs = [full_config]
    
    for factor in ablation_factors:
        # Create ablated configuration
        ablated_config = base_config.copy()
        
        # Apply ablation based on factor type
        if factor == 'data_augmentation':
            # No data augmentation
            ablated_config['data_augmentation'] = 'none'
            
        elif factor == 'activation':
            # Default activation (ReLU)
            ablated_config['activation'] = 'relu'
            
        elif factor == 'regularization':
            # No regularization
            ablated_config['weight_decay'] = 0.0
            
        elif factor == 'pretrained':
            # No pretrained weights
            ablated_config['pretrained'] = False
            
        elif factor == 'input_size':
            # Smaller input size
            ablated_config['input_size'] = 224
            
        elif factor == 'optimizer':
            # Default optimizer
            ablated_config['optimizer'] = 'adam'
            ablated_config['lr'] = 0.001
            
        elif factor == 'loss':
            # Default loss
            ablated_config['loss'] = 'cross_entropy'
        
        # Set experiment name
        ablated_config['experiment_name'] = f"{base_exp_name}_no_{factor}"
        ablation_configs.append(ablated_config)
    
    return ablation_configs


def run_ablation_experiments(configs):
    """
    Run ablation experiments with different configurations.
    
    Args:
        configs (list): List of experiment configurations
    """
    for i, config in enumerate(configs):
        # Create command with all parameters
        cmd = ["python", "main.py"]
        
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
        # Run experiment
        print(f"\n[{i+1}/{len(configs)}] Running experiment: {config['experiment_name']}")
        print(" ".join(cmd))
        subprocess.run(cmd)


def collect_ablation_results(results_dir):
    """
    Collect and analyze results from ablation experiments.
    
    Args:
        results_dir (str): Directory to save results
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
            
            # Extract experiment info
            experiment_name = summary["experiment_name"]
            
            # Check if this is an ablation experiment
            if "_full" in experiment_name or "_no_" in experiment_name:
                # Extract relevant metrics
                result = {
                    "experiment_name": experiment_name,
                    "model_name": summary["model_name"],
                    "val_accuracy": summary["best_metrics"]["val_acc"],
                    "val_f1": summary["best_metrics"]["val_f1"],
                    "val_precision": summary["best_metrics"]["val_precision"],
                    "val_recall": summary["best_metrics"]["val_recall"],
                    "top1_error": summary["best_metrics"]["val_top1_error"],
                    "top2_error": summary["best_metrics"]["val_top2_error"],
                    "inference_time_ms": summary["model_stats"]["inference_time_ms"]
                }
                
                # Determine experiment condition
                if "_full" in experiment_name:
                    result["condition"] = "Full Model"
                    base_name = experiment_name.split("_full")[0]
                    result["base_name"] = base_name
                elif "_no_" in experiment_name:
                    parts = experiment_name.split("_no_")
                    base_name = parts[0]
                    ablated_factor = parts[1]
                    result["condition"] = f"No {ablated_factor.replace('_', ' ').title()}"
                    result["ablated_factor"] = ablated_factor
                    result["base_name"] = base_name
                
                results.append(result)
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
    
    # Group results by base experiment name
    grouped_results = {}
    for result in results:
        base_name = result.get("base_name")
        if base_name:
            if base_name not in grouped_results:
                grouped_results[base_name] = []
            grouped_results[base_name].append(result)
    
    # Process each group of experiments
    all_ablation_results = []
    
    for base_name, group in grouped_results.items():
        # Find full model result
        full_model = next((r for r in group if r["condition"] == "Full Model"), None)
        
        if full_model:
            # Calculate relative performance for ablated models
            for result in group:
                if result["condition"] != "Full Model":
                    # Calculate relative performance decrease
                    result["rel_f1_change"] = (result["val_f1"] - full_model["val_f1"]) / full_model["val_f1"] * 100
                    result["rel_acc_change"] = (result["val_accuracy"] - full_model["val_accuracy"]) / full_model["val_accuracy"] * 100
                    all_ablation_results.append(result)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create DataFrame and sort by relative performance change
    df = pd.DataFrame(all_ablation_results)
    if not df.empty:
        df = df.sort_values("rel_f1_change")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"ablation_results_{timestamp}.csv")
        df.to_csv(results_file, index=False)
        
        # Generate ablation plots
        plot_ablation_results(df, results_dir, timestamp)
        
        # Generate summary report
        generate_ablation_report(df, full_model, results_dir, timestamp)
    else:
        print("No ablation results found.")


def plot_ablation_results(df, results_dir, timestamp):
    """
    Generate visualization of ablation study results.
    
    Args:
        df (pandas.DataFrame): DataFrame with ablation results
        results_dir (str): Directory to save plots
        timestamp (str): Timestamp for file naming
    """
    # 1. F1 Score Change by Ablated Factor
    plt.figure(figsize=(12, 6))
    sns.barplot(x='ablated_factor', y='rel_f1_change', data=df, palette='coolwarm')
    plt.title('Relative F1 Score Change by Ablated Factor', fontsize=14)
    plt.xlabel('Ablated Factor')
    plt.ylabel('Relative Change (%)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ablation_f1_change_{timestamp}.png"))
    plt.close()
    
    # 2. Accuracy Change by Ablated Factor
    plt.figure(figsize=(12, 6))
    sns.barplot(x='ablated_factor', y='rel_acc_change', data=df, palette='coolwarm')
    plt.title('Relative Accuracy Change by Ablated Factor', fontsize=14)
    plt.xlabel('Ablated Factor')
    plt.ylabel('Relative Change (%)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ablation_acc_change_{timestamp}.png"))
    plt.close()
    
    # 3. Combined Metrics Heatmap (if enough factors)
    if len(df) >= 3:
        # Pivot data for heatmap
        metrics = ['rel_f1_change', 'rel_acc_change', 'top1_error', 'top2_error']
        heatmap_data = []
        
        for factor in df['ablated_factor'].unique():
            factor_data = df[df['ablated_factor'] == factor]
            if not factor_data.empty:
                row_data = {'Factor': factor}
                for metric in metrics:
                    row_data[metric] = factor_data[metric].values[0]
                heatmap_data.append(row_data)
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df = heatmap_df.set_index('Factor')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
        plt.title('Impact of Ablation on Different Metrics', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"ablation_heatmap_{timestamp}.png"))
        plt.close()


def generate_ablation_report(df, full_model, results_dir, timestamp):
    """
    Generate a text report summarizing ablation study results.
    
    Args:
        df (pandas.DataFrame): DataFrame with ablation results
        full_model (dict): Full model results
        results_dir (str): Directory to save report
        timestamp (str): Timestamp for file naming
    """
    report_file = os.path.join(results_dir, f"ablation_report_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        f.write("========== POST-TORNADO DAMAGE RECOGNITION ABLATION STUDY ==========\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Model: {full_model['model_name']}\n\n")
        
        f.write("BASELINE PERFORMANCE (FULL MODEL):\n")
        f.write(f"F1 Score: {full_model['val_f1']:.4f}\n")
        f.write(f"Accuracy: {full_model['val_accuracy']:.4f}\n")
        f.write(f"Top-1 Error: {full_model['top1_error']:.4f}\n")
        f.write(f"Top-2 Error: {full_model['top2_error']:.4f}\n\n")
        
        f.write("ABLATION RESULTS (SORTED BY IMPACT ON F1 SCORE):\n")
        
        # Sort factors by impact on F1 score
        sorted_df = df.sort_values('rel_f1_change')
        
        for _, row in sorted_df.iterrows():
            factor = row['ablated_factor']
            f.write(f"\nFactor: {factor.replace('_', ' ').title()}\n")
            f.write(f"F1 Score: {row['val_f1']:.4f} (Change: {row['rel_f1_change']:.2f}%)\n")
            f.write(f"Accuracy: {row['val_accuracy']:.4f} (Change: {row['rel_acc_change']:.2f}%)\n")
            f.write(f"Top-1 Error: {row['top1_error']:.4f}\n")
            f.write(f"Top-2 Error: {row['top2_error']:.4f}\n")
        
        f.write("\n\nSUMMARY OF FINDINGS:\n")
        
        # Get most and least important factors
        most_important = sorted_df.iloc[0]['ablated_factor'].replace('_', ' ').title()
        least_important = sorted_df.iloc[-1]['ablated_factor'].replace('_', ' ').title()
        
        f.write(f"1. Most important factor: {most_important} (F1 change: {sorted_df.iloc[0]['rel_f1_change']:.2f}%)\n")
        f.write(f"2. Least important factor: {least_important} (F1 change: {sorted_df.iloc[-1]['rel_f1_change']:.2f}%)\n")
        
        # Identify factors with positive impact (if any)
        positive_factors = sorted_df[sorted_df['rel_f1_change'] > 0]
        if not positive_factors.empty:
            f.write("\n3. Factors with positive impact (removing them improves performance):\n")
            for _, row in positive_factors.iterrows():
                f.write(f"   - {row['ablated_factor'].replace('_', ' ').title()} (F1 change: +{row['rel_f1_change']:.2f}%)\n")
        
        f.write("\nRECOMMENDATIONS:\n")
        f.write("Based on the ablation study results, we recommend:\n")
        
        # Generate recommendations based on results
        if sorted_df.iloc[0]['rel_f1_change'] < -5:
            f.write(f"1. Keep {most_important} as it significantly improves model performance.\n")
        
        if not positive_factors.empty:
            for _, row in positive_factors.iterrows():
                f.write(f"2. Consider removing {row['ablated_factor'].replace('_', ' ').title()} as it may negatively impact performance.\n")
        
        f.write("\n")
    
    print(f"Ablation report saved to {report_file}")


def main():
    """Main function for running ablation studies."""
    args = parse_args()
    
    # Load base configuration
    base_config = load_base_config(args.base_config)
    print(f"Loaded base configuration")
    
    # Generate ablation configurations
    ablation_configs = generate_ablation_configs(base_config, args.ablation_factors)
    print(f"Generated {len(ablation_configs)} ablation configurations")
    
    # Run ablation experiments
    run_ablation_experiments(ablation_configs)
    
    # Collect and analyze results
    collect_ablation_results(args.results_dir)


if __name__ == "__main__":
    main()