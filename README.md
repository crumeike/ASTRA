# ASTRA: Architectural inSights for post-Tornado damAge Recognition
 
A PyTorch framework for evaluating and optimizing deep learning models for
post-tornado building damage recognition.
 
- Paper (DOI): https://doi.org/10.1016/j.eswa.2026.133915
  
## Status
 
This repository is being updated to match the published version of ASTRA. The
core framework (architectures, optimization sweeps, and metrics) description is
available now in the README.md file. The following components from the paper
are being added and will appear here soon:
 
- Frame-disjoint (viewpoint-group) data splitting used in the scene-level
  sensitivity analysis (Section 6.5, Appendix A.3).
- Cross-event zero-shot evaluation on the Tuscaloosa-Moore Tornado Damage
  (TMTD) dataset (Stage 3).
- Places365 pre-training paths for the scene-centric vs. object-centric
  comparison.
- The five best-performing QSTD model checkpoints used in the TMTD evaluation.

## Overview
 
ASTRA provides a systematic framework to evaluate convolutional and
attention-based architectures and optimization strategies for post-tornado
damage recognition. It supports:
 
- Training and evaluating a large suite of model architectures.
- Controlled hyperparameter experiments and ablation studies.
- Ordinal and standard classification metrics with confusion matrices.
- Resumable, large-scale experiment runs with checkpointing.
  
The study evaluates 79 open-source models (67 CNNs and 12 Vision Transformers)
across 38 hyperparameter variations in 2,420 controlled experiments on the
Quad-State Tornado Damage (QSTD) benchmark, with cross-event validation on the
Tuscaloosa-Moore Tornado Damage (TMTD) dataset.
 
## Damage Taxonomy
 
Classification follows the component-level IN-CORE taxonomy for residential
wood-frame structures, extended with an undamaged and a debris class:
 
| Label | Class       | Description                                             |
| ----- | ----------- | ------------------------------------------------------- |
| DS0   | Undamaged   | No visible tornado-induced structural damage.           |
| DS1   | Slight      | Minor roof-covering or single window/door failure.      |
| DS2   | Moderate    | Multiple component failures, partial roof-covering loss.|
| DS3   | Extensive   | Major roof-covering and sheathing loss.                 |
| DS4   | Complete    | Roof-to-wall connection failure or collapse.            |
| DS5   | Debris      | Non-structural: obstruction, debris, or no building.    |


## File Structure

```
├── main.py                    # Main entry point for single experiments (coming soon)
├── run_experiment.py          # Script for running multiple experiment configurations (coming soon)
├── data_utils.py              # Data loading and augmentation utilities (coming soon)
├── models.py                  # Model architecture definitions (coming soon)
├── training.py                # Model training and validation functions (coming soon)
├── metrics.py                 # Performance metrics calculation (coming soon)
├── utils.py                   # General utilities and experiment tracking (coming soon)
├── experiment_configs.json    # Example configuration for experiments 
└── run_example.sh             # Example script to run a single experiment
```

## Installation

```bash
git clone https://github.com/crumeike/ASTRA.git
cd ASTRA
pip install -r requirements.txt
```

## Dataset Structure
 
The framework expects an ImageFolder-style layout with the six damage classes:
 
```
data/
├── train/
│   ├── DS0-Undamaged/
│   ├── DS1-Slight/
│   ├── DS2-Moderate/
│   ├── DS3-Extensive/
│   ├── DS4-Complete/
│   └── DS5-Debris/
├── val/
│   └── (same six subfolders)
└── test/
    └── (same six subfolders)
```

## Usage

### Running a Single Experiment

To run a single experiment with specific configurations:

```bash
python main.py \
  --experiment_name "resnet50_baseline" \
  --model "resnet50" \
  --pretrained \
  --data_dir "path/to/data" \
  --input_size 224 \
  --batch_size 64 \
  --epochs 100 \
  --optimizer "adam" \
  --lr 0.001
```

See `run_example.sh` for a complete example.

### Running Multiple Experiments

To run multiple experiments with different configurations:

1. Create a configuration file (see `experiment_configs.json` for an example)
2. Run the experiment runner:

```bash
python run_experiment.py --config_file experiment_configs.json --results_dir experiment_results
```

### Running Ablation Studies

To understand the importance of different components:

```bash
python ablation_experiment.py \
  --base_config base_config.json \
  --ablation_factors data_augmentation regularization activation \
  --results_dir ablation_results
```

## Experiment Configuration

The framework supports the following configurations:

### Model Architectures (79 variants across 16 families)
- ResNet (18, 34, 50, 101, 152)
- VGG (11, 13, 16, 19)
- EfficientNet (B0-B7)
- Vision Transformers (ViT), AlexNet, SqueezeNet, GoogLeNet, RegNet
- ConvNeXt, DenseNet, MobileNet, ResNeXt, Wide ResNet, Swin, MaxViT, ShuffleNet


### Optimization Strategies
- Data augmentation (none, basic, standard, advanced, heavy)
- Input resolutions (224, 256, 384, 448, 512)
- Regularization techniques (weight decay, dropout)
- Activation functions (ReLU, Leaky ReLU, GELU, Swish, ELU)
- Loss functions (Cross Entropy, Focal Loss, weighted CE)
- Optimizers (SGD, Adam, AdamW)
- Learning rate schedules (step, cosine annealing)

## Evaluation Metrics

The framework automatically calculates and visualizes:

- Top-1 and Top-2 Error (standard and Ordinal)
- Precision, Recall, and F1 Score
- Training and validation curves
- Confusion matrices (standard and Ordinal)
- Model size (parameters count)
- GFLOPs (computational complexity)
- Inference time

## Results and Visualization

All experiment results are automatically saved to the specified output directory, including:

- Training and validation metrics (CSV)
- Learning curves and performance plots (PNG)
- Confusion matrix visualization (standard and Ordinal)
- Experiment configuration (JSON)
- Model weights (PyTorch checkpoint)
- Comprehensive experiment summary

## Regularization and Dropout Options for Tornado Damage Models

The experimental framework includes comprehensive support for model regularization through various dropout implementations, designed to improve model generalization and prevent overfitting on damage state classification tasks.

### Dropout Implementation

The framework supports multiple types of dropout regularization:

#### 1. Standard Dropout

Applied to fully connected layers in the model, randomly zeroing elements during training to prevent co-adaptation of neurons.

```python
# Example configuration with standard dropout
python main.py --model resnet50 --dropout_rate 0.3 --dropout_type standard
```

#### 2. Spatial Dropout

Drops entire feature maps instead of individual features, particularly effective for convolutional networks analyzing spatial data like tornado damage imagery.

```python
# Example configuration with spatial dropout
python main.py --model resnet50 --dropout_rate 0.3 --dropout_type spatial
```

#### 3. Feature Dropout

A variant that applies dropout at the feature level in convolutional layers, maintaining spatial coherence while providing regularization benefits.

```python
# Example configuration with feature dropout
python main.py --model resnet50 --dropout_rate 0.3 --dropout_type feature
```

### Configuration Options

The framework provides flexible configuration of dropout parameters:

| Parameter | Description | Supported Values |
|-----------|-------------|------------------|
| `dropout_rate` | Probability of zeroing elements | 0.0 to 1.0 |
| `dropout_type` | Type of dropout implementation | 'standard', 'spatial', 'feature' |

### Architecture-Specific Implementations

Dropout is intelligently applied based on the neural network architecture:

- **ResNet Models**: Dropout applied before the final classification layer
- **VGG Models**: Dropout inserted between fully connected layers
- **EfficientNet & MobileNet**: Classifier dropout with adjusted rates
- **Transformer Models**: Dropout on attention output and classifier heads
- **Convolutional Models**: Optional spatial dropout after convolutional blocks

### Integration with Other Regularization Techniques

Dropout can be combined with other regularization strategies:

```python
# Combined with weight decay regularization
python main.py --model resnet50 --dropout_rate 0.3 --weight_decay 0.0001

# Combined with data augmentation
python main.py --model resnet50 --dropout_rate 0.3 --data_augmentation advanced
```


## Resumable Experiment Framework

The framework includes a robust resumable experiment runner designed to handle large-scale experimental studies with built-in recovery capabilities. This feature is especially valuable for long-running experiments that might be interrupted due to system crashes, timeouts, or manual interruptions.

### Key Features

- **Checkpoint-based Recovery**: Automatically saves the state of your experiment queue after each completed run, allowing seamless resumption from the exact point of interruption.
- **Experiment Deduplication**: Uses cryptographic hashing to uniquely identify experiment configurations, preventing duplicate runs even if experiment definitions change.
- **Graceful Interruption Handling**: Captures keyboard interrupts (Ctrl+C) and saves current progress before exiting.
- **Flexible Execution Options**: Run experiments sequentially or in parallel, with the ability to limit batch size for testing purposes.

### Usage

```bash
# Run all experiments from scratch
python run_experiments_resumable.py --config_file experiment_configs.json

# Resume experiments from where they left off
python run_experiments_resumable.py --config_file experiment_configs.json --resume

# Run a limited number of experiments (for testing)
python run_experiments_resumable.py --config_file experiment_configs.json --max_experiments 5

# Execute experiments in parallel
python run_experiments_resumable.py --config_file experiment_configs.json --parallel
```

### Comprehensive Results

The framework automatically aggregates results from all completed experiments, generating both detailed CSV datasets and human-readable summary reports that highlight:

- Top-performing models by F1 score
- Fastest models for time-sensitive applications
- Most efficient models by parameter count
- Ordinal accuracy metrics for damage assessment applications

This resumable design ensures that no computational resources are wasted on repeated experiments and that large-scale studies can be conducted reliably even in environments with limited runtime constraints.

## Citation

If you use this framework or code in your research, please cite:

```bibtex
@article{umeike2026astra,
  title     = {ASTRA: Architectural Insights for Post-Tornado Damage Recognition},
  author    = {Umeike, Robinson and Dao, Thang and Crawford, Shane and van de Lindt, John and Johnston, Blythe and Wang, Wanting and Do, Trung and Mofikoya, Ajibola and Banjara, Sarbesh and Pham, Cuong},
  journal   = {Expert Systems with Applications},
  year      = {2026},
  articleno = {133915},
  doi       = {10.1016/j.eswa.2026.133915}
}
```

## Contact
 
Robinson Umeike, crumeike@crimson.ua.edu
 
## License
 
Released under the MIT License. 
