# Post-Tornado Damage Recognition Experimental Framework

A comprehensive PyTorch-based framework for evaluating and optimizing deep learning models for post-tornado damage recognition.

## Overview

This framework provides a systematic approach to evaluate various convolutional neural network architectures and optimization strategies for post-tornado damage recognition. It includes tools for:

- Training and evaluating multiple model architectures
- Conducting controlled experiments across hyperparameters
- Performing ablation studies to identify critical components
- Generating comprehensive performance metrics and visualizations
- Class activation mapping (CAM) for model interpretability

## File Structure

```
├── main.py                    # Main entry point for single experiments
├── run_experiment.py          # Script for running multiple experiment configurations
├── ablation_experiment.py     # Script for conducting ablation studies
├── data_utils.py              # Data loading and augmentation utilities
├── models.py                  # Model architecture definitions
├── training.py                # Model training and validation functions
├── metrics.py                 # Performance metrics calculation
├── utils.py                   # General utilities and experiment tracking
├── cam_utils.py               # Class Activation Mapping utilities
├── experiment_configs.json    # Example configuration for experiments
└── run_example.sh             # Example script to run a single experiment
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tornado-damage-recognition.git
cd tornado-damage-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Alternatively, install individual packages:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow opencv-python tensorboard thop torchsummary kornia
```

## Dataset Structure

The framework expects the dataset to be organized as follows:

```
data/
├── train/
│   ├── undamaged/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── slight/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── moderate/
│   │   ├── image1.jpg
│   │   └── ...
│   └── extensive/
│       ├── image1.jpg
│       └── ...
└── val/
│   ├── undamaged/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── slight/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── moderate/
│   │   ├── image1.jpg
│   │   └── ...
│   └── extensive/
│       ├── image1.jpg
│       └── ...
└── test/
    ├── undamaged/
    │   ├── image1.jpg
    │   └── ...
    ├── slight/
    │   ├── image1.jpg
    │   └── ...
    ├── moderate/
    │   ├── image1.jpg
    │   └── ...
    └── extensive/
        ├── image1.jpg
        └── ...
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

### Model Architectures
- ResNet (18, 34, 50, 101, 152)
- VGG (11, 13, 16, 19)
- EfficientNet (B0-B7)
- Vision Transformers (ViT)
- ConvNeXt, DenseNet, MobileNet, etc.

### Optimization Strategies
- Data augmentation (basic, standard, advanced)
- Input resolutions (224, 256, 384, 448, 512)
- Regularization techniques (weight decay, dropout)
- Activation functions (ReLU, Leaky ReLU, GELU, Swish)
- Loss functions (Cross Entropy, Focal Loss)
- Optimizers (SGD, Adam, AdamW)
- Learning rate schedules (step, cosine)

## Evaluation Metrics

The framework automatically calculates and visualizes:

- Top-1 and Top-2 Error (standard and Ordinal)
- Precision, Recall, and F1 Score
- Training and validation curves
- Confusion matrices (standard and Ordinal)
- Model size (parameters count)
- GFLOPs (computational complexity)
- Inference time

## Class Activation Mapping

The framework includes tools for model interpretability:

```python
from cam_utils import GradCAM, visualize_cam
from models import create_model, find_target_layer

# Load model
model = create_model("resnet50", num_classes=4)
target_layer = find_target_layer(model)

# Create GradCAM instance
grad_cam = GradCAM(model, target_layer)

# Generate and visualize heatmap
cam = grad_cam(input_image, class_idx=None)  # None uses predicted class
visualization = visualize_cam(input_image, cam, output_path="cam_visualization.png")
```

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

### Experimental Results

Our ablation studies show that model performance on damage state classification can be significantly improved through the appropriate selection of dropout parameters:

- Standard dropout rates between 0.2-0.3 typically yield the best results for ResNet architectures
- Spatial dropout at 0.2 can improve EfficientNet performance on complex damage patterns
- Vision Transformers benefit from 0.3-0.5 dropout in attention layers when training data is limited

The framework's automatic hyperparameter search can identify optimal regularization settings for each architecture and dataset combination.

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