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
- VGG (16, 19)
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

- Top-1 and Top-2 Error
- Precision, Recall, and F1 Score
- Training and validation curves
- Confusion matrices
- Model size (parameters count)
- FLOPs (computational complexity)
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
- Confusion matrix visualization
- Experiment configuration (JSON)
- Model weights (PyTorch checkpoint)
- Comprehensive experiment summary

## Citation

If you use this framework in your research, please cite:

```
@article{yourarticle2025,
  title={Experimental Framework for Post-Tornado Damage Recognition},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.