#!/bin/bash

# Example script to run a single experiment with the framework

# Set data directory (replace with actual data path)
DATA_DIR="./data/tornado_damage"

# Create experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="resnet50_baseline_${TIMESTAMP}"

# Run experiment
python main.py \
  --experiment_name ${EXP_NAME} \
  --save_dir "experiments" \
  --seed 42 \
  --model "resnet50" \
  --pretrained \
  --activation "relu" \
  --data_dir ${DATA_DIR} \
  --input_size 224 \
  --batch_size 64 \
  --data_augmentation "basic" \
  --epochs 100 \
  --patience 10 \
  --optimizer "adam" \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --scheduler "cosine" \
  --loss "cross_entropy"

# Print completion message
echo "Experiment ${EXP_NAME} completed!"