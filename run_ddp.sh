#!/bin/bash

# DDP Training Launch Script
# Usage: ./run_ddp.sh [num_gpus]

NUM_GPUS=${1:-2}  # Default to 2 GPUs if not specified

echo "Launching DDP training on $NUM_GPUS GPUs..."

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    main.py

echo "Training completed!"