#!/bin/bash

# DDP Training Launch Script
# Usage: ./run_ddp.sh [num_gpus] [arg1] [arg2] ...
# Example: ./run_ddp.sh 2 --batch_size=32 --n_layers=12

# The first argument is the number of GPUs
NUM_GPUS=${1:-2}

# This shifts the arguments, so $@ will now contain everything *after* the first argument
shift 

echo "Launching DDP training on $NUM_GPUS GPUs..."

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    scripts/fine_tune.py "$@" # Pass all remaining arguments to the python script

echo "Training completed!"