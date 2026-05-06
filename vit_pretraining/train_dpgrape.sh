#!/bin/bash

DATASET=${DATASET:-CIFAR10}
PHYSICAL_BS=${PHYSICAL_BS:-100}     # Number of samples loaded at one time
LR=${LR:-1e-3}   

NUM_GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [[ $NUM_GPU > 1 ]]; then
    # Randomly set a port number
    # If you encounter "address already used" error, just run again or manually set an available port id.
    PORT_ID=$(expr $RANDOM + 1000)

    # Allow multiple threads
    export OMP_NUM_THREADS=4

    # Make sure torchrun from env is called
    $CONDA_PREFIX/bin/torchrun --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py --physical_bs $PHYSICAL_BS --lr $LR --dp --rand_proj gaussian --dataset $DATASET
else
    python train.py --physical_bs $PHYSICAL_BS --lr $LR --dp --rand_proj gaussian --dataset $DATASET
fi