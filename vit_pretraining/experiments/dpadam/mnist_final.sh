#!/bin/bash

DATASET=${DATASET:-MNIST}
PHYSICAL_BS=${PHYSICAL_BS:-100}     # Maximum number of samples loaded at one time
LR=${LR:-0.001}
C=${C:-10.0}

EPSILONS=(1.0 2.0 4.0 8.0)

mkdir -p output_logs_vit_final/dpadam

for DP_EPS in "${EPSILONS[@]}"; do

    TAG=dpadam-$DATASET-dpeps$DP_EPS-lr$LR-C$C
    OUT_FILE="output_logs_vit_final/dpadam/${TAG}.txt"

    NUM_GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    if [[ $NUM_GPU > 1 ]]; then
        # Randomly set a port number
        # If you encounter "address already used" error, just run again or manually set an available port id.
        PORT_ID=$(expr $RANDOM + 1000)

        # Allow multiple threads
        export OMP_NUM_THREADS=4

        # Make sure torchrun from env is called
        $CONDA_PREFIX/bin/torchrun --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
            --dp \
            --epsilon $DP_EPS \
            --clip_C $C \
            --physical_bs $PHYSICAL_BS \
            --lr $LR \
            --dataset $DATASET \
            --log_file $OUT_FILE  \
            --use_val False
    else
        python train.py \
            --dp \
            --epsilon $DP_EPS \
            --clip_C $C \
            --physical_bs $PHYSICAL_BS \
            --lr $LR \
            --dataset $DATASET \
            --log_file $OUT_FILE \
            --use_val False
    fi
done