#!/bin/bash

DATASET=${DATASET:-CIFAR10}
PHYSICAL_BS=${PHYSICAL_BS:-100}     # Maximum number of samples loaded at one time  
DP_EPS=${DP_EPS:-2.0}

LRS=(0.00001 0.00005 0.0001 0.0005)
CS=(0.1 1.0 10.0)

mkdir -p output_logs_vit_c_lr_search/naive_dpgalore

for LR in "${LRS[@]}";
do
    for C in "${CS[@]}";
    do

        TAG=naive-dpgalore-$DATASET-dpeps$DP_EPS-lr$LR-C$C-finetuning
        OUT_FILE="output_logs_vit_c_lr_search/naive_dpgalore/${TAG}.txt"

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
                --naive_dp_galore \
                --epsilon $DP_EPS \
                --clip_C $C \
                --physical_bs $PHYSICAL_BS \
                --lr $LR \
                --dataset $DATASET \
                --log_file $OUT_FILE  \
                --use_val True \
                --finetuning \
                --epochs 20
        else
            python train.py \
                --dp \
                --naive_dp_galore \
                --epsilon $DP_EPS \
                --clip_C $C \
                --physical_bs $PHYSICAL_BS \
                --lr $LR \
                --dataset $DATASET \
                --log_file $OUT_FILE \
                --use_val True \
                --finetuning \
                --epochs 20
        fi
    done
done
