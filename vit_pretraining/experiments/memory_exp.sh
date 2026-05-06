#!/bin/bash

DATASET=${DATASET:-CIFAR10}

mkdir -p output_logs_vit_memory_exp

BATCH_SIZES=(1 10 25 50 75 100 125 150 175 200 225 250 275 300 350 400 450 500 550 600 650 700)
SUBSPACE_RS=(4 16 64 256)
for BS in ${BATCH_SIZES[@]}; do
    # Run max memory exp for dpgrape with different rs, naive dp galore with different rs, dpadam, adam
    for R in ${SUBSPACE_RS[@]}; do
        TAG=dpgrape-$DATASET-bs$BS-subspace_r$R
        OUT_FILE="output_logs_vit_memory_exp/${TAG}.txt"
        python max_memory_usage_exp.py \
            --log_file $OUT_FILE \
            --batch_size $BS \
            --method DPGrape \
            --subspace_r $R

        TAG=naive-dpgalore-$DATASET-bs$BS-subspace_r$R
        OUT_FILE="output_logs_vit_memory_exp/${TAG}.txt"
        python max_memory_usage_exp.py \
            --log_file $OUT_FILE \
            --batch_size $BS \
            --method NaiveDPGaLore \
            --subspace_r $R
    done

    TAG=dpadam-$DATASET-bs$BS
    OUT_FILE="output_logs_vit_memory_exp/${TAG}.txt"
    python max_memory_usage_exp.py \
            --log_file $OUT_FILE \
            --batch_size $BS \
            --method DPAdam

    TAG=adam-$DATASET-bs$BS
    OUT_FILE="output_logs_vit_memory_exp/${TAG}.txt"   
    python max_memory_usage_exp.py \
            --log_file $OUT_FILE \
            --batch_size $BS \
            --method Adam
done
