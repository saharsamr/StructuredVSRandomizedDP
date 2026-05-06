#!/bin/bash

DATASET=${DATASET:-CIFAR10}

mkdir -p output_logs_vit_timing_exp

#TAG=dpgrape-$DATASET-subspace_r64
#OUT_FILE="output_logs_vit_timing_exp/${TAG}.txt"
#python timing_exp.py \
#    --log_file $OUT_FILE \
#    --method DPGrape \
#    --subspace_r $64

TAG=naive-dpgalore-$DATASET-subspace_r64
OUT_FILE="output_logs_vit_timing_exp/${TAG}.txt"
python timing_exp.py \
    --log_file $OUT_FILE \
    --method NaiveDPGaLore \
    --subspace_r $64

TAG=dpadam-$DATASET
OUT_FILE="output_logs_vit_timing_exp/${TAG}.txt"
python timing_exp.py \
        --log_file $OUT_FILE \
        --method DPAdam

TAG=adam-$DATASET
OUT_FILE="output_logs_vit_timing_exp/${TAG}.txt"   
python timing_exp.py \
        --log_file $OUT_FILE \
        --method Adam
