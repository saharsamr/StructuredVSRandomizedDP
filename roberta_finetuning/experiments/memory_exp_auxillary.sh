#!/bin/bash

# Additional experiment for memory usage of DP-GRAPE across different r

TASK=${TASK:-SST-2}
SEED=${SEED:-100}
C=${C:-1.0}
K=${K:-512}

gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | sed 's/,$//' | awk -F',' '{print NF}')
echo "CUDA_VISIBLE_DEVICES GPU Count:$gpu_count"

if [ "$gpu_count" -eq 1 ]; then
    # Include gradient accumulation in memory estimate
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
else
    echo "Please set CUDA_VISIBLE_DEVICES to include 1 GPU only"
    exit 1
fi

LR=${LR:-5e-4}
EPS=${EPS:-1e-3}
WD=${WD:-0}
STEP=${STEP:-30}
EVAL_STEP=${EVAL_STEP:-10000}
MODEL=${MODEL:-roberta-large}
PRIVACY_EPS=${PRIVACY_EPS:-6.0}
PRIVACY_DELTA=${PRIVACY_DELTA:-1e-5}

SUBSPACE_T=${SUBSPACE_T:-100}

LOGITS=2

#NUM_GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
BATCH_SIZES=(1 2 4 8 12 16 20 24 32 40 48 56 64 72 96 128)
RS=(4 8 16 32 64 128 256 512)

mkdir -p output_logs_roberta_memory_exp

for BS in "${BATCH_SIZES[@]}";
do
    # Get results for dpgrape r=4,16,64,256, then dpadam, adam, dpzero
    for SUBSPACE_R in "${RS[@]}";
    do
    
        GR_TAG=memory-auxillary-sst2-dpgrape-$TASK-bs$BS-subspace_r$SUBSPACE_R
        OUT_FILE="output_logs_roberta_memory_exp/${GR_TAG}.txt"

        EXTRA_TAG=${EXTRA_TAG:-ft-}
        TAG=${TAG:-k${K}-${MODEL}-dpgrape-${EXTRA_TAG}}
        echo "Grid search tag: $GR_TAG"
        echo "Tag: $TAG"

        TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
            bash roberta_finetuning_fewshot.sh \
            --per_device_train_batch_size $BS \
            --learning_rate $LR \
            --eval_steps $EVAL_STEP \
            --weight_decay $WD \
            --lr_scheduler_type "constant" \
            --optimizer "adam" \
            --dp_clip_threshold $C \
            --dp_epsilon $PRIVACY_EPS \
            --dp_delta $PRIVACY_DELTA \
            --dp_clip_strategy flat \
            --dpgrape True \
            --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
            --subspace_r $SUBSPACE_R \
            --subspace_T $SUBSPACE_T \
            --report_to none \
            --log_file $OUT_FILE 
    done
done
