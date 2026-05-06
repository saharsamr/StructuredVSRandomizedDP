#!/bin/bash

TASK=${TASK:-trec}
K=${K:-512}
SEED=${SEED:-100}

# Determine number of GPUs to set GRAD_ACCUM_STEPS and PER_DEVICE_TRAIN_BS to get total batch size of 64
gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | sed 's/,$//' | awk -F',' '{print NF}')
echo "CUDA_VISIBLE_DEVICES GPU Count:$gpu_count"

if [ "$gpu_count" -eq 1 ]; then
    PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-64}
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
elif [ "$gpu_count" -eq 2 ]; then
    PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-32}
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
elif [ "$gpu_count" -eq 4 ]; then
    PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-16}
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
elif [ "$gpu_count" -eq 8 ]; then
    PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-8}
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
elif [ "$gpu_count" -eq 16 ]; then
    PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-4}
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
else
    echo "Please set CUDA_VISIBLE_DEVICES to include 1, 2, 4, 8, or 16 GPUs"
    exit 1
fi

LR=${LR:-5e-4}
EPS=${EPS:-1e-3}
WD=${WD:-0}
STEP=${STEP:-1000}
EVAL_STEP=${EVAL_STEP:-10000}
MODEL=${MODEL:-roberta-large}

PRIVACY_EPS=${PRIVACY_EPS:-6.0}
PRIVACY_DELTA=${PRIVACY_DELTA:-1e-5}

#SUBSPACE_R=${SUBSPACE_R:-16}
SUBSPACE_T=${SUBSPACE_T:-100}

if [ "$TASK" = "SNLI" ]; then
    LOGITS=3
elif [ "$TASK" = "MNLI" ]; then
    LOGITS=3
elif [ "$TASK" = "trec" ]; then
    LOGITS=6
elif [ "$TASK" = "SST-5" ]; then
    LOGITS=5
else
    LOGITS=2
fi

NUM_GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
BS=$((PER_DEVICE_TRAIN_BS * GRAD_ACCUM_STEPS * NUM_GPU))

Cs=(0.1 0.5 1.0 5.0 10.0 20.0)
#rs=(4 8 32 64 128 256)
rs=(256)

mkdir -p output_logs_roberta_C_search_r_ablation

for r in "${rs[@]}";
do
    for C in "${Cs[@]}";
    do

        GR_TAG=dpgrape-$TASK-seed$SEED-bs$BS-lr$LR-dpeps$PRIVACY_EPS-dpdelta$PRIVACY_DELTA-dpC$C-totalsteps$STEP-evalstep$EVAL_STEP-subspace_r$r-subspace_T$SUBSPACE_T
        OUT_FILE="output_logs_roberta_C_search_r_ablation/${GR_TAG}.txt"

        EXTRA_TAG=${EXTRA_TAG:-ft-}
        TAG=${TAG:-k${K}-${MODEL}-dpgrape-${EXTRA_TAG}}
        echo "Grid search tag: $GR_TAG"
        echo "Tag: $TAG"

        TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
            bash roberta_finetuning_fewshot.sh \
            --per_device_train_batch_size $PER_DEVICE_TRAIN_BS \
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
            --subspace_r $r \
            --subspace_T $SUBSPACE_T \
            --report_to none \
            --log_file $OUT_FILE 
    done
done