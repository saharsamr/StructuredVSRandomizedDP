#!/bin/bash

TASK=${TASK:-SST-2}
K=${K:-512}

gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | sed 's/,$//' | awk -F',' '{print NF}')
if [ "$gpu_count" -eq 1 ]; then
    # Include gradient accumulation in memory estimate
    a=0
else
    echo "Please set CUDA_VISIBLE_DEVICES to include 1 GPU only"
    exit 1
fi

EVAL_STEP=${EVAL_STEP:-50}
MODEL=${MODEL:-roberta-large}
PRIVACY_EPS=${PRIVACY_EPS:-6.0}
PRIVACY_DELTA=${PRIVACY_DELTA:-1e-5}

LOGITS=2

EXTRA_TAG=${EXTRA_TAG:-ft-}
TAG=${TAG:-k${K}-${MODEL}-dpgrape-${EXTRA_TAG}}
echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

mkdir -p output_logs_roberta_convergence_exp

SEEDS=(42 13 21)

for SEED in "${SEEDS[@]}";
do
    GR_TAG=convergence-sst2-dpgrape-$TASK-seed$SEED-bs64-accumsteps1-subspace_r16
    OUT_FILE="output_logs_roberta_convergence_exp/${GR_TAG}.txt"
    TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=1000 TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
        bash roberta_finetuning_fewshot.sh \
        --per_device_train_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-4 \
        --eval_steps $EVAL_STEP \
        --weight_decay 0 \
        --lr_scheduler_type "constant" \
        --optimizer "adam" \
        --dp_clip_threshold 0.5 \
        --dp_epsilon $PRIVACY_EPS \
        --dp_delta $PRIVACY_DELTA \
        --dp_clip_strategy flat \
        --dpgrape True \
        --subspace_r 16 \
        --subspace_T 100 \
        --report_to none \
        --no_train True \
        --log_file $OUT_FILE 

    GR_TAG=convergence-sst2-dpzero-$TASK-seed$SEED-bs64
    OUT_FILE="output_logs_roberta_convergence_exp/${GR_TAG}.txt"

    EXTRA_TAG=${EXTRA_TAG:-ft-}
    TAG=${TAG:-k${K}-${MODEL}-dpzero-${EXTRA_TAG}}

    #TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=10000 TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
    #    bash roberta_finetuning_fewshot.sh \
    #    --per_device_train_batch_size 64 \
    #    --learning_rate 1e-6 \
    #    --eval_steps $EVAL_STEP \
    #    --weight_decay 0 \
    #    --lr_scheduler_type "constant" \
    #    --optimizer "sgd" \
    #    --zero_order_eps 1e-3 \
    #    --zero_order_optim \
    #    --dpzero_clip_threshold 200.0 \
    #    --dpzero True \
    #    --efficient_zero_order True \
    #    --report_to none \
    #    --log_file $OUT_FILE 
done