#!/bin/bash

TASK=${TASK:-SST-2}
SEED=${SEED:-100}
C=${C:-1.0}
K=${K:-512}

gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | sed 's/,$//' | awk -F',' '{print NF}')
if [ "$gpu_count" -eq 1 ]; then
    a=0
else
    echo "Please set CUDA_VISIBLE_DEVICES to include 1 GPU only"
    exit 1
fi

LR=${LR:-5e-4}
EPS=${EPS:-1e-3}
WD=${WD:-0}
STEP=${STEP:-50}
EVAL_STEP=${EVAL_STEP:-10000}
MODEL=${MODEL:-roberta-large}

# Logs and results also stream to Weights & Biases; set USE_WANDB=false to turn that off (see README).
USE_WANDB=${USE_WANDB:-true}

PRIVACY_EPS=${PRIVACY_EPS:-6.0}
PRIVACY_DELTA=${PRIVACY_DELTA:-1e-5}

LOGITS=2

# Get results for dpgrape (r=16), then dpadam, adam, dpzero

GR_TAG=memory-sst2-dpgrape-$TASK-bs64-accumsteps1-subspace_r16
OUT_FILE="output_logs_roberta_timing_exp/${GR_TAG}.txt"

EXTRA_TAG=${EXTRA_TAG:-ft-}
TAG=${TAG:-k${K}-${MODEL}-dpgrape-${EXTRA_TAG}}
echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

mkdir -p output_logs_roberta_timing_exp

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
    bash roberta_finetuning_fewshot.sh \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
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
    --subspace_r 16 \
    --subspace_T 100 \
    --use_wandb $USE_WANDB \
    --report_to none \
    --log_file $OUT_FILE 

GR_TAG=memory-sst2-dpadam-$TASK-bs32-accumsteps-2
OUT_FILE="output_logs_roberta_timing_exp/${GR_TAG}.txt"

EXTRA_TAG=${EXTRA_TAG:-ft-}
TAG=${TAG:-k${K}-${MODEL}-dpadam-${EXTRA_TAG}}
echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
    bash roberta_finetuning_fewshot.sh \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate $LR \
    --eval_steps $EVAL_STEP \
    --weight_decay $WD \
    --lr_scheduler_type "constant" \
    --optimizer "adam" \
    --dp_clip_threshold $C \
    --dp_epsilon $PRIVACY_EPS \
    --dp_delta $PRIVACY_DELTA \
    --dp_clip_strategy flat \
    --dpadam True \
    --use_wandb $USE_WANDB \
    --report_to none \
    --log_file $OUT_FILE 

GR_TAG=memory-sst2-adam-$TASK-bs64
OUT_FILE="output_logs_roberta_timing_exp/${GR_TAG}.txt"

EXTRA_TAG=${EXTRA_TAG:-ft-}
TAG=${TAG:-k${K}-${MODEL}-adam-${EXTRA_TAG}}

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
    bash roberta_finetuning_fewshot.sh \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate $LR \
    --eval_steps $EVAL_STEP \
    --weight_decay $WD \
    --lr_scheduler_type "constant" \
    --optimizer "adam" \
    --use_wandb $USE_WANDB \
    --report_to none \
    --log_file $OUT_FILE 

GR_TAG=memory-sst2-dpzero-$TASK-bs64
OUT_FILE="output_logs_roberta_timing_exp/${GR_TAG}.txt"

EXTRA_TAG=${EXTRA_TAG:-ft-}
TAG=${TAG:-k${K}-${MODEL}-dpzero-${EXTRA_TAG}}

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
    bash roberta_finetuning_fewshot.sh \
    --per_device_train_batch_size 64 \
    --learning_rate $LR \
    --eval_steps $EVAL_STEP \
    --weight_decay $WD \
    --lr_scheduler_type "constant" \
    --optimizer "sgd" \
    --zero_order_eps $EPS \
    --zero_order_optim \
    --dpzero_clip_threshold $C \
    --dpzero True \
    --efficient_zero_order True \
    --use_wandb $USE_WANDB \
    --report_to none \
    --log_file $OUT_FILE 
