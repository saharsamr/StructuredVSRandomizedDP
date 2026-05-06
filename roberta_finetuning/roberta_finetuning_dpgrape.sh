#!/bin/bash

TASK=${TASK:-SST-2}
K=${K:-512}
SEED=${SEED:-42}
PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-64}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
LR=${LR:-5e-4}
WD=${WD:-0}
STEP=${STEP:-100}
EVAL_STEP=${EVAL_STEP:-10000}
MODEL=${MODEL:-roberta-large}

DP_CLIP_THRESHOLD=${DP_CLIP_THRESHOLD:-10.0}
PRIVACY_EPS=${PRIVACY_EPS:-6.0}
PRIVACY_DELTA=${PRIVACY_DELTA:-1e-5}

SUBSPACE_R=${SUBSPACE_R:-16}
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

GR_TAG=dpgrape-$TASK-seed$SEED-bs$BS-lr$LR-dpeps$PRIVACY_EPS-dpdelta$PRIVACY_DELTA-totalsteps$STEP-evalstep$EVAL_STEP-subspace_r$SUBSPACE_R-subspace_T$SUBSPACE_T
OUT_FILE="output_logs/${GR_TAG}.txt"

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
    --dp_clip_threshold $DP_CLIP_THRESHOLD \
    --dp_epsilon $PRIVACY_EPS \
    --dp_delta $PRIVACY_DELTA \
    --dpgrape True \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --subspace_r $SUBSPACE_R \
    --subspace_T $SUBSPACE_T \
    --report_to none \
    --log_file $OUT_FILE \
    --no_train False \
    --dp_clip_strategy flat 
