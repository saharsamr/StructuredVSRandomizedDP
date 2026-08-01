#!/bin/bash
#
# DP-GaLore (bare-gradient SVD subspace) on RoBERTa-Large, few-shot.
#
# WARNING: the subspace is the top-r SVD of the BARE (unclipped, unnoised) batch gradient,
# recomputed every SUBSPACE_T steps. The weight updates are DP; the subspace is not, so
# this method is eps = infinity as a whole. Do not put its numbers in the same column as
# DP-GRAPE without saying so. See DPTRACK_DESIGN.md section 4.
#
# Fine-tunes and evaluates in one invocation: roberta_finetuning_fewshot.sh passes
# --do_train --do_eval --do_predict, so dev and test metrics land in the log file.
#
# Usage:  TASK=SST-2 SEED=42 C=0.5 PRIVACY_EPS=6.0 bash roberta_finetuning_dpgalore.sh

TASK=${TASK:-SST-2}
K=${K:-512}
SEED=${SEED:-42}
PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-64}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
LR=${LR:-5e-4}
WD=${WD:-0}
STEP=${STEP:-1000}
EVAL_STEP=${EVAL_STEP:-10000}
MODEL=${MODEL:-roberta-large}

# Flat clipping threshold, tuned per task by the DP-GRAPE paper (App. C.2). Keep this block
# identical across roberta_finetuning_dp{grape,galore,track}.sh -- C must be the same for all
# three methods or the comparison is not controlled.
# Override with C=<value>; DP_CLIP_THRESHOLD is kept as an alias for old callers.
case $TASK in
    SST-2)       C_DEFAULT=0.5  ;;
    sst-5|SST-5) C_DEFAULT=20.0 ;;
    SNLI)        C_DEFAULT=0.1  ;;
    MNLI)        C_DEFAULT=10.0 ;;
    RTE)         C_DEFAULT=0.5  ;;
    trec)        C_DEFAULT=0.5  ;;
    *)
        echo "WARNING: no paper-tuned C for TASK=$TASK; using 0.5. Set C=<value> to be explicit." >&2
        C_DEFAULT=0.5
        ;;
esac
C=${C:-${DP_CLIP_THRESHOLD:-$C_DEFAULT}}
PRIVACY_EPS=${PRIVACY_EPS:-6.0}
PRIVACY_DELTA=${PRIVACY_DELTA:-1e-5}

SUBSPACE_R=${SUBSPACE_R:-16}
SUBSPACE_T=${SUBSPACE_T:-100}
ORACLE_BATCH_MODE=${ORACLE_BATCH_MODE:-shared}

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

GR_TAG=dpgalore-$TASK-seed$SEED-bs$BS-lr$LR-dpeps$PRIVACY_EPS-dpdelta$PRIVACY_DELTA-dpC$C-totalsteps$STEP-evalstep$EVAL_STEP-subspace_r$SUBSPACE_R-subspace_T$SUBSPACE_T-batchmode$ORACLE_BATCH_MODE

mkdir -p output_logs
OUT_FILE="output_logs/${GR_TAG}.txt"

EXTRA_TAG=${EXTRA_TAG:-ft-}
TAG=${TAG:-k${K}-${MODEL}-dpgalore-${EXTRA_TAG}}
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
    --dpgalore True \
    --oracle_batch_mode $ORACLE_BATCH_MODE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --subspace_r $SUBSPACE_R \
    --subspace_T $SUBSPACE_T \
    --report_to none \
    --log_file $OUT_FILE \
    --no_train False
