#!/bin/bash
#
# DPTrack-Oracle (bare-gradient SubTrack++ subspace) on RoBERTa-Large, few-shot.
#
# WARNING: the subspace is initialized by SVD of the BARE (unclipped, unnoised) batch
# gradient and then advanced by one rank-1 geodesic step per SUBSPACE_T steps, again on a
# bare batch gradient. The weight updates are DP; the subspace is not, so this method is
# eps = infinity as a whole. Do not put its numbers in the same column as DP-GRAPE without
# saying so. See DPTRACK_DESIGN.md section 4.
#
# Everything except the projector is identical to DP-GRAPE and DP-GaLore: same flat
# clipping, same noise, same accountant, same DPAdamW, same rank and update period.
#
# ST_STEP_SIZE is an uncalibrated starting guess, not a tuned value. The rotation per update
# is exactly ST_STEP_SIZE * Sigma radians, where Sigma is the top singular value of the
# tangent vector -- quadratic in the gradient scale, so it varies by orders of magnitude
# across layers and shrinks as training converges. Check mean_rotation_deg in the log before
# trusting any result (near 0 means the tracker is a no-op; a huge value means it is
# thrashing). Target ~90 * SUBSPACE_T / STEP degrees per update, i.e. ~10 deg at the
# defaults below; rescale linearly, since ST_STEP_SIZE and the angle are proportional.
#
# Usage:  TASK=SST-2 SEED=42 C=0.5 PRIVACY_EPS=6.0 bash roberta_finetuning_dptrack.sh

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
ST_STEP_SIZE=${ST_STEP_SIZE:-10}
ST_STEP_SIZE_SCHEDULER=${ST_STEP_SIZE_SCHEDULER:-constant}

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

GR_TAG=dptrack-$TASK-seed$SEED-bs$BS-lr$LR-dpeps$PRIVACY_EPS-dpdelta$PRIVACY_DELTA-dpC$C-totalsteps$STEP-evalstep$EVAL_STEP-subspace_r$SUBSPACE_R-subspace_T$SUBSPACE_T-batchmode$ORACLE_BATCH_MODE-ststep$ST_STEP_SIZE

mkdir -p output_logs
OUT_FILE="output_logs/${GR_TAG}.txt"

EXTRA_TAG=${EXTRA_TAG:-ft-}
TAG=${TAG:-k${K}-${MODEL}-dptrack-${EXTRA_TAG}}
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
    --dptrack True \
    --oracle_batch_mode $ORACLE_BATCH_MODE \
    --st_step_size $ST_STEP_SIZE \
    --st_step_size_scheduler $ST_STEP_SIZE_SCHEDULER \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --subspace_r $SUBSPACE_R \
    --subspace_T $SUBSPACE_T \
    --report_to none \
    --log_file $OUT_FILE \
    --no_train False
