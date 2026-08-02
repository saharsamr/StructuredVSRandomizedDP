#!/bin/bash
#
# Dev-split canary experiment: does the data-driven subspace respond to single records?
#
# Runs the identical DPTrack configuration three times, changing only where the subspace
# gradient comes from, and prints one comparison table at the end.
#
#   shared        subspace fit to the TRAIN batch      BASELINE -- the dev canaries never
#                                                      touch the subspace, so whatever gap
#                                                      it shows is geometry, not leakage.
#                                                      Subtracted from the other arms.
#   skip          subspace fit to BARE dev gradients   eps = infinity. Expect a gap.
#   private-skip  subspace fit to CLIPPED+NOISED dev   finite eps. Expect no gap.
#
# Half the dev examples get a flipped label. A flipped label makes that example's gradient
# large and unusual; if the subspace chases it, the flipped group's gradient energy is
# captured more than the clean group's. That difference is the leak, and it is measured
# directly rather than inferred from model outputs -- the subspace carries directions, not
# labels, so no weight update ever pushes the model toward a dev example's label and asking
# the final model to predict it would measure a channel that does not exist.
#
# Do not drop the shared arm to save time. A mislabeled example's gradient is unusual, and
# a subspace fit to any natural batch captures unusual directions less well -- so there is
# a nonzero gap even when the subspace provably never saw the canaries. The shared arm
# measures that offset; the reported leakage is gap(arm) - gap(shared). Without it the
# analyzer will refuse to interpret the numbers, which is the correct behaviour.
#
# WARNING: the dev labels are deliberately corrupted in these runs. Eval and test metrics
# from them are meaningless. Use the real dptrack scripts for utility numbers.
#
# Usage:  bash experiments/canary_leak_exp.sh                  # from roberta_finetuning/
#         TASK=RTE NUM_CANARIES=64 bash experiments/canary_leak_exp.sh
#         MODES="skip private-skip" bash experiments/canary_leak_exp.sh
#         ANALYZE_ONLY=true bash experiments/canary_leak_exp.sh   # re-print from old logs

set -u

TASK=${TASK:-SST-2}
K=${K:-512}
SEED=${SEED:-42}
STEP=${STEP:-1000}
SUBSPACE_R=${SUBSPACE_R:-16}
SUBSPACE_T=${SUBSPACE_T:-100}
PRIVACY_EPS=${PRIVACY_EPS:-6.0}

# How many dev examples go under test. Every canary costs one extra single-example backward
# per subspace update, so the total is NUM_CANARIES * (STEP/SUBSPACE_T + 1) backwards --
# 128 * 11 = 1408 at the defaults, a few minutes on RoBERTa-Large. Raise it for more
# statistical power, lower it if the probe is dominating your runtime. -1 means the whole
# dev split, which is the right choice only when that split is small.
NUM_CANARIES=${NUM_CANARIES:--1}
CANARY_SEED=${CANARY_SEED:-0}

# The three arms. Keep 'shared' in the list: it is the baseline that gets subtracted, and a
# result without it is not interpretable.
MODES=${MODES:-"shared skip private-skip"}

ANALYZE_ONLY=${ANALYZE_ONLY:-false}
USE_WANDB=${USE_WANDB:-true}

LOG_DIR=${LOG_DIR:-canary_logs/${TASK}-k${K}-seed${SEED}-r${SUBSPACE_R}-T${SUBSPACE_T}-eps${PRIVACY_EPS}-cseed${CANARY_SEED}}
mkdir -p "$LOG_DIR"

echo "=========================================================================="
echo "Canary leak experiment"
echo "  task            $TASK  (K=$K, seed=$SEED)"
echo "  steps           $STEP,  subspace r=$SUBSPACE_R, T=$SUBSPACE_T"
echo "  epsilon         $PRIVACY_EPS"
echo "  canaries        $NUM_CANARIES (half flipped), canary seed $CANARY_SEED"
echo "  arms            $MODES"
echo "  logs            $LOG_DIR"
echo "=========================================================================="

LOGS=""
for MODE in $MODES; do
    LOG="$LOG_DIR/$MODE.jsonl"
    LOGS="$LOGS $LOG"

    if [ "$ANALYZE_ONLY" = "true" ]; then
        continue
    fi

    echo
    echo ">>> arm: $MODE  ->  $LOG"
    # Everything except ORACLE_BATCH_MODE is fixed, so the arms differ in exactly one
    # thing. The extra flags land at the end of the python command line and override the
    # dptrack script's own, which is why the hyperparameters do not have to be repeated
    # here -- they come from roberta_finetuning_dptrack.sh, the same file the real runs use.
    TASK=$TASK K=$K SEED=$SEED STEP=$STEP \
    SUBSPACE_R=$SUBSPACE_R SUBSPACE_T=$SUBSPACE_T \
    PRIVACY_EPS=$PRIVACY_EPS ORACLE_BATCH_MODE=$MODE \
    USE_WANDB=$USE_WANDB \
    EXTRA_TAG=canary- \
        bash roberta_finetuning_dptrack.sh \
        --canary_probe True \
        --num_canaries $NUM_CANARIES \
        --canary_seed $CANARY_SEED \
        --canary_log "$LOG" \
        --evaluate_during_training False

    if [ ! -s "$LOG" ]; then
        echo "!!! arm $MODE produced no canary log at $LOG -- check the training output above" >&2
    fi
done

echo
echo "=========================================================================="
echo "Results"
echo "=========================================================================="
EXISTING=""
for LOG in $LOGS; do
    [ -s "$LOG" ] && EXISTING="$EXISTING $LOG"
done

if [ -z "$EXISTING" ]; then
    echo "no canary logs to analyze" >&2
    exit 1
fi

python analyze_canaries.py $EXISTING | tee "$LOG_DIR/summary.txt"
echo
echo "saved: $LOG_DIR/summary.txt"
