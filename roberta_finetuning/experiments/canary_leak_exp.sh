#!/bin/bash
#
# Dev-split canary experiment: does the data-driven subspace respond to single records?
#
# Runs the identical DPTrack configuration three times, changing only where the subspace
# gradient comes from, and prints one comparison table at the end.
#
#   skip          subspace fit to BARE dev gradients   eps = infinity. Expect a gap.
#   private-skip  subspace fit to CLIPPED+NOISED dev   finite eps. Expect no gap.
#   shared        subspace fit to the TRAIN batch      optional. Only needed for the
#                                                      secondary flipped-vs-clean number,
#                                                      as its geometry baseline.
#
# THE TEST is membership. Each subspace update draws one batch (64 of the 1024 dev
# examples), so only those 64 fed the subspace being scored. The probe compares the
# canaries that fed this update against the ones that have not been drawn yet. Both groups
# are the same records with the same flipped labels, split only by the sampler, so they are
# identical in distribution -- nothing has to be subtracted off, and the question it asks
# is the privacy question directly: was this record in the data that built this subspace?
#
# It is measured on the subspace, not on model outputs, because the subspace carries
# directions and not labels: no weight update ever pushes the model toward a dev example's
# label, so asking the final model to predict it would measure a channel that does not
# exist.
#
# Half the dev labels are flipped to make those gradients large and unusual, which gives
# the subspace more to latch onto. Flipped vs clean is reported too, but it is secondary --
# it needs the shared arm as a baseline, because a mislabeled gradient is captured less
# well by any natural subspace whether or not that subspace ever saw it.
#
# WARNING: the dev labels are deliberately corrupted in these runs. Eval and test metrics
# from them are meaningless. Use the real dptrack scripts for utility numbers.
#
# Usage:  bash experiments/canary_leak_exp.sh                  # from roberta_finetuning/
#         METHOD=subtrack bash experiments/canary_leak_exp.sh   # DPTrack instead of GaLore
#         TASK=RTE NUM_CANARIES=256 bash experiments/canary_leak_exp.sh
#         MODES="skip private-skip" bash experiments/canary_leak_exp.sh
#         ANALYZE_ONLY=true bash experiments/canary_leak_exp.sh   # re-print from old logs

set -u

TASK=${TASK:-SST-2}
K=${K:-512}
SEED=${SEED:-42}
STEP=${STEP:-1000}
SUBSPACE_R=${SUBSPACE_R:-16}
PRIVACY_EPS=${PRIVACY_EPS:-6.0}

# Which subspace trackers to audit. Space-separated; both by default, so one invocation
# produces the whole experiment.
#
#   galore    refits the SVD from the current batch every update
#   subtrack  takes one rank-1 geodesic step, so the batch only nudges the subspace
#
# Expect galore to leak much harder, and expect to need more updates to resolve subtrack.
# In a scaled-down simulation of this exact setup (same 1-in-16 batch dilution) galore hit
# p = 5e-10 in 11 updates while subtrack was still at p = 0.17; subtrack reached p < 0.05
# at 40 updates and p = 3e-5 at 80.
#
# That difference is itself a result worth reporting: how much a data-driven subspace leaks
# depends on how hard it refits, not just on whether it is data-driven.
#
# METHOD=galore alone is the fastest path to a first answer.
METHODS=${METHODS:-${METHOD:-"galore subtrack"}}

# The test statistic is one number per subspace update, so the number of updates IS the
# sample size: STEP/SUBSPACE_T + 1. At the dptrack default of 100 a 1000-step run gives 11,
# which is enough for galore and not for subtrack. 25 gives 41.
#
# Note this audits a T=25 configuration. To audit your production T instead, keep T=100 and
# raise STEP -- the runtime is similar and the method being measured is the one you ship.
SUBSPACE_T=${SUBSPACE_T:-25}

# NOT the dptrack script's default of 10. At 10 the first run of this experiment logged
# mean_rotation_deg = 42-48, against the ~10 deg/update that roberta_finetuning_dptrack.sh
# documents as healthy -- the tracker was thrashing, throwing the subspace roughly 45 deg
# away every update instead of accumulating. Rotation is proportional to ST_STEP_SIZE, so
# 10 -> 2 should land near 9-10 deg. CHECK mean_rotation_deg in the training log and
# rescale linearly if it is off; a leak measured on a thrashing subspace says nothing about
# a properly tuned one.
ST_STEP_SIZE=${ST_STEP_SIZE:-2}

# How many dev examples go under test. Every canary costs one extra single-example backward
# per subspace update: NUM_CANARIES * (STEP/SUBSPACE_T + 1) in total. At -1 on SST-2 that is
# 1024 * 41 = 42k backwards, which the first run clocked at ~35 s per update, so ~25 min of
# probe time per arm on top of training. Halve it if that is too slow -- the per-step groups
# shrink proportionally, but the test statistic is per-update, so power drops slowly.
NUM_CANARIES=${NUM_CANARIES:--1}
CANARY_SEED=${CANARY_SEED:-0}

# Fraction of dev fenced off from every subspace batch: the permanent control group.
# Needed because the alternative control -- 'not drawn yet' -- runs out after
# |pool|/batch_size updates and the comparison then has nothing to compare against.
CANARY_HOLDOUT_FRAC=${CANARY_HOLDOUT_FRAC:-0.25}

# skip and private-skip are enough for the membership test. 'shared' only buys the baseline
# for the secondary flipped-vs-clean number; drop it with MODES="skip private-skip" to save
# a third of the runtime.
MODES=${MODES:-"shared skip private-skip"}

ANALYZE_ONLY=${ANALYZE_ONLY:-false}
USE_WANDB=${USE_WANDB:-true}

for M in $METHODS; do
    case $M in
        galore|subtrack) ;;
        *) echo "METHODS entries must be 'galore' or 'subtrack', got '$M'" >&2; exit 1 ;;
    esac
done

NUM_RUNS=0
for M in $METHODS; do for MODE in $MODES; do NUM_RUNS=$((NUM_RUNS + 1)); done; done
UPDATES=$((STEP / SUBSPACE_T + 1))

RUN_ROOT=${RUN_ROOT:-canary_logs/${TASK}-k${K}-seed${SEED}-r${SUBSPACE_R}-T${SUBSPACE_T}-eps${PRIVACY_EPS}-cseed${CANARY_SEED}}

echo "=========================================================================="
echo "Canary leak experiment"
echo "  methods         $METHODS"
echo "  arms            $MODES"
echo "  -> $NUM_RUNS training runs"
echo "  task            $TASK  (K=$K, seed=$SEED)"
echo "  steps           $STEP,  subspace r=$SUBSPACE_R, T=$SUBSPACE_T"
echo "                  -> $UPDATES subspace updates per run = the sample size"
echo "  epsilon         $PRIVACY_EPS,  st_step_size $ST_STEP_SIZE (subtrack only)"
echo "  canaries        $NUM_CANARIES (half flipped), holdout $CANARY_HOLDOUT_FRAC, seed $CANARY_SEED"
echo "  logs            $RUN_ROOT"
echo "--------------------------------------------------------------------------"
echo "  This is a long job. To cut it down:"
echo "    METHODS=galore              first answer fastest; galore leaks hardest"
echo "    MODES=\"skip private-skip\"   drops the arm only the secondary test needs"
echo "    NUM_CANARIES=512           halves the probe cost per update"
echo "=========================================================================="

ALL_LOGS=""
for METHOD in $METHODS; do
    case $METHOD in
        galore)   METHOD_SCRIPT=roberta_finetuning_dpgalore.sh ;;
        subtrack) METHOD_SCRIPT=roberta_finetuning_dptrack.sh  ;;
    esac
    LOG_DIR="$RUN_ROOT/$METHOD"
    mkdir -p "$LOG_DIR"

    for MODE in $MODES; do
        LOG="$LOG_DIR/$MODE.jsonl"
        ALL_LOGS="$ALL_LOGS $LOG"

        if [ "$ANALYZE_ONLY" = "true" ]; then
            continue
        fi

        echo
        echo ">>> $METHOD / $MODE  ->  $LOG"
        # Everything except METHOD and ORACLE_BATCH_MODE is fixed, so the arms differ in
        # exactly one thing. The extra flags land at the end of the python command line and
        # override the method script's own, which is why the hyperparameters do not have to
        # be repeated here -- they come from the same file the real runs use.
        TASK=$TASK K=$K SEED=$SEED STEP=$STEP \
        SUBSPACE_R=$SUBSPACE_R SUBSPACE_T=$SUBSPACE_T \
        PRIVACY_EPS=$PRIVACY_EPS ORACLE_BATCH_MODE=$MODE \
        ST_STEP_SIZE=$ST_STEP_SIZE \
        USE_WANDB=$USE_WANDB \
        EXTRA_TAG=canary- \
            bash "$METHOD_SCRIPT" \
            --canary_probe True \
            --num_canaries $NUM_CANARIES \
            --canary_seed $CANARY_SEED \
            --canary_holdout_frac $CANARY_HOLDOUT_FRAC \
            --canary_log "$LOG" \
            --evaluate_during_training False

        if [ ! -s "$LOG" ]; then
            echo "!!! $METHOD/$MODE produced no canary log at $LOG -- check the output above" >&2
        fi
    done
done
LOGS="$ALL_LOGS"

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

python analyze_canaries.py $EXISTING | tee "$RUN_ROOT/summary.txt"
echo
echo "saved: $RUN_ROOT/summary.txt"
