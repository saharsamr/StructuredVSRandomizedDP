
MODEL=${MODEL:-facebook/opt-1.3b}
#MODEL=${MODEL:-facebook/opt-2.7b}
#MODEL=${MODEL:-facebook/opt-6.7b}

TASK=${TASK:-SQuAD}
BS=${BS:-8}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-4000}
DP_EPS=${DP_EPS:-6.0}
DP_CLIP=${DP_CLIP:-10.0}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
TAG=dpzero-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-$DP_EPS-$DP_CLIP

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

echo $TAG
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

mkdir -p output

MODEL_STR="${MODEL//\//-}"
GR_TAG=dpzero-$MODEL_STR-$TASK-seed$SEED-train-$TRAIN-bs$BS-lr$LR-dpeps$PRIVACY_EPS-dpdelta$PRIVACY_DELTA-dpC$DP_CLIP_THRESHOLD-totalsteps$STEP-subspace_r$SUBSPACE_R-subspace_T$SUBSPACE_T
OUT_FILE="output/${GR_TAG}.txt"
OUT_DIR="output"

mkdir -p output

python opt_run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir $OUT_DIR \
    --log_file $OUT_FILE \
    --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer zo --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --dpzero True \
    --dpzero_clip_threshold $DP_CLIP \
    --dp_epsilon $DP_EPS \
    --dp_delta 1e-5 \
    --log_file $OUT_FILE \
    --report_to none \
    $EXTRA_ARGS \
    $TASK_ARGS 