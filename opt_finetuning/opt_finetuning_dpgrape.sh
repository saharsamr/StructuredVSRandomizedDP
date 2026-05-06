
MODEL=${MODEL:-facebook/opt-1.3b}
#MODEL=${MODEL:-facebook/opt-2.7b}
#MODEL=${MODEL:-facebook/opt-6.7b}

TASK=${TASK:-SQuAD}
BS=${BS:-8}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
LR=${LR:-1e-4}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-2000}
EVAL_STEPS=${EVAL_STEPS:-100}

DP_CLIP_THRESHOLD=${DP_CLIP_THRESHOLD:-1.0}
PRIVACY_EPS=${PRIVACY_EPS:-6.0}
PRIVACY_DELTA=${PRIVACY_DELTA:-1e-5}

SUBSPACE_R=${SUBSPACE_R:-16}
SUBSPACE_T=${SUBSPACE_T:-100}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi
TAG=dpgrape-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-$DP_EPS-$DP_CLIP

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
    SST2)
        TASK_ARGS="--train_as_classification True"
        ;;
    BoolQ)
        TASK_ARGS="--train_as_classification True"
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

MODEL_STR="${MODEL//\//-}"
GR_TAG=dpgrape-$MODEL_STR-$TASK-seed$SEED-train-$TRAIN-bs$BS-lr$LR-dpeps$PRIVACY_EPS-dpdelta$PRIVACY_DELTA-dpC$DP_CLIP_THRESHOLD-totalsteps$STEP-subspace_r$SUBSPACE_R-subspace_T$SUBSPACE_T
OUT_FILE="output/${GR_TAG}.txt"
OUT_DIR="output"

mkdir -p output

python opt_run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir $OUT_DIR \
    --log_file $OUT_FILE \
    --tag $GR_TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer regular \
    --load_bfloat16 False \
    --per_device_train_batch_size $BS \
    --learning_rate $LR --lr_scheduler_type "constant" \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_model False \
    --save_strategy no \
    --save_total_limit 0 \
    --gradient_accumulation_steps 1 \
    --dpgrape True \
    --dp_clip_threshold $DP_CLIP_THRESHOLD \
    --dp_epsilon $PRIVACY_EPS \
    --dp_delta $PRIVACY_DELTA \
    --subspace_r $SUBSPACE_R \
    --subspace_T $SUBSPACE_T \
    --report_to none \
    $EXTRA_ARGS \
    $TASK_ARGS
