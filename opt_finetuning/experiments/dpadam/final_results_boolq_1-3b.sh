#!/bin/bash

TASK=${TASK:-BoolQ}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
EVAL_STEPS=${EVAL_STEPS:-100}

PRIVACY_DELTA=${PRIVACY_DELTA:-1e-5}

# Determine number of GPUs to set GRAD_ACCUM_STEPS and PER_DEVICE_TRAIN_BS to get total batch size of 8
gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | sed 's/,$//' | awk -F',' '{print NF}')
echo "CUDA_VISIBLE_DEVICES GPU Count:$gpu_count"

if [ "$gpu_count" -eq 1 ]; then
    PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-4}
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
elif [ "$gpu_count" -eq 2 ]; then
    PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-4}
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
elif [ "$gpu_count" -eq 4 ]; then
    PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-2}
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
elif [ "$gpu_count" -eq 8 ]; then
    PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-1}
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
else
    echo "Please set CUDA_VISIBLE_DEVICES to include 1, 2, 4, or 8 GPUs"
    exit 1
fi

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi

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

NUM_GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
BS=$((PER_DEVICE_TRAIN_BS * GRAD_ACCUM_STEPS * NUM_GPU))

LR=1e-5
DP_CLIP_THRESHOLD=5.0
MODEL=facebook/opt-1.3b
STEPS=3000

SEEDS=(0 29 83)
PRIVACY_EPSILONS=(2.0 6.0)

mkdir -p output_logs_opt_final

for SEED in "${SEEDS[@]}";
do
    for PRIVACY_EPS in "${PRIVACY_EPSILONS[@]}";
    do
        MODEL_STR="${MODEL//\//-}"
        GR_TAG=dpadam-$MODEL_STR-$TASK-seed$SEED-train-$TRAIN-bs$BS-lr$LR-dpeps$PRIVACY_EPS-dpdelta$PRIVACY_DELTA-dpC$DP_CLIP_THRESHOLD-totalsteps$STEP
        OUT_FILE="output_logs_opt_final/${GR_TAG}.txt"
        OUT_DIR="output_logs_opt_final"

        python opt_run.py \
            --model_name $MODEL \
            --task_name $TASK \
            --output_dir $OUT_DIR \
            --log_file $OUT_FILE \
            --tag $GR_TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 50 \
            --max_steps $STEPS \
            --trainer regular \
            --load_bfloat16 False \
            --learning_rate $LR --per_device_train_batch_size $PER_DEVICE_TRAIN_BS --lr_scheduler_type "constant" \
            --eval_strategy steps \
            --eval_steps $EVAL_STEPS \
            --save_model False \
            --save_strategy no \
            --save_total_limit 0 \
            --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
            --dpadam True \
            --dp_clip_threshold $DP_CLIP_THRESHOLD \
            --dp_epsilon $PRIVACY_EPS \
            --dp_delta $PRIVACY_DELTA \
            --report_to none \
            $EXTRA_ARGS \
            $TASK_ARGS \
            "$@"
    done
done