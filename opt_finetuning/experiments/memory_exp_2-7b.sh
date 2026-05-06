#!/bin/bash

TASK=${TASK:-SQuAD}
DP_CLIP_THRESHOLD=${DP_CLIP_THRESHOLD:-1.0}
SEED=${SEED:-0}

# Determine number of GPUs to set GRAD_ACCUM_STEPS and PER_DEVICE_TRAIN_BS to get total batch size of 64
gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | sed 's/,$//' | awk -F',' '{print NF}')
echo "CUDA_VISIBLE_DEVICES GPU Count:$gpu_count"

if [ "$gpu_count" -eq 1 ]; then
    # Include gradient accumulation in memory estimate
    GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
else
    echo "Please set CUDA_VISIBLE_DEVICES to include 1 GPU only"
    exit 1
fi

LR=${LR:-1e-4}
EPS=${EPS:-1e-3}
WD=${WD:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEP:-30}
EVAL_STEP=${EVAL_STEP:-10000}
PRIVACY_EPS=${PRIVACY_EPS:-6.0}
PRIVACY_DELTA=${PRIVACY_DELTA:-1e-5}

SUBSPACE_T=${SUBSPACE_T:-100}

MODEL=${MODEL:-facebook/opt-2.7b}
MODEL_STR="${MODEL//\//-}"

mkdir -p output_logs_opt_memory_exp
mkdir -p output_ignore

#BATCH_SIZES=(1 2 3 4 5 6 7 8 10 12 14 16 20 24)
BATCH_SIZES=(7 8 10 12 14 16 20 24)
RS=(4 16 32 64 256)

for BS in "${BATCH_SIZES[@]}";
do
    # Get results for dpgrape r=4,16,64,256, then dpadam and non-dp adam
    for SUBSPACE_R in "${RS[@]}";
    do
        TAG=$MODEL_STR-memory-$TASK-dpgrape-bs$BS-subspace_r$SUBSPACE_R
        OUT_FILE="output_logs_opt_memory_exp/${TAG}.txt"
        python opt_run.py \
            --model_name $MODEL \
            --task_name $TASK \
            --log_file $OUT_FILE \
            --output_dir output_ignore \
            --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
            --max_steps $STEPS \
            --trainer regular \
            --load_bfloat16 False \
            --per_device_train_batch_size $BS \
            --learning_rate $LR --lr_scheduler_type "constant" \
            --no_eval True \
            --save_model False \
            --save_strategy no \
            --save_total_limit 0 \
            --gradient_accumulation_steps 1 \
            --dpzero False \
        --dpgrape True \
            --dp_clip_threshold $DP_CLIP_THRESHOLD \
            --dp_epsilon $PRIVACY_EPS \
            --dp_delta $PRIVACY_DELTA \
            --subspace_r $SUBSPACE_R \
            --subspace_T $SUBSPACE_T \
            --report_to none \
            $EXTRA_ARGS \
            $TASK_ARGS \
            "$@"
    done

    TAG=$MODEL_STR-memory-$TASK-dpadam-bs$BS
    OUT_FILE="output_logs_opt_memory_exp/${TAG}.txt"
    python opt_run.py \
        --model_name $MODEL \
        --task_name $TASK \
        --log_file $OUT_FILE \
        --output_dir output_ignore \
        --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
        --max_steps $STEPS \
        --trainer regular \
        --load_bfloat16 False \
        --per_device_train_batch_size $BS \
        --learning_rate $LR --lr_scheduler_type "constant" \
        --no_eval True \
        --save_model False \
        --save_strategy no \
        --save_total_limit 0 \
        --gradient_accumulation_steps 1 \
        --dpadam True \
        --dpzero False \
        --dpgrape False \
        --dp_clip_threshold $DP_CLIP_THRESHOLD \
        --dp_epsilon $PRIVACY_EPS \
        --dp_delta $PRIVACY_DELTA \
        --report_to none \
        $EXTRA_ARGS \
        $TASK_ARGS \
        "$@"
    
    TAG=$MODEL_STR-memory-$TASK-adam-bs$BS
    OUT_FILE="output_logs_opt_memory_exp/${TAG}.txt"
    python opt_run.py \
        --model_name $MODEL \
        --task_name $TASK \
        --log_file $OUT_FILE \
        --output_dir output_ignore \
        --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
        --max_steps $STEPS \
        --trainer regular \
        --load_bfloat16 False \
        --per_device_train_batch_size $BS \
        --learning_rate $LR --lr_scheduler_type "constant" \
        --no_eval True \
        --save_model False \
        --save_strategy no \
        --save_total_limit 0 \
        --gradient_accumulation_steps 1 \
        --dpadam False \
        --dpzero False \
        --dpgrape False \
        --report_to none \
        $EXTRA_ARGS \
        $TASK_ARGS \
        "$@"

    TAG=$MODEL_STR-memory-$TASK-dpzero-bs$BS
    OUT_FILE="output_logs_opt_memory_exp/${TAG}.txt"
    python opt_run.py \
        --model_name $MODEL \
        --task_name $TASK \
        --log_file $OUT_FILE \
        --output_dir output_ignore \
        --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
        --max_steps $STEPS \
        --trainer regular \
        --load_bfloat16 False \
        --per_device_train_batch_size $BS \
        --learning_rate $LR --lr_scheduler_type "constant" \
        --no_eval True \
        --save_model False \
        --save_strategy no \
        --save_total_limit 0 \
        --trainer zo \
        --dpadam False \
        --dpzero True \
        --dpgrape False \
        --dpzero_clip_threshold $DP_CLIP_THRESHOLD \
        --dp_epsilon $PRIVACY_EPS \
        --dp_delta $PRIVACY_DELTA \
        --report_to none \
        $EXTRA_ARGS \
        $TASK_ARGS \
        "$@"
done