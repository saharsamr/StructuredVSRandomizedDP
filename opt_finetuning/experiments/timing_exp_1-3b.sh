#!/bin/bash

TASK=${TASK:-SQuAD}
DP_CLIP_THRESHOLD=${DP_CLIP_THRESHOLD:-1.0}
SEED=${SEED:-0}

# Determine number of GPUs to set GRAD_ACCUM_STEPS and PER_DEVICE_TRAIN_BS to get total batch size of 64
gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | sed 's/,$//' | awk -F',' '{print NF}')
echo "CUDA_VISIBLE_DEVICES GPU Count:$gpu_count"

if [ "$gpu_count" -eq 1 ]; then
    # Include gradient accumulation in memory estimate
    a=3
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

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_STR="${MODEL//\//-}"

mkdir -p output_logs_opt_timing_exp
mkdir -p output_ignore

# Get results for dpgrape (r=16), dp-adam, non-dp adam, dpzero,
# using same batch size as final results experiment
TAG=$MODEL_STR-timing-squad-dpgrape-bs$BS-subspace_r16
OUT_FILE="output_logs_opt_timing_exp/${TAG}.txt"
python opt_run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --log_file $OUT_FILE \
    --output_dir output_ignore \
    --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer regular \
    --load_bfloat16 False \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate $LR --lr_scheduler_type "constant" \
    --no_eval True \
    --save_model False \
    --save_strategy no \
    --save_total_limit 0 \
    --dpzero False \
    --dpgrape True \
    --dp_clip_threshold $DP_CLIP_THRESHOLD \
    --dp_epsilon $PRIVACY_EPS \
    --dp_delta $PRIVACY_DELTA \
    --subspace_r 16 \
    --subspace_T 100 \
    --report_to none \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"

TAG=$MODEL_STR-timing-squad-dpadam-bs$BS
OUT_FILE="output_logs_opt_timing_exp/${TAG}.txt"
python opt_run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --log_file $OUT_FILE \
    --output_dir output_ignore \
    --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer regular \
    --load_bfloat16 False \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate $LR --lr_scheduler_type "constant" \
    --no_eval True \
    --save_model False \
    --save_strategy no \
    --save_total_limit 0 \
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

TAG=$MODEL_STR-timing-squad-adam-bs$BS
OUT_FILE="output_logs_opt_timing_exp/${TAG}.txt"
python opt_run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --log_file $OUT_FILE \
    --output_dir output_ignore \
    --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer regular \
    --load_bfloat16 False \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate $LR --lr_scheduler_type "constant" \
    --no_eval True \
    --save_model False \
    --save_strategy no \
    --save_total_limit 0 \
    --dpadam False \
    --dpzero False \
    --dpgrape False \
    --report_to none \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"

TAG=$MODEL_STR-timing-squad-dpzero-bs$BS
OUT_FILE="output_logs_opt_timing_exp/${TAG}.txt"
python opt_run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --log_file $OUT_FILE \
    --output_dir output_ignore \
    --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer regular \
    --load_bfloat16 False \
    --per_device_train_batch_size 8 \
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
