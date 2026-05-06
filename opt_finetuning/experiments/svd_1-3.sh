#!/bin/bash

TASK=${TASK:-SST2}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
EVAL_STEPS=${EVAL_STEPS:-100}
MODE=${MODE:-ft}
PRIVACY_DELTA=${PRIVACY_DELTA:-1e-5}
SEED=${SEED:-0}

SUBSPACE_R=${SUBSPACE_R:-64}
SUBSPACE_T=${SUBSPACE_T:-100}

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

LR=1e-4
MODEL=facebook/opt-1.3b
STEPS=130

OUT_FILE="/home/alex/dp_exp/opt_finetuning/output_opt_svd_exp_1-3/ignore.txt"

# No clipping or noise
#OUT_DIR="/home/alex/dp_exp/opt_finetuning/output_opt_svd_exp_1-3"
#    python opt_run.py \
#        --model_name $MODEL \
#        --task_name $TASK \
#        --output_dir $OUT_DIR \
#        --log_file $OUT_FILE \
#        --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 50 \
#        --max_steps $STEPS \
#        --trainer regular \
#        --load_bfloat16 False \
#        --learning_rate $LR --per_device_train_batch_size 4 --lr_scheduler_type "constant" \
#        --eval_strategy steps \
#        --eval_steps $EVAL_STEPS \
#        --save_model False \
#        --save_strategy no \
#        --save_total_limit 0 \
#        --gradient_accumulation_steps 125 \
#        --dp_delta $PRIVACY_DELTA \
#        --subspace_r $SUBSPACE_R \
#        --subspace_T $SUBSPACE_T \
#        --report_to none \
#        --svd_exp_galore True \
#        --svd_exp_save_folder_name $OUT_DIR \
#        $EXTRA_ARGS \
#        $TASK_ARGS \
#        "$@"

CS=(0.01 0.1 1.0 10.0 100.0)
NOISE_MULTIPLIERS=(0.01 0.1 0.5 1.0 2.0 5.0)

for NOISE_MULTIPLIER in ${NOISE_MULTIPLIERS[@]}; do
    # No clipping
    OUT_DIR="/home/alex/dp_exp/opt_finetuning/output_opt_svd_exp_1-3"
    python opt_run.py \
        --model_name $MODEL \
        --task_name $TASK \
        --output_dir $OUT_DIR \
        --log_file $OUT_FILE \
        --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 50 \
        --max_steps $STEPS \
        --trainer regular \
        --load_bfloat16 False \
        --learning_rate $LR --per_device_train_batch_size 2 --lr_scheduler_type "constant" \
        --eval_strategy steps \
        --eval_steps $EVAL_STEPS \
        --save_model False \
        --save_strategy no \
        --save_total_limit 0 \
        --gradient_accumulation_steps 250 \
        --dp_delta $PRIVACY_DELTA \
        --subspace_r $SUBSPACE_R \
        --subspace_T $SUBSPACE_T \
        --report_to none \
        --svd_exp_naive_dpgalore True \
        --svd_exp_no_clip True \
        --svd_exp_sigma $NOISE_MULTIPLIER \
        --svd_exp_save_folder_name $OUT_DIR \
        $EXTRA_ARGS \
        $TASK_ARGS \
        "$@"
    for C in ${CS[@]}; do
        # Clipping and noise
        OUT_DIR="/home/alex/dp_exp/opt_finetuning/output_opt_svd_exp_1-3"
        python opt_run.py \
            --model_name $MODEL \
            --task_name $TASK \
            --output_dir $OUT_DIR \
            --log_file $OUT_FILE \
            --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 50 \
            --max_steps $STEPS \
            --trainer regular \
            --load_bfloat16 False \
            --learning_rate $LR --per_device_train_batch_size 2 --lr_scheduler_type "constant" \
            --eval_strategy steps \
            --eval_steps $EVAL_STEPS \
            --save_model False \
            --save_strategy no \
            --save_total_limit 0 \
            --gradient_accumulation_steps 250 \
            --dp_delta $PRIVACY_DELTA \
            --subspace_r $SUBSPACE_R \
            --subspace_T $SUBSPACE_T \
            --report_to none \
            --svd_exp_naive_dpgalore True \
            --dp_clip_threshold $C \
            --svd_exp_sigma $NOISE_MULTIPLIER \
            --svd_exp_save_folder_name $OUT_DIR \
            $EXTRA_ARGS \
            $TASK_ARGS \
            "$@"
    done
done

