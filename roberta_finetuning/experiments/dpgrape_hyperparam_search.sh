#!/bin/bash

TASK=${TASK:-"$1"}  # Options are SST-2, sst-5, SNLI, MNLI, trec, RTE (all case-sensitive)
K=${K:-512}
SEED=${SEED:-100}
BS=${BS:-64}
PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-64}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
EPS=${EPS:-1e-3}
WD=${WD:-0}
EVAL_STEP=${EVAL_STEP:-100}
MODEL=${MODEL:-roberta-large}

# Logs and results also stream to Weights & Biases; set USE_WANDB=false to turn that off (see README).
USE_WANDB=${USE_WANDB:-true}

PRIVACY_EPS=${PRIVACY_EPS:-6.0}
PRIVACY_DELTA=${PRIVACY_DELTA:-1e-5}
if [ "$TASK" = "SNLI" ]; then
    LOGITS=3
elif [ "$TASK" = "MNLI" ]; then
    LOGITS=3
elif [ "$TASK" = "trec" ]; then
    LOGITS=6
elif [ "$TASK" = "sst-5" ]; then
    LOGITS=5
else
    LOGITS=2
fi

lrs=(0.0001 0.0005 0.001 0.005)
Cs=(0.1 1.0 10.0 100.0)
rs=(4 16 64)
Fs=(10 100 1000)
total_steps=(50 200 400 700 1000)

mkdir -p output_logs_roberta_hyperparams

for lr in "${lrs[@]}";
do
    for C in "${Cs[@]}";
    do
        for r in "${rs[@]}";
        do
            for F in "${Fs[@]}";
            do
                for steps in "${total_steps[@]}";
                do

                    echo $lr $r $F $steps $C
                    GR_TAG=dpgrape-seed$SEED-bs$BS-lr$lr-eps$EPS-wd$WD-step$steps-evalstep$EVAL_STEP
                    EXTRA_TAG=${EXTRA_TAG:-ft-}
                    TAG=${TAG:-k${K}-${MODEL}-dpgrape-${EXTRA_TAG}-r${r}-F${F}-C${C}}
                    OUT_FILE="output_logs_roberta_hyperparams/${GR_TAG}.txt"
                    echo "Grid search tag: $GR_TAG"
                    echo "Tag: $TAG"

                    TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$steps TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
                        bash roberta_finetuning_fewshot.sh \
                        --per_device_train_batch_size $PER_DEVICE_TRAIN_BS \
                        --learning_rate $lr \
                        --eval_steps $EVAL_STEP \
                        --weight_decay $WD \
                        --lr_scheduler_type "constant" \
                        --optimizer "adam" \
                        --dp_clip_threshold $C \
                        --dp_epsilon $PRIVACY_EPS \
                        --dp_delta $PRIVACY_DELTA \
                        --dp_clip_strategy flat \
                        --dpgrape True \
                        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
                        --subspace_r $r \
                        --subspace_T $F \
                        --no_train False \
                        --no_predict True \
                        --use_wandb $USE_WANDB \
                        --report_to none \
                        --log_file $OUT_FILE 
                done
            done
        done
    done
done
