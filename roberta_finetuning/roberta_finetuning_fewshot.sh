#!/bin/bash

TRAINER=${TRAINER:-"standard"}               
OPT=${OPT:-"adam"}

TASK_EXTRA=""
case $TASK in
    SST-2)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{'0':'terrible','1':'great'}"
        ;;
    sst-5)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20 --double_demo"
        ;;
    QQP)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    QNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        ;;
    MNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
    SNLI)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        TASK_EXTRA="--max_seq_len 256 --num_sample 4"
        ;;
    trec)
        TEMPLATE="*cls**mask*:*+sent_0**sep+*"
        MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
        TASK_EXTRA="--first_sent_limit 110"
        ;;
    mr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    cr)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    mpqa)
        TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
        MAPPING="{0:'terrible',1:'great'}"
        TASK_EXTRA="--first_sent_limit 110"
        ;;
    CoLA)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{'0':'incorrect','1':'correct'}"
        ;;
    subj)
        TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
        MAPPING="{0:'subjective',1:'objective'}"
        TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
        ;;
    MRPC)
        TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
        MAPPING="{'0':'No','1':'Yes'}"
        ;;
    RTE)
        TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
        MAPPING="{'not_entailment':'No','entailment':'Yes'}"
        TASK_EXTRA="--max_seq_len 256 --first_sent_limit 240"
        ;;
esac

# Weights & Biases, on unless USE_WANDB=false is in the environment. The calling scripts each
# pass --use_wandb $USE_WANDB themselves (they read the same variable, and $@ comes last
# below, so the two always agree); what is added here is the rest of the W&B settings, which
# every script would otherwise repeat. The run name defaults to $GRID_TAG, which already
# encodes the full hyperparameter setting, so runs line up with the .txt log files on disk.
#   TASK=SST-2 bash roberta_finetuning_dpgrape.sh
#   WANDB_PROJECT=... WANDB_ENTITY=... WANDB_GROUP=c-search WANDB_MODE=offline ...
#   USE_WANDB=false TASK=SST-2 bash roberta_finetuning_dpgrape.sh   # opt out
WANDB_ARGS=""
if [ "${USE_WANDB:-true}" = "true" ]; then
    WANDB_ARGS="--use_wandb True"  # for anyone invoking this script directly
    WANDB_ARGS="$WANDB_ARGS --wandb_project ${WANDB_PROJECT:-structured-vs-randomized-dp}"
    # Each flag is added only when it has a value: an empty one would swallow the next flag,
    # since ALL_ARGS_TOGETHER is expanded unquoted. Empty ones fall back to the defaults in
    # roberta_finetuning_run.py (the run name is derived from --log_file there).
    RUN_NAME=${WANDB_RUN_NAME:-${GRID_TAG:-$TAG}}
    [ -n "$RUN_NAME" ] && WANDB_ARGS="$WANDB_ARGS --wandb_run_name $RUN_NAME"
    [ -n "$WANDB_ENTITY" ] && WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"
    [ -n "$WANDB_GROUP" ] && WANDB_ARGS="$WANDB_ARGS --wandb_group $WANDB_GROUP"
    [ -n "$WANDB_TAGS" ] && WANDB_ARGS="$WANDB_ARGS --wandb_tags $WANDB_TAGS"
    [ -n "$WANDB_MODE" ] && WANDB_ARGS="$WANDB_ARGS --wandb_mode $WANDB_MODE"
    [ -n "$WANDB_DIR" ] && WANDB_ARGS="$WANDB_ARGS --wandb_dir $WANDB_DIR"
fi

ALL_ARGS_TOGETHER="
    --model_name_or_path $MODEL --few_shot_type $TYPE
    --task_name $TASK --template $TEMPLATE --mapping $MAPPING
    --data_dir data/k-shot-1k-test/$TASK/$K-$SEED
    --overwrite_output_dir --output_dir result/$TASK-$MODEL-$TYPE-$TRAINER-$TAG$GRID_TAG/$K-$SEED
    --num_k $K
    --tag $TAG
    --max_seq_length 128
    --seed $SEED
    --do_eval --do_predict --do_train
    --trainer $TRAINER
    --optimizer $OPT --max_steps $STEPS
    --logging_steps 10
    --per_device_eval_batch_size 4
    --per_device_train_batch_size 32
    --tie_emb True
    $TASK_EXTRA
    $LOAD_KERNELS
    $WANDB_ARGS
    $@
"

NUM_GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [[ $NUM_GPU > 1 ]]; then
    # Randomly set a port number
    # If you encounter "address already used" error, just run again or manually set an available port id.
    PORT_ID=$(expr $RANDOM + 1000)

    # Allow multiple threads
    export OMP_NUM_THREADS=4

    # Make sure torchrun from env is called
    $CONDA_PREFIX/bin/torchrun --nproc_per_node $NUM_GPU --master_port $PORT_ID roberta_finetuning_run.py $ALL_ARGS_TOGETHER

else
    python roberta_finetuning_run.py \
        $ALL_ARGS_TOGETHER
fi

rm -rf result/$TASK-$MODEL-$TYPE-$TRAINER-$TAG$GRID_TAG/$K-$SEED
