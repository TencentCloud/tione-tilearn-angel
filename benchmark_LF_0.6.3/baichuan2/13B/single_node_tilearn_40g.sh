#!/bin/bash

### Torch DPP ARGS
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-23460}
NNODES=${NODE_NUM:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${GPUS_NUM_PER_NODE:-$(nvidia-smi -L | wc -l)}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

### Demo Args
# llama factory model random initialization
export LF_MODEL_RANDOM_INIT=1

MODEL_NAME=baichuan-inc/Baichuan2-13B-Chat
TEMPLATE=baichuan2
SEQ_LENGTH=2048
BATCH_SIZE_PER_GPU=1
GRADIENT_ACCUMULATION_STEPS=64
BASE_PATH=../../
#MODEL_PATH=$BASE_PATH/ckpt/$MODEL_NAME/sft/
MODEL_PATH=$BASE_PATH/models/$MODEL_NAME
#MODEL_PATH=/mnt/cfs/tilearn/pretrain_models/$MODEL_NAME
DATA_PATH=$BASE_PATH/data
RESULT_PATH=$BASE_PATH/ckpt/$MODEL_NAME/sft

export TILEARN_DEBUG=1
export TILEARN_HYBRID_TP_SIZE=2
export TILEARN_HYBRID_PP_SIZE=2
GLOBAL_BATCH_SZIE_PER_NODE=$(($GPUS_PER_NODE * $BATCH_SIZE_PER_GPU * $GRADIENT_ACCUMULATION_STEPS / $TILEARN_HYBRID_TP_SIZE / $TILEARN_HYBRID_PP_SIZE))

### Create Task CMD
CMD="torchrun  $DISTRIBUTED_ARGS \
    $BASE_PATH/utils/train_bash_tilearn.py \
    --deepspeed $BASE_PATH/utils/ds_config/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset advertise_gen_train \
    --dataset_dir $DATA_PATH \
    --template $TEMPLATE \
    --finetuning_type full \
    --output_dir $RESULT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len $SEQ_LENGTH \
    --packing true \
    --max_length $SEQ_LENGTH \
    --disable_gradient_checkpointing true \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 3 \
    --save_steps 500 \
    --learning_rate 5e-5 \
    --max_steps 30 \
    --ddp_timeout 180000000 \
    --bf16
    "
    #--flash_attn true \
    #--dataset alpaca_en \
    #--plot_loss \
    #--max_samples 200 \
    #--val_size 0.1 \
    #--eval_steps 100 \
    #--evaluation_strategy steps \
    #--num_train_epochs 1.0 \

### RUN Task CMD
if [ ! -d "./log/" ];then
  mkdir log
fi
echo ${CMD}
echo "TILEARN - HYBRID PARALLEL - BASH GLOBAL_BATCH_SZIE_PER_NODE:$GLOBAL_BATCH_SZIE_PER_NODE"
eval ${CMD} 2>&1 | tee ./log/tilearn_40g.log

errorCode=${PIPESTATUS[0]}
#errorCode=$?
if [ $errorCode -ne 0 ]; then
  echo "Training process has an error! Stopping evaluation process. errorCode: ${errorCode}"
  # We exit the all script with the same error, if you don't want to
  # exit it and continue, just delete this line.
  exit $errorCode
fi
