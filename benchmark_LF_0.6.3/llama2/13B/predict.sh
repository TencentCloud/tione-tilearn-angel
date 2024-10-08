#!/bin/bash

MODEL_NAME=Llama-2-13b-chat-hf
TEMPLATE=llama2
SEQ_LENGTH=1024
BASE_PATH=../../
MODEL_PATH=$BASE_PATH/ckpt/$MODEL_NAME/sft/
#MODEL_PATH=/mnt/cfs/tilearn/pretrain_models/$MODEL_NAME
DATA_PATH=$BASE_PATH/data
RESULT_PATH=$BASE_PATH/result/$MODEL_NAME/predict

#CUDA_VISIBLE_DEVICES=0 python3 $BASE_PATH/utils/train_bash.py \
python3 $BASE_PATH/utils/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path $MODEL_PATH \
    --dataset alpaca_en \
    --dataset_dir $DATA_PATH \
    --template $TEMPLATE \
    --finetuning_type full \
    --output_dir $RESULT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len $SEQ_LENGTH \
    --preprocessing_num_workers 16 \
    --max_samples 20 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate

#    --max_samples 20 \
