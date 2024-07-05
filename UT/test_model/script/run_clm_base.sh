set -x
. ~/.bashrc

MODEL_PATH=${MODEL_PATH:-"not set"}
HOST_NUM=${HOST_NUM:-1}
INDEX=${INDEX:-0}
CHIEF_IP=${CHIEF_IP:-"127.0.0.1"}
HOST_GPU_NUM=${HOST_GPU_NUM:-1}
#HOST_GPU_NUM=${HOST_GPU_NUM:-8}

BS=${BS:-1}
MAX_STEP=${MAX_STEP:-10}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-"--gradient_checkpointing"}
DS_CONFIG=${DS_CONFIG:-""}   #"--deepspeed=../ds_config/ds_config_zero2.json"

TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-""}  #"--trust_remote_code"
CONFIG_NAME=${CONFIG_NAME:-"config_tiny.json"}
if [ "$TRUST_REMOTE_CODE" = "--trust_remote_code" ]; then
    CONFIG_NAME=""
fi

export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
#export NCCL_IB_QPS_PER_CONNECTION=2

CMD="python -m torch.distributed.launch --use-env --nnodes=$HOST_NUM \
    --node_rank=$INDEX \
    --nproc_per_node $HOST_GPU_NUM \
    --master_addr $CHIEF_IP \
    --master_port 19198 \
    ../run_clm.py \
    --config_name=$MODEL_PATH/${CONFIG_NAME} \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --tokenizer_name=$MODEL_PATH \
    --train_file ../wiki_0000.json \
    --per_device_train_batch_size $BS \
    --do_train \
    --output_dir ../ckpt-llm/ \
    --overwrite_output_dir \
    --max_steps=${MAX_STEP} \
    --block_size=${SEQ_LENGTH} \
    --lr_scheduler_type constant_with_warmup \
    --bf16 \
    ${TRUST_REMOTE_CODE} \
    ${DS_CONFIG} \
    ${GRADIENT_CHECKPOINTING} \
    --lr_scheduler_type constant_with_warmup
    "
eval ${CMD}
    #--config_yyname=$MODEL_PATH/config.json \
    #--deepspeed=../ds_config/ds_config_zero3.json \
    #--fsdp "full_shard auto_wrap" \
    #--fsdp_transformer_layer_cls_to_wrap 'TELlamaDecoderLayer' \
    #--fsdp_config fsdp_config_compile.json \
    #--torch_compile \
    #--deepspeed=ds_config_zero3.json \

    #--deepspeed=ds_config_zero3.json \
    #--gradient_checkpointing \
    #--fsdp "full_shard auto_wrap" \
    #--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    #--fsdp_config fsdp_config_compile.json \
    # --torch_compile \
    
    #--deepspeed=ds_config_zero3.json

    
