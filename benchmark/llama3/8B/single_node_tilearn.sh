#!/bin/bash

# tione.tencentcloudcr.com/qcloud-ti-platform/llm-train:24.03-gpu-py310-cu124-tilearn-llm-v1.8.0
# cd LLaMA-Factory
# pip3 install -e ".[torch,metrics]"
# transformers 4.39.3

### Torch DPP ARGS
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-23456}
NNODES=${NODE_NUM:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${GPUS_NUM_PER_NODE:-$(nvidia-smi -L | wc -l)}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

### Demo Args
# llama factory model random initialization
export LF_MODEL_RANDOM_INIT=1

export TILEARN_DEBUG=1
export TILEARN_HYBRID_TP_SIZE=1
export TILEARN_HYBRID_PP_SIZE=1
export TILEARN_HYBRID_OFFLOAD=1
#export TIACC_FASTER_ROPE=1

export TILEARN_HYBRID_MODE='AutoZero'
export TILEARN_HYBRID_AUTOZERO_SHARD_PARAM=1
export TILEARN_HYBRID_AUTOZERO_OFFLOAD_OPTIM=0
export TILEARN_HYBRID_AUTOZERO_OFFLOAD_PARAM=0


#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
CMD="torchrun \
    $DISTRIBUTED_ARGS \
    ../../utils/train_tilearn.py \
    full_sft_tilearn.yaml  
    "    

### RUN Task CMD
if [ ! -d "./log/" ];then
  mkdir log
fi
echo ${CMD}
eval ${CMD} 2>&1 | tee ./log/tilearn.log

errorCode=${PIPESTATUS[0]}
#errorCode=$?
if [ $errorCode -ne 0 ]; then
  echo "Training process has an error! Stopping evaluation process. errorCode: ${errorCode}"
  # We exit the all script with the same error, if you don't want to
  # exit it and continue, just delete this line.
  exit $errorCode
fi

