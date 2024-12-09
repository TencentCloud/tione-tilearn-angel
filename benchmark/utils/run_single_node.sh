#!/bin/bash

### CODE ARGS
CODE_PATH=${1:-"NotSet"}
YAML_PATH=${2:-"NotSet"}
LOG_PATH=${3:-""}

### Torch DPP ARGS
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-23456}
NNODES=${NODE_NUM:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${GPUS_NUM_PER_NODE:-$(nvidia-smi -L | wc -l)}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

### Demo Args
# llama factory model random initialization
#export LF_MODEL_RANDOM_INIT=1

#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
CMD="torchrun \
    $DISTRIBUTED_ARGS \
    $CODE_PATH \
    $YAML_PATH
    "    

### RUN Task CMD
if [ ! -d "./log/" ];then
  mkdir log
fi
echo ${CMD}
eval ${CMD} 2>&1 | tee $LOG_PATH

errorCode=${PIPESTATUS[0]}
#errorCode=$?
if [ $errorCode -ne 0 ]; then
  echo "Training process has an error! Stopping evaluation process. errorCode: ${errorCode}"
  # We exit the all script with the same error, if you don't want to
  # exit it and continue, just delete this line.
  exit $errorCode
fi

