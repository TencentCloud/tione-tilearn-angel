#!/bin/bash

# tione.tencentcloudcr.com/qcloud-ti-platform/llm-train:24.03-gpu-py310-cu124-tilearn-llm-v1.8.0
# cd LLaMA-Factory
# pip3 install -e ".[torch,metrics]"
# transformers 4.39.3

### Demo Args
# llama factory model random initialization
export LF_MODEL_RANDOM_INIT=1
#export LF_UT_TEST=1
#export LF_UT_MODEL_PATH_PREFIX="../../models ||| /mnt/cfs/tilearn/pretrain_models"

CODE_PATH=../../utils/train_baseline.py
YAML_PATH=full_sft_baseline_lora_65G.yaml
#YAML_PATH=full_sft_baseline_65G.yaml
LOG_PATH=./log/baseline_65G.log

export GPUS_NUM_PER_NODE=8
export NODE_NUM=${1:-1}   #NNODES
export RANK=${2:-0}       #NODE_RANK
export MASTER_ADDR=${3:-localhost}
export MASTER_PORT=${4:-23456}
export GPUS_NUM_PER_NODE=8

export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_ALGO=level0:NA;level1:pipeline
export HCCL_RDMA_TC=160
export HCCL_RDMA_TIMEOUT=20
export HCCL_RDMA_SL=5
export HCCL_SOCKET_IFNAME=bond1

bash ../../utils/run_single_node.sh $CODE_PATH $YAML_PATH $LOG_PATH
exit $?
