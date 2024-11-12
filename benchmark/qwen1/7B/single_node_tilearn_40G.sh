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

export TILEARN_DEBUG=1
export TILEARN_HYBRID_TP_SIZE=2
export TILEARN_HYBRID_PP_SIZE=2
export TILEARN_HYBRID_OFFLOAD=0
export TILEARN_HYBRID_ZERO_STAGE=1
#export TIACC_FASTER_ROPE=1
export TILEARN_HYBRID_FUSE_NORMALIZATION=1

#export TILEARN_HYBRID_MODE='AutoZero'
#export TILEARN_HYBRID_AUTOZERO_SHARD_PARAM=1
#export TILEARN_HYBRID_AUTOZERO_OFFLOAD_OPTIM=0
#export TILEARN_HYBRID_AUTOZERO_OFFLOAD_PARAM=0


CODE_PATH=../../utils/train_tilearn.py
YAML_PATH=full_sft_tilearn_40G.yaml
LOG_PATH=./log/tilearn_40G.log

bash ../../utils/run_single_node.sh $CODE_PATH $YAML_PATH $LOG_PATH
exit $?
