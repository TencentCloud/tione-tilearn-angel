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
YAML_PATH=full_sft_baseline.yaml
LOG_PATH=./log/baseline.log

#export ENABLE_FP8=1
#export LD_LIBRARY_PATH=//mnt/data/boyyang/cublas-x86_64-centos7-cuda12.3_r545/lib64/:$LD_LIBRARY_PATH

bash ../../utils/run_single_node.sh $CODE_PATH $YAML_PATH $LOG_PATH
exit $?
