#sleep 4h

### Tilearn UT 
#export GPU_MEM=${GPU_MEM:-'96G'}
export GPU_MEM=${GPU_MEM:-'40G'}

export LF_UT_MODEL_PATH_PREFIX=${LF_UT_MODEL_PATH_PREFIX:-"../../models ||| ../../models"}
export LF_MODEL_RANDOM_INIT=${LF_MODEL_RANDOM_INIT:-1}

export ENABLE_FP8=${ENABLE_FP8:-0}
#export LD_LIBRARY_PATH=/mnt/cfs/cublas-x86_64-centos7-cuda12.3_r545/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=//mnt/data/boyyang/cublas-x86_64-centos7-cuda12.3_r545/lib64/:$LD_LIBRARY_PATH

###################################################################################
# baseline
export RUN_BASELINE=1
export RUN_TILEARN=0

datename=$(date +%Y%m%d-%H%M%S)
echo "run_all_test_autozero.sh $datename"
export BASE_LOG_PATH="./log/${datename}-log_run_all_test_autozero/"

export LF_UT_DSYAML_ORIGIN="not_set"
export LF_UT_DSYAML_TARGET="../../utils/ds_config/ds_z3_offload_config.json"
export LF_UT_DIS_GRAD_CKPT=0
export LF_UT_MAX_LENGTH=4096
export LF_UT_BS=1
export LF_UT_GRAD_ACC=4
export LF_UT_MAX_STEPS=50
export LF_UT_LOG_STEPS=10

bash run_all_test.sh 

##################################################################################
# tilearn
export RUN_BASELINE=0
export RUN_TILEARN=1

datename=$(date +%Y%m%d-%H%M%S)
echo "run_all_test_autozero.sh $datename"
export BASE_LOG_PATH="./log/${datename}-log_run_all_test_autozero/"

export TILEARN_HYBRID_MODE='AutoZero'
export TILEARN_HYBRID_AUTOZERO_SHARD_PARAM=1
export TILEARN_HYBRID_AUTOZERO_OFFLOAD_OPTIM=1
export TILEARN_HYBRID_AUTOZERO_OFFLOAD_PARAM=1
export TILEARN_HYBRID_FUSE_NORMALIZATION=1
export TIACC_FASTER_RMS=0

export TILEARN_HYBRID_TP_SIZE=1
export TILEARN_HYBRID_PP_SIZE=1
export TILEARN_HYBRID_OFFLOAD=0
export TILEARN_HYBRID_ZERO_STAGE=1
export LF_UT_DIS_GRAD_CKPT=0
export LF_UT_MAX_LENGTH=4096
export LF_UT_BS=1
export LF_UT_GRAD_ACC=4
export LF_UT_MAX_STEPS=50
export LF_UT_LOG_STEPS=10

bash run_all_test.sh
