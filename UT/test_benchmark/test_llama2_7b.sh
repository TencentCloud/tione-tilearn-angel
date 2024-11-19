### Demo Args

RUN_TILEARN=${1:-0}
GPU_MEM=${GPU_MEM:-"40G"}

echo RUN_TILEARN:${RUN_TILEARN} GPU_MEM:${GPU_MEM}
#echo LF_UT_DIS_GRAD_CKPT:${LF_UT_DIS_GRAD_CKPT}

BASE_ROOT=../../benchmark/llama2/7B/
cd $BASE_ROOT

if [ $RUN_TILEARN -eq 1 ]; then

    if [ "$GPU_MEM" = "40G" ]; then

        # llama factory model random initialization
        export LF_MODEL_RANDOM_INIT=${LF_MODEL_RANDOM_INIT:-0}
        export LF_UT_TEST=${LF_UT_TEST:-1}
        export LF_UT_MODEL_PATH_PREFIX=${LF_UT_MODEL_PATH_PREFIX:-"../../models ||| /mnt/cfs/tilearn/pretrain_models"}
        export LF_UT_DSYAML_PATH_PREFIX=${LF_UT_DSYAML_PATH_PREFIX:-"ds_z3_config.json ||| ds_z3_config.json"}
        export LF_UT_DIS_GRAD_CKPT=${LF_UT_DIS_GRAD_CKPT:-1}
        export LF_UT_MAX_LENGTH=${LF_UT_MAX_LENGTH:-4096}
        export LF_UT_BS=${LF_UT_BS:-1}
        export LF_UT_GRAD_ACC=${LF_UT_GRAD_ACC:-16}
        export LF_UT_MAX_STEPS=${LF_UT_MAX_STEPS:-50}
        export LF_UT_LOG_STEPS=${LF_UT_LOG_STEPS:-10}

        export TILEARN_DEBUG=1
        export TILEARN_HYBRID_TP_SIZE=${TILEARN_HYBRID_TP_SIZE:-2}
        export TILEARN_HYBRID_PP_SIZE=${TILEARN_HYBRID_PP_SIZE:-2}
        export TILEARN_HYBRID_OFFLOAD=${TILEARN_HYBRID_OFFLOAD:-0}
        export TILEARN_HYBRID_ZERO_STAGE=${TILEARN_HYBRID_ZERO_STAGE:-1}

    elif [ $GPU_MEM = "80G" ]; then
        sleep 1s

    elif [ $GPU_MEM = "96G" ]; then
        # llama factory model random initialization
        export LF_MODEL_RANDOM_INIT=${LF_MODEL_RANDOM_INIT:-0}
        export LF_UT_TEST=${LF_UT_TEST:-1}
        export LF_UT_MODEL_PATH_PREFIX=${LF_UT_MODEL_PATH_PREFIX:-"../../models ||| /mnt/cfs/tilearn/pretrain_models"}
        export LF_UT_DSYAML_PATH_PREFIX=${LF_UT_DSYAML_PATH_PREFIX:-"ds_z1_config.json ||| ds_z1_config.json"}
        export LF_UT_DIS_GRAD_CKPT=${LF_UT_DIS_GRAD_CKPT:-1}
        export LF_UT_MAX_LENGTH=${LF_UT_MAX_LENGTH:-4096}
        export LF_UT_BS=${LF_UT_BS:-1}
        export LF_UT_GRAD_ACC=${LF_UT_GRAD_ACC:-4}
        export LF_UT_MAX_STEPS=${LF_UT_MAX_STEPS:-50}
        export LF_UT_LOG_STEPS=${LF_UT_LOG_STEPS:-10}

        export TILEARN_DEBUG=1
        export TILEARN_HYBRID_MODE='None'
        export TILEARN_HYBRID_TP_SIZE=${TILEARN_HYBRID_TP_SIZE:-1}
        export TILEARN_HYBRID_PP_SIZE=${TILEARN_HYBRID_PP_SIZE:-1}
        export TILEARN_HYBRID_OFFLOAD=${TILEARN_HYBRID_OFFLOAD:-0}
	export TILEARN_HYBRID_ZERO_STAGE=${TILEARN_HYBRID_ZERO_STAGE:-1}

    fi

    #export TIACC_FASTER_ROPE=1

    #export TILEARN_HYBRID_MODE='AutoZero'
    #export TILEARN_HYBRID_AUTOZERO_SHARD_PARAM=1
    #export TILEARN_HYBRID_AUTOZERO_OFFLOAD_OPTIM=0
    #export TILEARN_HYBRID_AUTOZERO_OFFLOAD_PARAM=0

    LOG_NAME=GpuMem${GPU_MEM}_RandomInit${LF_MODEL_RANDOM_INIT}_UT${LF_UT_TEST}_DisGradCkpt${LF_UT_DIS_GRAD_CKPT}_MaxLength${LF_UT_MAX_LENGTH}_BS${LF_UT_BS}_GradAcc${LF_UT_GRAD_ACC}
    LOG_NAME=${LOG_NAME}_TP${TILEARN_HYBRID_TP_SIZE}_PP${TILEARN_HYBRID_PP_SIZE}_OffLoad${TILEARN_HYBRID_OFFLOAD}_ZeroStage${TILEARN_HYBRID_ZERO_STAGE}
    echo LOG_NAME:tilearn_${LOG_NAME}

    CODE_PATH=../../utils/train_tilearn.py
    YAML_PATH=full_sft_tilearn.yaml
    LOG_PATH=./log/tilearn_${LOG_NAME}.log

elif [ $RUN_TILEARN -eq 0 ]; then

    if [ "$GPU_MEM" = "40G" ]; then
        # llama factory model random initialization
        export LF_MODEL_RANDOM_INIT=${LF_MODEL_RANDOM_INIT:-0}
        export LF_UT_TEST=${LF_UT_TEST:-1}
        export LF_UT_MODEL_PATH_PREFIX=${LF_UT_MODEL_PATH_PREFIX:-"../../models ||| /mnt/cfs/tilearn/pretrain_models"}
        export LF_UT_DSYAML_PATH_PREFIX=${LF_UT_DSYAML_PATH_PREFIX:-"ds_z3_config.json ||| ds_z3_config.json"}
        export LF_UT_DIS_GRAD_CKPT=${LF_UT_DIS_GRAD_CKPT:-0}
        export LF_UT_MAX_LENGTH=${LF_UT_MAX_LENGTH:-4096}
        export LF_UT_BS=${LF_UT_BS:-1}
        export LF_UT_GRAD_ACC=${LF_UT_GRAD_ACC:-4}
        export LF_UT_MAX_STEPS=${LF_UT_MAX_STEPS:-50}
        export LF_UT_LOG_STEPS=${LF_UT_LOG_STEPS:-10}

    elif [ "$GPU_MEM" = "80G" ]; then
        sleep 1s

    elif [ "$GPU_MEM" = "96G" ]; then

        # llama factory model random initialization
        export LF_MODEL_RANDOM_INIT=${LF_MODEL_RANDOM_INIT:-0}
        export LF_UT_TEST=${LF_UT_TEST:-1}
        export LF_UT_MODEL_PATH_PREFIX=${LF_UT_MODEL_PATH_PREFIX:-"../../models ||| /mnt/cfs/tilearn/pretrain_models"}
        export LF_UT_DSYAML_PATH_PREFIX=${LF_UT_DSYAML_PATH_PREFIX:-"ds_z1_config.json ||| ds_z1_config.json"}
        export LF_UT_DIS_GRAD_CKPT=${LF_UT_DIS_GRAD_CKPT:-1}
        export LF_UT_MAX_LENGTH=${LF_UT_MAX_LENGTH:-4096}
        export LF_UT_BS=${LF_UT_BS:-1}
        export LF_UT_GRAD_ACC=${LF_UT_GRAD_ACC:-4}
        export LF_UT_MAX_STEPS=${LF_UT_MAX_STEPS:-50}
        export LF_UT_LOG_STEPS=${LF_UT_LOG_STEPS:-10}

    fi
    LOG_NAME=GpuMem${GPU_MEM}_RandomInit${LF_MODEL_RANDOM_INIT}_UT${LF_UT_TEST}_DisGradCkpt${LF_UT_DIS_GRAD_CKPT}_MaxLength${LF_UT_MAX_LENGTH}_BS${LF_UT_BS}_GradAcc${LF_UT_GRAD_ACC}
    echo LOG_NAME:baseline_${LOG_NAME}
  
    CODE_PATH=../../utils/train_baseline.py
    YAML_PATH=full_sft_baseline.yaml
    LOG_PATH=./log/baseline_${LOG_NAME}.log

fi

bash ../../utils/run_single_node.sh $CODE_PATH $YAML_PATH $LOG_PATH
error_code=$?

### delete model
rm -r ../../ckpt/*

cd -
exit $error_code
