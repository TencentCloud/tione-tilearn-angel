datename=$(date +%Y%m%d-%H%M%S)
echo $datename

export GPU_MEM=${GPU_MEM:-'96G'}
export LF_UT_MAX_STEPS=${LF_UT_MAX_STEPS:-40}
export BASE_LOG_PATH=${BASE_LOG_PATH:-"./log/${datename}-log_run_one_test_autozero/"}

############################## test 
#export LF_UT_MODEL_PATH_PREFIX="../../models ||| ../../models"
#export LF_MODEL_RANDOM_INIT=1
##export ENABLE_FP8=1
###export LD_LIBRARY_PATH=/mnt/cfs/boyyang/cublas-x86_64-centos7-cuda12.3_r545/lib64/:$LD_LIBRARY_PATH
##export LD_LIBRARY_PATH=//mnt/data/boyyang/cublas-x86_64-centos7-cuda12.3_r545/lib64/:$LD_LIBRARY_PATH
##############################


CASE_INDEX=${1:-0}
RUN_TILEARN=1

#SCRIPT=test_qwen2.5_7b.sh
SCRIPT=${SCRIPT:-'test_qwen2.5_7b.sh'}

export TILEARN_HYBRID_TP_SIZE=${TILEARN_HYBRID_TP_SIZE:-1}
export TILEARN_HYBRID_PP_SIZE=${TILEARN_HYBRID_PP_SIZE:-1}
export TILEARN_HYBRID_OFFLOAD=${TILEARN_HYBRID_OFFLOAD:-0}
export TILEARN_HYBRID_ZERO_STAGE=${TILEARN_HYBRID_ZERO_STAGE:-1}

if [ $CASE_INDEX -eq 0 ]; then

    BS=("1" "1" "1" "1" "1" "1" "1" "1")
    GRAD_ACC=("4" "4" "4" "4" "4" "4" "4" "4")
    DIS_GRAD_CKPT=("0" "0" "0" "0" "0" "0" "0" "0")
    SHARD_PARAM=("0" "0" "0" "0" "1" "1" "1" "1")
    OFFLOAD_OPTIM=("0" "0" "1" "1" "0" "0" "1" "1")
    OFFLOAD_PARAM=("0" "1" "0" "1" "0" "1" "0" "1")

    export TILEARN_HYBRID_MODE='AutoZero'
    #export TILEARN_HYBRID_AUTOZERO_SHARD_PARAM=1
    #export TILEARN_HYBRID_AUTOZERO_OFFLOAD_OPTIM=0
    #export TILEARN_HYBRID_AUTOZERO_OFFLOAD_PARAM=1
    export TILEARN_HYBRID_FUSE_NORMALIZATION=1
    export TIACC_FASTER_RMS=0


elif [ $CASE_INDEX -eq 1 ]; then

    BS=("1" "1" "1" "1" "1" "1" "1" "1")
    GRAD_ACC=("4" "4" "4" "4" "4" "4" "4" "4")
    DIS_GRAD_CKPT=("1" "1" "1" "1" "1" "1" "1" "1")
    SHARD_PARAM=("0" "0" "0" "0" "1" "1" "1" "1")
    OFFLOAD_OPTIM=("0" "0" "1" "1" "0" "0" "1" "1")
    OFFLOAD_PARAM=("0" "1" "0" "1" "0" "1" "0" "1")

    export TILEARN_HYBRID_MODE='AutoZero'
    #export TILEARN_HYBRID_AUTOZERO_SHARD_PARAM=1
    #export TILEARN_HYBRID_AUTOZERO_OFFLOAD_OPTIM=0
    #export TILEARN_HYBRID_AUTOZERO_OFFLOAD_PARAM=1
    export TILEARN_HYBRID_FUSE_NORMALIZATION=1
    export TIACC_FASTER_RMS=0

fi


if [ -d $BASE_LOG_PATH ]; then
    rm -r $BASE_LOG_PATH
    mkdir -p $BASE_LOG_PATH
    echo "rm -r $BASE_LOG_PATH"
else
    mkdir -p $BASE_LOG_PATH
fi


length=${#BS[@]}

for ((i=0; i<$length; i++))
do
    bs=${BS[$i]}
    dis_grad_ckpt=${DIS_GRAD_CKPT[$i]}
    grad_acc=${GRAD_ACC[$i]}
    shard_param=${SHARD_PARAM[$i]}
    offload_optim=${OFFLOAD_OPTIM[$i]}
    offload_param=${OFFLOAD_PARAM[$i]}

    #export TILEARN_HYBRID_AUTOZERO_SHARD_PARAM=1
    #export TILEARN_HYBRID_AUTOZERO_OFFLOAD_OPTIM=0
    #export TILEARN_HYBRID_AUTOZERO_OFFLOAD_PARAM=1

    USE_TILEARN=${RUN_TILEARN}

    CONFIG="CASE_INDEX:${CASE_INDEX}, USE_TILEARN:${USE_TILEARN}, BS:${bs}, DIS_GRAD_CKPT:${dis_grad_ckpt}, GRAD_ACC:${grad_acc}, SHARD_PARAM:${shard_param}, OFFLOAD_OPTIM:${offload_optim}, OFFLOAD_PARAM:${offload_param}"
    echo " "
    echo $CONFIG

    export TILEARN_HYBRID_AUTOZERO_SHARD_PARAM=$shard_param
    export TILEARN_HYBRID_AUTOZERO_OFFLOAD_OPTIM=$offload_optim
    export TILEARN_HYBRID_AUTOZERO_OFFLOAD_PARAM=$offload_param
    export LF_UT_BS=$bs
    export LF_UT_DIS_GRAD_CKPT=$dis_grad_ckpt
    export LF_UT_GRAD_ACC=$grad_acc

    CMD="bash ./${SCRIPT}"
    LOG_PATH="${BASE_LOG_PATH}/${SCRIPT}.Case${CASE_INDEX}_Tilearn${USE_TILEARN}_BS${bs}_DisGradCkpt${dis_grad_ckpt}_GradACC${grad_acc}_ShardParam${shard_param}_OffloadOptim${offload_optim}_OffloadParam${offload_param}.log"
    echo "USE_TILEARN:${USE_TILEARN} - ${CMD} ${USE_TILEARN} > ${LOG_PATH} 2>&1"
    eval ${CMD} ${USE_TILEARN} > ${LOG_PATH} 2>&1

    errorCode=$?
    if [ $errorCode -ne 0 ]; then
        echo ${SCRIPT} error!!!
    else
        grep "train_samples_per_second" ${LOG_PATH}
        grep "20/${LF_UT_MAX_STEPS}" ${LOG_PATH} -A 11
        echo ${SCRIPT} pass!!!
    fi

    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 10
    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 10
    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 10

done
