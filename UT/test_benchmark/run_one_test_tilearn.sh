datename=$(date +%Y%m%d-%H%M%S)
echo $datename

export GPU_MEM=${GPU_MEM:-'96G'}
export LF_UT_MAX_STEPS=${LF_UT_MAX_STEPS:-40}
export BASE_LOG_PATH=${BASE_LOG_PATH:-"./log/${datename}-log_run_one_test/"}

############################## test 
#export LF_UT_MODEL_PATH_PREFIX="../../models ||| ../../models"
#export LF_MODEL_RANDOM_INIT=1
##export ENABLE_FP8=1
###export LD_LIBRARY_PATH=/mnt/cfs/boyyang/cublas-x86_64-centos7-cuda12.3_r545/lib64/:$LD_LIBRARY_PATH
##export LD_LIBRARY_PATH=//mnt/data/boyyang/cublas-x86_64-centos7-cuda12.3_r545/lib64/:$LD_LIBRARY_PATH
##############################


CASE_INDEX=${1:-3}
RUN_TILEARN=1

#SCRIPT=test_qwen2.5_7b.sh
SCRIPT=${SCRIPT:-'test_qwen2.5_7b.sh'}

if [ $CASE_INDEX -eq 0 ]; then

    #TP=("1" "1" "1" "1" "1" "1" "1" "1")
    #PP=("1" "1" "1" "1" "1" "1" "1" "1")
    #BS=("1" "1" "1" "1" "1" "1" "1" "1")
    #DIS_GRAD_CKPT=("1" "0" "1" "0" "1" "0" "1" "0")
    #GRAD_ACC=("4" "4" "4" "4" "4" "4" "4" "4")
    #OFFLOAD=("0" "0" "1" "1" "0" "0" "1" "1")
    #ZERO_STAGE=("1" "1" "1" "1" "2" "2" "2" "2")

    TP=("1" "1")
    PP=("1" "1")
    BS=("1" "1")
    DIS_GRAD_CKPT=("1" "0")
    GRAD_ACC=("4" "4")
    OFFLOAD=("0" "0")
    ZERO_STAGE=("1" "1")

    export TILEARN_HYBRID_MODE='None'

elif [ $CASE_INDEX -eq 1 ]; then

    TP=("2" "2" "2" "2" "2" "2" "2" "2")
    PP=("1" "1" "1" "1" "1" "1" "1" "1")
    BS=("1" "1" "1" "1" "1" "1" "1" "1")
    DIS_GRAD_CKPT=("1" "0" "1" "0" "1" "0" "1" "0")
    GRAD_ACC=("8" "8" "8" "8" "8" "8" "8" "8")
    OFFLOAD=("0" "0" "1" "1" "0" "0" "1" "1")
    ZERO_STAGE=("1" "1" "1" "1" "2" "2" "2" "2")

    export TILEARN_HYBRID_MODE='default'

elif [ $CASE_INDEX -eq 2 ]; then

    TP=("1" "2" "1" "4" "2" "2" "4" "1" "2" "1" "4" "2" "2" "4")
    PP=("2" "1" "4" "1" "2" "4" "2" "2" "1" "4" "1" "2" "4" "2")
    BS=("1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1")
    DIS_GRAD_CKPT=("1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1")
    GRAD_ACC=("8" "8" "16" "16" "16" "32" "32" "8" "8" "16" "16" "16" "32" "32")
    OFFLOAD=("0" "0" "0" "0" "0" "0" "0" "1" "1" "1" "1" "1" "1" "1")
    ZERO_STAGE=("1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1")

    export TILEARN_HYBRID_MODE='default'

elif [ $CASE_INDEX -eq 3 ]; then

    TP=("1" "2" "1" "4" "2" "2" "4" "1" "2" "1" "4" "2" "2" "4")
    PP=("2" "1" "4" "1" "2" "4" "2" "2" "1" "4" "1" "2" "4" "2")
    BS=("1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1")
    DIS_GRAD_CKPT=("0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0" "0")
    GRAD_ACC=("8" "8" "16" "16" "16" "32" "32" "8" "8" "16" "16" "16" "32" "32")
    OFFLOAD=("0" "0" "0" "0" "0" "0" "0" "1" "1" "1" "1" "1" "1" "1")
    ZERO_STAGE=("1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1" "1")

    export TILEARN_HYBRID_MODE='default'

fi


if [ -d $BASE_LOG_PATH ]; then
    rm -r $BASE_LOG_PATH
    mkdir -p $BASE_LOG_PATH
    echo "rm -r $BASE_LOG_PATH"
else
    mkdir -p $BASE_LOG_PATH
fi


length=${#TP[@]}

for ((i=0; i<$length; i++))
do
    tp=${TP[$i]}
    pp=${PP[$i]}
    bs=${BS[$i]}
    dis_grad_ckpt=${DIS_GRAD_CKPT[$i]}
    grad_acc=${GRAD_ACC[$i]}
    offload=${OFFLOAD[$i]}
    zero_stage=${ZERO_STAGE[$i]}

    USE_TILEARN=${RUN_TILEARN}

    CONFIG="CASE_INDEX:${CASE_INDEX}, USE_TILEARN:${USE_TILEARN}, TP:${tp}, PP:${pp}, BS:${bs}, DIS_GRAD_CKPT:${dis_grad_ckpt}, GRAD_ACC:${grad_acc}, OFFLOAD:${offload}, ZERO_STAGE:${zero_stage}"
    echo " "
    echo $CONFIG

    export TILEARN_HYBRID_TP_SIZE=$tp
    export TILEARN_HYBRID_PP_SIZE=$pp
    export TILEARN_HYBRID_OFFLOAD=$offload
    export TILEARN_HYBRID_ZERO_STAGE=$zero_stage
    export LF_UT_BS=$bs
    export LF_UT_DIS_GRAD_CKPT=$dis_grad_ckpt
    export LF_UT_GRAD_ACC=$grad_acc

    CMD="bash ./${SCRIPT}"
    LOG_PATH="${BASE_LOG_PATH}/${SCRIPT}.Case${CASE_INDEX}_Tilearn${USE_TILEARN}_TP${tp}_PP${pp}_BS${bs}_DisGradCkpt${dis_grad_ckpt}_GradACC${grad_acc}_Offload${offload}_ZeroStage${zero_stage}.log"
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
