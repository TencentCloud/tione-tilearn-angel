export GPU_MEM='40G'

RUN_TILEARN=${1:-1}
SCRIPT=test_llama31_8b.sh

TP=("2" "1" "2")
PP=("2" "2" "1")
BS=("1" "1" "1")
DIS_GRAD_CKPT=("1" "0" "0")
GRAD_ACC=("16" "8" "8")

if [ -d log_run_one_test/ ]; then
    rm log_run_one_test/*
else
    mkdir log_run_one_test/
fi


length=${#TP[@]}

for ((i=0; i<$length; i++))
do
    tp=${TP[$i]}
    pp=${PP[$i]}
    bs=${BS[$i]}
    dis_grad_ckpt=${DIS_GRAD_CKPT[$i]}
    grad_acc=${GRAD_ACC[$i]}
    USE_TILEARN=${RUN_TILEARN}

    CONFIG="USE_TILEARN:${USE_TILEARN}, TP:${tp}, PP:${pp}, BS:${bs}, DIS_GRAD_CKPT:${dis_grad_ckpt}, GRAD_ACC:${grad_acc}"
    echo " "
    echo $CONFIG

    export TILEARN_HYBRID_TP_SIZE=$tp
    export TILEARN_HYBRID_PP_SIZE=$pp
    export LF_UT_BS=$bs
    export LF_UT_DIS_GRAD_CKPT=$dis_grad_ckpt
    export LF_UT_GRAD_ACC=$grad_acc

    CMD="bash ./${SCRIPT}"
    LOG_PATH="./log_run_one_test/${SCRIPT}.Tilearn${USE_TILEARN}_TP${tp}_PP${pp}_BS${bs}_DisGradCkpt${dis_grad_ckpt}_GradACC${grad_acc}.log"
    echo "USE_TILEARN:${USE_TILEARN} - ${CMD} ${USE_TILEARN} > ${LOG_PATH} 2>&1"
    eval ${CMD} ${USE_TILEARN} > ${LOG_PATH} 2>&1

    errorCode=$?
    if [ $errorCode -ne 0 ]; then
        echo ${SCRIPT} error!!!
    else
        grep "train_samples_per_second" ${LOG_PATH}
        grep "30/50" $-A 8
        echo ${SCRIPT} pass!!!
    fi

    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 10
    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 10
    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 10

done
