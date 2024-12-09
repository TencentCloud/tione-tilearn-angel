#sleep 4h
datename=$(date +%Y%m%d-%H%M%S)
echo "run_all_test.sh $datename"

### Tilearn UT 
export GPU_MEM=${GPU_MEM:-'96G'}
#export GPU_MEM=${GPU_MEM:-'40G'}

export RUN_BASELINE=${RUN_BASELINE:-1}
export RUN_TILEARN=${RUN_TILEARN:-1}

export LF_UT_MODEL_PATH_PREFIX=${LF_UT_MODEL_PATH_PREFIX:-"../../models ||| ../../models"}
export LF_MODEL_RANDOM_INIT=${LF_MODEL_RANDOM_INIT:-1}
export BASE_LOG_PATH=${BASE_LOG_PATH:-"./log/${datename}-log_run_all_test/"}

export ENABLE_FP8=${ENABLE_FP8:-0}
#export LD_LIBRARY_PATH=/mnt/cfs/cublas-x86_64-centos7-cuda12.3_r545/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=//mnt/data/boyyang/cublas-x86_64-centos7-cuda12.3_r545/lib64/:$LD_LIBRARY_PATH

echo "run_all_test.sh GPU_MEM:${GPU_MEM} RUN_BASELINE:${RUN_BASELINE} RUN_TILEARN:${RUN_TILEARN} LF_UT_MODEL_PATH_PREFIX:${LF_UT_MODEL_PATH_PREFIX}, LF_MODEL_RANDOM_INIT:${LF_MODEL_RANDOM_INIT} BASE_LOG_PATH:${BASE_LOG_PATH} ENABLE_FP8:${ENABLE_FP8}"

if [ -d $BASE_LOG_PATH ]; then
    rm -r $BASE_LOG_PATH
    mdkir -p $BASE_LOG_PATH
    echo "rm -r $BASE_LOG_PATH"
else
    mkdir -p $BASE_LOG_PATH
fi

function run_task() {
    SCRIPT=$1
    USE_TILEARN=$2
    #cd ./script

    CMD="bash ./${SCRIPT}"
    LOG_PATH="${BASE_LOG_PATH}/${SCRIPT}.tilearn_${USE_TILEARN}.log"
    echo " "
    echo "USE_TILEARN:${USE_TILEARN} - ${CMD} ${USE_TILEARN} > ${LOG_PATH} 2>&1"
    eval ${CMD} ${USE_TILEARN} > ${LOG_PATH} 2>&1

    errorCode=$?
    if [ $errorCode -ne 0 ]; then
        echo ${SCRIPT} error!!!
    else
	grep "train_samples_per_second" ${LOG_PATH}
        grep "30/50" ${LOG_PATH} -A 11
        echo ${SCRIPT} pass!!!
    fi

    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 30
    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 30
    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 30

    #cd ..
    return $errorCode
}

function run_model() {
    SCRIPT=$1

    if [ $RUN_BASELINE -eq 1 ]; then
        run_task ${SCRIPT} 0
    fi
    if [ $RUN_TILEARN -eq 1 ]; then
        run_task ${SCRIPT} 1
    fi

}

run_model "test_llama2_7b.sh"
run_model "test_llama3_8b.sh"
run_model "test_llama31_8b.sh"
run_model "test_qwen_7b.sh"
run_model "test_qwen2_7b.sh"
run_model "test_qwen2.5_7b.sh"
#run_model "test_baichuan2_13b.sh"
#run_model "test_baichuan2_13b_old.sh"
#run_model "test_bloom_7b_old.sh"
