sleep 4h

### Tilearn UT 
export GPU_MEM='40G'

RUN_BASELINE=1
RUN_TILEARN=1

if [ -d log_run_all_test/ ]; then
    rm log_run_all_test/*
else
    mkdir log_run_all_test/
fi

function run_task() {
    SCRIPT=$1
    USE_TILEARN=$2
    #cd ./script

    CMD="bash ./${SCRIPT}"
    LOG_PATH="./log_run_all_test/${SCRIPT}.tilearn_${USE_TILEARN}.log"
    echo " "
    echo "USE_TILEARN:${USE_TILEARN} - ${CMD} ${USE_TILEARN} > ${LOG_PATH} 2>&1"
    eval ${CMD} ${USE_TILEARN} > ${LOG_PATH} 2>&1

    errorCode=$?
    if [ $errorCode -ne 0 ]; then
        echo ${SCRIPT} error!!!
    else
	grep "train_samples_per_second" ${LOG_PATH}
        grep "30/50" ${LOG_PATH} -A 8
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
