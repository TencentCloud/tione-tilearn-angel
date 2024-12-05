datename=$(date +%Y%m%d-%H%M%S)
echo $datename

export BASE_LOG_PATH=${BASE_LOG_PATH:-"./log/${datename}-log_run_one_test_benchmark/"}
export GPU_MEM=${GPU_MEM:-'96G'}
############################## test 
#export LF_UT_MODEL_PATH_PREFIX="../../models ||| ../../models"
#export LF_MODEL_RANDOM_INIT=1
##############################


RUN_TILEARN=${1:-0}
#SCRIPT=test_qwen2.5_7b.sh
SCRIPT=${SCRIPT:-'test_qwen2.5_7b.sh'}

#TP=("2" "1" "2")
#PP=("2" "2" "1")
BS=("1" "1" "1" "1" "1" "1" "1" "1" "1" "1")
DIS_GRAD_CKPT=("1" "0" "1" "0" "1" "0" "1" "0" "1" "0")
GRAD_ACC=("4" "4" "4" "4" "4" "4" "4" "4" "4" "4")
DSYAML_PATH_PREFIX=("not_set ||| ../../utils/ds_config/ds_z3_config.json" \
	            "not_set ||| ../../utils/ds_config/ds_z3_config.json" \
                    "not_set ||| ../../utils/ds_config/ds_z2_config.json" \
                    "not_set ||| ../../utils/ds_config/ds_z2_config.json" \
		    "not_set ||| ../../utils/ds_config/ds_z1_config.json" \
		    "not_set ||| ../../utils/ds_config/ds_z1_config.json" \
		    "not_set ||| ../../utils/ds_config/ds_z3_offload_config.json" \
		    "not_set ||| ../../utils/ds_config/ds_z3_offload_config.json" \
		    "not_set ||| ../../utils/ds_config/ds_z2_offload_config.json" \
		    "not_set ||| ../../utils/ds_config/ds_z2_offload_config.json" \
	            )

# llama factory model random initialization
export LF_UT_TEST=${LF_UT_TEST:-1}
#export LF_MODEL_RANDOM_INIT=${LF_MODEL_RANDOM_INIT:-0}
#export LF_UT_MODEL_PATH_PREFIX=${LF_UT_MODEL_PATH_PREFIX:-"../../models ||| /mnt/cfs/tilearn/pretrain_models"}
export LF_UT_MAX_LENGTH=${LF_UT_MAX_LENGTH:-4096}
export LF_UT_MAX_STEPS=${LF_UT_MAX_STEPS:-50}
export LF_UT_LOG_STEPS=${LF_UT_LOG_STEPS:-10}

if [ -d $BASE_LOG_PATH ]; then
    rm -r $BASE_LOG_PATH
    mkdir -p $BASE_LOG_PATH
else
    mkdir -p $BASE_LOG_PATH
fi


length=${#BS[@]}

for ((i=0; i<$length; i++))
do
    #tp=${TP[$i]}
    #pp=${PP[$i]}
    bs=${BS[$i]}
    dis_grad_ckpt=${DIS_GRAD_CKPT[$i]}
    grad_acc=${GRAD_ACC[$i]}
    ds_yaml=${DSYAML_PATH_PREFIX[$i]}
    USE_TILEARN=${RUN_TILEARN}

    CONFIG="USE_TILEARN:${USE_TILEARN}, DSYAML_PATH_PREFIX:${ds_yaml} BS:${bs}, DIS_GRAD_CKPT:${dis_grad_ckpt}, GRAD_ACC:${grad_acc}"
    echo " "
    echo $CONFIG

    #export TILEARN_HYBRID_TP_SIZE=$tp
    #export TILEARN_HYBRID_PP_SIZE=$pp
    export LF_UT_BS=$bs
    export LF_UT_DIS_GRAD_CKPT=$dis_grad_ckpt
    export LF_UT_GRAD_ACC=$grad_acc
    export LF_UT_DSYAML_PATH_PREFIX=$ds_yaml

    CMD="bash ./${SCRIPT}"
    ds_yaml_sed=${ds_yaml// ||| /_to_}
    LOG_PATH="${BASE_LOG_PATH}/${SCRIPT}.Tilearn${USE_TILEARN}_DSYAM${ds_yaml_sed}_BS${bs}_DisGradCkpt${dis_grad_ckpt}_GradACC${grad_acc}.log"
    echo "USE_TILEARN:${USE_TILEARN} - ${CMD} ${USE_TILEARN} > ${LOG_PATH} 2>&1"

    eval ${CMD} ${USE_TILEARN} > ${LOG_PATH} 2>&1

    errorCode=$?
    if [ $errorCode -ne 0 ]; then
        echo ${SCRIPT} error!!!
    else
        grep "train_samples_per_second" ${LOG_PATH}
        grep "20/50" ${LOG_PATH} -A 11
        echo ${SCRIPT} pass!!!
    fi

    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 10
    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 10
    pkill -9 -f train_tilearn.py && pkill -9 -f train_baseline.py
    sleep 10

done
