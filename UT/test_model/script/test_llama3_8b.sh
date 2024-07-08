export MODEL_PATH="../models/Meta-Llama-3-8B/"
export USE_TILEARN=${1:-0}
export TIACC_FASTER_ROPE=1

REAL_TEST=${REAL_TEST:-"0"}
if [ "$REAL_TEST" = "80G" ]; then
    export HOST_GPU_NUM=8 #default 1
    export DS_CONFIG="--deepspeed=../ds_config/ds_config_zero2.json" # default ""
    export MAX_STEP=40 #default 10
    export SEQ_LENGTH=4096  #default 1024
    export BS=1 #default 1
    export CONFIG_NAME="config.json" #default "config_tiny.json"
    export GRADIENT_ACCUMULATION_STEPS=32 #default 1
    export GRADIENT_CHECKPOINTING="" #default "--gradient_checkpointing"
    export TRUST_REMOTE_CODE= "" ##default "" can set "trust_remote_code"
fi

bash run_clm_base.sh
