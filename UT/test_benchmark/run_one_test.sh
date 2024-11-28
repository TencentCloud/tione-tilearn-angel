export GPU_MEM='96G'
############################# test 
export LF_UT_MODEL_PATH_PREFIX="../../models ||| ../../models"
export LF_MODEL_RANDOM_INIT=1
#export ENABLE_FP8=1
##export LD_LIBRARY_PATH=/mnt/cfs/boyyang/cublas-x86_64-centos7-cuda12.3_r545/lib64/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=//mnt/data/boyyang/cublas-x86_64-centos7-cuda12.3_r545/lib64/:$LD_LIBRARY_PATH
#############################


#bash run_one_test_baseline.sh
#bash run_one_test_tilearn.sh 0
bash run_one_test_tilearn.sh 1
bash run_one_test_tilearn.sh 2
bash run_one_test_tilearn.sh 3
