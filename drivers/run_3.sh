#! /bin/bash

# immediately exit the bash if an error encountered
set -e

DURATION=1
#TRACE_PATH="../data/Norway-DR-exp/val-norm-0.6-0.5-noise"
# TRACE_PATH="../sim/train_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
# TRACE_PATH="../data/test"
#TRACE_PATH="../data/train"
# SUMMARY_DIR="../results/duration_exp/duration_quarter"
# NN_MODELS=( "../sim/results_noise0.5/nn_model_ep_5300.ckpt" )
SIMULATOR_DIR="../sim"


# LOG_FILES=( 'pretrain' 'duration' 'duration_half' 'duration_quarter' 'duration_double' )
#LOG_FILES=( '1' '2' '3' '4' '5' '6' '7' '8')
LOG_FILES=( '1')



NN_MODELS="../DR-results/lesley-random/model_saved/nn_model_ep_7100.ckpt"


#for i_folder in 100 200 300 400 500 600 700 800 900; do
for (( i_folder=41; i_folder<=60; i_folder++ )); do
        TRACE_PATH="../data/synthetic_test_lesley_3/${i_folder}"
        SUMMARY_DIR="../results/pensieve-mpc-lesley-test-3/test-on-${i_folder}"

        #for ((i=0;i<${#NN_MODELS[@]};++i)); do
            python ${SIMULATOR_DIR}/rl_test.py \
                   --test_trace_dir ${TRACE_PATH} \
                   --summary_dir ${SUMMARY_DIR}/seed_1\
                   --model_path ${NN_MODELS} \
                   --random_seed=1 \
                   --ROBUST_NOISE=0 \
                   --SAMPLE_LENGTH=0 \
                   --NUMBER_PICK=0 \
                   --duration ${DURATION} &
##
#            python ${SIMULATOR_DIR}/rl_test_pensieve.py \
#                   --test_trace_dir ${TRACE_PATH} \
#                   --summary_dir ${SUMMARY_DIR}/seed_1\
#                   --model_path ${NN_MODELS} \
#                   --random_seed=1 \
#                   --ROBUST_NOISE=0 \
#                   --SAMPLE_LENGTH=0 \
#                   --NUMBER_PICK=0 \
#                   --duration ${DURATION} &
####
            python ${SIMULATOR_DIR}/mpc.py \
                 --test_trace_dir ${TRACE_PATH} \
                   --summary_dir ${SUMMARY_DIR}/seed_1\
                 --random_seed=1  \
                 --ROBUST_NOISE=0 \
                 --SAMPLE_LENGTH=0 \
                 --NUMBER_PICK=0 \
                 --duration ${DURATION}
##          #done

done