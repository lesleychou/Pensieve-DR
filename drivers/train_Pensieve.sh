#! /bin/bash

# immediately exit the bash if an error encountered
set -e

NOISE=0
DURATION=1
TRAIN_TRACE_PATH="../data/generated_traces_random/train"
VAL_TRACE_PATH="../data/generated_traces_random/fixed-test/val_FCC"
# TRAIN_TRACE_PATH="../data/exponential_traces/train"
# VAL_TRACE_PATH="../data/exponential_traces/val"
# TEST_TRACE_PATH="../data/exponential_traces/test"
# TRAIN_TRACE_PATH="../data/step_traces/train"
# VAL_TRACE_PATH="../data/step_traces/val"
# TEST_TRACE_PATH="../data/step_traces/test"
# TRAIN_TRACE_PATH="../data/step_traces_period20/train"
# VAL_TRACE_PATH="../data/step_traces_period20/val"
# TEST_TRACE_PATH="../data/step_traces_period20/test"
# TRAIN_TRACE_PATH="../data/step_traces_period40_changing_peak/train"
# VAL_TRACE_PATH="../data/step_traces_period40_changing_peak/val"
# TEST_TRACE_PATH="../data/step_traces_period40_changing_peak/test"
# TRAIN_TRACE_PATH="../data/step_traces_period50/train"
# VAL_TRACE_PATH="../data/step_traces_period50/val"
# TEST_TRACE_PATH="../data/step_traces_period50/test"
# TRAIN_TRACE_PATH="../data/constant_trace/train"
# VAL_TRACE_PATH="../data/constant_trace/val"
# TEST_TRACE_PATH="../data/constant_trace/test"
SIMULATOR_DIR="../sim"
#--nn_model='../DR-results/fifth/model_saved/nn_model_ep_1100.ckpt'\


LOG_FILES=( '0' '1' '2' '3' )

for NOISE in 0 ; do
    # SUMMARY_DIR="../results/noise_exp/noise_${NOISE}"
    # SUMMARY_DIR="../results/tmp/noise_${NOISE}"
    # SUMMARY_DIR="../results/exponential_traces/results_noise_${NOISE}_duration_${DURATION}"
    # SUMMARY_DIR="../results/step_traces_period20/results_noise_${NOISE}_duration_${DURATION}"
    # SUMMARY_DIR="../results/step_traces_period40_changing_peak/results_noise_${NOISE}_duration_${DURATION}"
    # SUMMARY_DIR="../results/step_traces_period50/results_noise_${NOISE}_duration_${DURATION}"
    # SUMMARY_DIR="../results/constant_trace/results_noise_${NOISE}_duration_${DURATION}"
    # SUMMARY_DIR="../results/eval_train_e2e/results_noise_${NOISE}_duration_${DURATION}"
    # SUMMARY_DIR="../results/eval_train_e2e_new/results_noise_${NOISE}_duration_${DURATION}"
    SUMMARY_DIR="../UDR-results/rtt-UDR/"
    python ${SIMULATOR_DIR}/multi_agent.py \
        --RANDOM_SEED=171 \
        --NUM_AGENT=8\
        --A_DIM=3\
        --S_LEN=6\
        --train_trace_dir ${TRAIN_TRACE_PATH} \
        --val_trace_dir ${VAL_TRACE_PATH} \
        --summary_dir ${SUMMARY_DIR} \
        --noise ${NOISE} \
        --duration ${DURATION} \
        --description="jump-action"
        #--nn_model='../BO-results/randomize-BW-TS-0-with-trained-model/model_saved/nn_model_ep_100.ckpt'


done
