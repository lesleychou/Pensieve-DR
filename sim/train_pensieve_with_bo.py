# This is the main training setup for training a Pensieve model, but
# stopping at intervals to add new training data based on performance
# (increases generalizability, we hope!)
import os
import subprocess
# Inputs:
#
# - experiment results directory
# - training data directory (with traces in subdirectories)
# - total epochs
# - bayesian optimizer interval (e.g., every 5000 epochs) - this is how many epoch each training run will go

# Defaults
# Improvement: Probably better if replaced with argparse and passed in (later)
TOTAL_EPOCHS = 10000
BAYESIAN_OPTIMIZER_INTERVAL = 1000
TRAINING_DATA_DIR = "../data/training_default/"
VAL_TRACE_DIR = '../data/generated_traces_ts_float-BO/val'
RESULTS_DIR = "../results/bo_example/"
NN_MODEL='../new-DR-results/sanity-check-1/model_saved/nn_model_ep_3900.ckpt'

num_training_runs = int(TOTAL_EPOCHS / BAYESIAN_OPTIMIZER_INTERVAL)

# Example Flow:
for i in range(2):
    if i > 0:
        print("Running bayesian optimization to get new training data input parameter.")
        print("Run test generation based on BO output and put in training data dir.")
        command = "python bo.py"
        new_training_param = float( subprocess.check_output( command ,shell=True ,text=True ).strip() )
        print("new_training_param:", new_training_param)

    print("Check results dir for any saved model file.")
    print("If it exists, take the latest.")
    command = "python multi_agent.py \
                    --TOTAL_EPOCH=10\
                    --train_trace_dir={training_dir} \
                    --val_trace_dir='{val_dir}'\
                    --summary_dir={results_dir}\
                    --description='first-run' \
                    --nn_model={model_path}" \
                    .format(training_dir=TRAINING_DATA_DIR, val_dir=VAL_TRACE_DIR, results_dir=RESULTS_DIR, model_path=NN_MODEL)
    os.system(command)

    print("Get the file and pass it to the training script, if it exists.\n")
    print("Running training:", i)
    i += 1

print("Hooray!")
