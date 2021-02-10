# This is the main training setup for training a Pensieve model, but
# stopping at intervals to add new training data based on performance
# (increases generalizability, we hope!)
import sys
import subprocess
import numpy as np
import os
from bayes_opt import BayesianOptimization
# Inputs:
#
# - experiment results directory
# - training data directory (with traces in subdirectories)
# - total epochs
# - bayesian optimizer interval (e.g., every 5000 epochs) - this is how many epoch each training run will go

# Defaults
# Improvement: Probably better if replaced with argparse and passed in (later)
# TOTAL_EPOCHS = 10000
# BAYESIAN_OPTIMIZER_INTERVAL = 1000
TRAINING_DATA_DIR = "../data/generated_traces_ts_float-BO/train/"
VAL_TRACE_DIR = '../data/generated_traces_ts_float-BO/val'
RESULTS_DIR = "../results/bo_example/"
#NN_MODEL='../new-DR-results/sanity-check-2/model_saved/nn_model_ep_33200.ckpt'

# num_training_runs = int(TOTAL_EPOCHS / BAYESIAN_OPTIMIZER_INTERVAL)

MIN_BW = 1
MAX_BW = 1000


def map_lin_to_log(x):
    x_log = (np.log(x) - np.log(MIN_BW)) / (np.log(MAX_BW) - np.log(MIN_BW))
    return x_log


def map_log_to_lin(x):
    x_lin = np.exp((np.log(MAX_BW)-np.log(MIN_BW))*x + np.log(MIN_BW))
    return x_lin


def latest_actor_from(path):
    """
    Returns latest tensorflow checkpoint file from a directory.
    Assumes files are named:
    nn_model_ep_<EPOCH#>.ckpt.meta
    """
    mtime = lambda f: os.stat( os.path.join( path ,f ) ).st_mtime
    files = list( sorted( os.listdir( path ) ,key=mtime ) )
    actors = [a for a in files if "nn_model_ep_" in a]
    actor_path = str( path + '/' + actors[-1] )
    return os.path.splitext( actor_path )[0]


def black_box_function(x):
    '''
    :param x: input is the current params
    :return: reward is the mpc-rl reward
    '''
    # TODO: this need to be args.summary_dir
    # TODO: do i need to load the actor_path here?
    path = os.path.join( RESULTS_DIR, 'model_saved' )
    latest_model_path = latest_actor_from(path)
    #print(latest_model_path)

    x_map = map_log_to_lin(x)

    command = " python rl_test.py  \
                --CURRENT_PARAM={current_max_tp_param} \
                --test_trace_dir='../data/example_traces/' \
                --summary_dir='../MPC_RL_test_results/' \
                --model_path='{model_path}' \
                ".format(current_max_tp_param=x_map, model_path=latest_model_path)

    r = float(subprocess.check_output(command, shell=True, text=True).strip())
    return r


# Example Flow:
for i in range(10):
    # if i > 0:
    pbounds = {'x': (0 ,1)}
    optimizer = BayesianOptimization(
        f=black_box_function ,
        pbounds=pbounds
        #random_state=2
    )

    optimizer.maximize(
        init_points=5 ,
        n_iter=5,
        kappa=20,
        xi=0.1
    )
    next = optimizer.max
    param = next.get( 'params' ).get( 'x' )
    #bo_best_param = round( param ,2 )
    bo_best_param = map_log_to_lin(param)
    print( "BO chose this best param........", bo_best_param )

    # Use the new param, add more traces into Pensieve, train more based on before
    path = os.path.join( RESULTS_DIR ,'model_saved' )
    latest_model_path = latest_actor_from( path )

    command = "python multi_agent.py \
                    --TOTAL_EPOCH=3000\
                    --train_trace_dir={training_dir} \
                    --val_trace_dir='{val_dir}'\
                    --summary_dir={results_dir}\
                    --description='first-run' \
                    --nn_model={model_path} \
                    --CURRENT_PARAM={bo_output_param}"  \
                    .format(training_dir=TRAINING_DATA_DIR, val_dir=VAL_TRACE_DIR,
                            results_dir=RESULTS_DIR, model_path=latest_model_path, bo_output_param=bo_best_param)
    os.system(command)

    print("Get the file and pass it to the training script, if it exists.\n")
    print("Running training:", i)
    i += 1

print("Hooray!")
