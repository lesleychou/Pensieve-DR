import sys
import subprocess
import os
from bayes_opt import BayesianOptimization


def latest_actor_from(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    files = list(sorted(os.listdir(path), key=mtime))
    actors = [a for a in files if "nn_model_ep_" in a]
    actor_path = str(path + '/' + actors[-1])
    return actor_path


def black_box_function(x):
    '''
    :param x: input is the current params
    :return: reward is the mpc-rl reward
    '''
    # TODO: this need to be args.summary_dir
    # TODO: do i need to load the actor_path here?
    # path = './new-DR-results/sanity-check-3/model_saved'
    #actor_path = latest_actor_from(path)
    latest_model_path = "../data/sanity-check-3/model_saved/nn_model_ep_5200.ckpt"

    # --test-trace-dir='../pensieve/data/generated_traces_ts_float/train/123/' \
    #   --video-size-file-dir='../pensieve/data/video_size_6_larger/' \
    #   --summary-dir='../pensieve/tmp/2021021750'

    command = " python rl_test.py  \
                --CURRENT_PARAM={current_max_tp_param} \
                --test_trace_dir='../data/example_traces/' \
                --summary_dir='../MPC_RL_test_results/' \
                --model_path='{model_path}' \
                ".format(current_max_tp_param=x, model_path=latest_model_path)

    r = float(subprocess.check_output(command, shell=True, text=True).strip())
    print(r)
    return r

pbounds = {'x': (0 ,100)}
optimizer = BayesianOptimization(
        f=black_box_function ,
        pbounds=pbounds ,
        random_state=2,
    )

optimizer.maximize(
    init_points=1,
    n_iter=1,
    kappa=10,
    xi=0.1
)
next = optimizer.max
param = next.get( 'params' ).get( 'x' )
current_params = round( param ,2 )
print(current_params)
