import sys
sys.path.append("../")

import subprocess
import os
from bayes_opt import BayesianOptimization


def latest_actor_from(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    files = list(sorted(os.listdir(path), key=mtime))
    actors = [a for a in files if "actor_ep_" in a]
    actor_path = str(path + '/' +actors[-1])
    return actor_path


def black_box_function(x):
    '''
    :param x: input is the current params
    :return: reward is the mpc-rl reward
    '''
    # TODO: this need to be args.summary_dir
    # TODO: do i need to load the actor_path here?
    path = '../pensieve/tmp/results-sanity-check-2'
    #actor_path = latest_actor_from(path)

    # --test-trace-dir='../pensieve/data/generated_traces_ts_float/train/123/' \
    #   --video-size-file-dir='../pensieve/data/video_size_6_larger/' \
    #   --summary-dir='../pensieve/tmp/2021021750'

    command = " python test_pensieve.py  \
                --CURRENT_PARAM={current_max_tp_param} \
                --video-size-file-dir='../pensieve/data/video_size_6_larger'   \
                --test-trace-dir='../pensieve/data/generated_traces_ts_float/train/123/' \
                --summary-dir='../pensieve/tmp/2021021750'" \
                .format(current_max_tp_param=x)

    r = float(subprocess.check_output(command, shell=True, text=True).strip())
    return r

pbounds = {'x': (2 ,1000)}
optimizer = BayesianOptimization(
        f=black_box_function ,
        pbounds=pbounds ,
        random_state=2,
    )

optimizer.maximize(
    init_points=1 ,
    n_iter=2 ,
    kappa=10,
    xi=0.1
)
next = optimizer.max
param = next.get( 'params' ).get( 'x' )
current_params = round( param ,2 )
print(current_params)

