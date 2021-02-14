# Pensieve DR

## Setup

```
conda create -n pensieve-dr python=3.7
conda activate pensieve-dr
pip install tensorflow==1.14.0 visdom numpy tflearn numba bayesian-optimization scikit-learn==0.22
pip list
```

## Get Data

`data.zip` is provided separately

```
cd data
cp data.zip .
unzip data.zip
```

All data trace directories are subdirs of `./data`.

## Running Tests

Example:

```
cd sim
python rl_test.py \
       --summary_dir="../MPC_RL_test_results/" \
       --model_path="../data/sanity-check-3/model_saved/nn_model_ep_5200.ckpt" \
       --test_trace_dir="../data/example_traces" \
       --CURRENT_PARAM=200
```


python rl_test.py  \
            --CURRENT_PARAM=10 \
            --test_trace_dir='../data/example_traces/' \
            --summary_dir='../MPC_RL_test_results/' \
            --model_path='../results/bo_example/model_saved/nn_model_ep_300.ckpt'
     
```       
python rl_test.py  \
            --CURRENT_PARAM=2 \
            --test_trace_dir='../data/puffer-val-data/' \
            --summary_dir='../MPC_RL_test_results/' \
            --model_path='../BO-results/randomize-TS-4th/model_saved/nn_model_ep_2500.ckpt'
```