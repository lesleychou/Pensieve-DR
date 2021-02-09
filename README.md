# Pensieve DR

## Setup

```
conda create -n pensieve-dr python=3.7
conda activate pensieve-dr
pip install tensorflow==1.14.0 visdom numpy tflearn
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
       --summary_dir "../../results/seed_1" \
       --model_path "../data/sanity-check-3/model_saved/nn_model_ep_5200.ckpt" \
       --random_seed=1 \
       --ROBUST_NOISE=0 \
       --SAMPLE_LENGTH=0 \
       --NUMBER_PICK=0
```