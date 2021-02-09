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



