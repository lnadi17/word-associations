#!/bin/bash

export CC=gcc-12
export CXX=g++-12
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export VLLM_CONFIGURE_LOGGING=0
export OMP_NUM_THREADS=1
python ./vllm_runner.py --cues ~/master_cues.json --num_gpus 1
