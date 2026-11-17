#!/usr/bin/env bash
set -e

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CONFIGURE_LOGGING=0
export OMP_NUM_THREADS=1
python ./vllm_runner.py --cues ~/datasets/master_cues.json
