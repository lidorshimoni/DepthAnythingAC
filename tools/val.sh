#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/depthanything_AC_vits.yaml
torchrun \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    evaluate_depth.py \
    --config=$config  \
    --port $2 \
    