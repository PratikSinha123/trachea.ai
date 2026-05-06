#!/bin/bash

export CUDA_VISIBLE_DEVICES=""
export TOTALSEG_DEVICE=cpu

source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate ai_env

cd /home/pratiksinha1064/trachea.ai

python auto_train.py \
  --database /home/pratiksinha1064/dataset \
  --max-patients 10 \
  --epochs 1 \
  --device cpu
