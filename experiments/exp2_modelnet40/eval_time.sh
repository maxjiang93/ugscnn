#!/bin/bash
# export OMP_NUM_THREADS=1

FEAT=32
TY=""

python test.py --eval_time \
--batch_size 1 \
--neval 64 \
--feat $FEAT \
--log_dir logs/davinci/modelnet40_drop_ft${FEAT}_b16$TY \
--jobs 12 \
# --ty \
# --no_cuda
