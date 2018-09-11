#!/bin/bash
source activate
python train.py \
--batch-size 36 \
--test-batch-size 36 \
--epochs 200 \
--data_folder data \
--max_level 7 \
--min_level 0 \
--feat 4 \
--fold 1 \
--log_dir log_f4_cv1_ceil \
--decay
