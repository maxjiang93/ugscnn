#!/bin/bash
source activate
python train.py \
--batch-size 16 \
--test-batch-size 16 \
--epochs 200 \
--data_folder data \
--max_level 5 \
--min_level 0 \
--feat 16 \
--fold 3 \
--log_dir logs/log_f16_cv3 \
--decay \
--in_ch rgbd
