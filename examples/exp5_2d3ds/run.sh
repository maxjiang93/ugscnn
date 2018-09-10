#!/bin/bash
source activate
python train.py \
--batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--data_folder /data1/tmp/maxjiang/climate_data/data_5_all \
--max_level 7 \
--min_level 0 \
--feat 4 \
--fold 0 \
--log_dir log_f4_cv0 \
--decay
