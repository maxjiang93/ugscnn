#!/bin/bash
source activate
python train.py \
--batch-size 256 \
--test-batch-size 256 \
--epochs 100 \
--data_folder /data1/tmp/maxjiang/climate_data/data_5_all \
--max_level 5 \
--min_level 0 \
--feat 8 \
--log_dir log_f8 \
--decay