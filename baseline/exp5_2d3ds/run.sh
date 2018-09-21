#!/bin/bash
source activate
python train.py \
--batch-size 4 \
--test-batch-size 4 \
--epochs 200 \
--data_folder data \
--max_level 5 \
--min_level 0 \
--feat 4 \
--fold 2 \
--log_dir log/log_f32_cv2_l5_lw \
--decay \
--train_stats_freq 5

#--resume log_f8_cv1_l5_lw/checkpoint_latest.pth.tar_UNet_127.pth.tar
