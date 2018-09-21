#!/bin/bash
source activate
python train.py \
--batch-size 32 \
--test-batch-size 32 \
--epochs 200 \
--data_folder data_small \
--fold 3 \
--log_dir log/log_unet_cv3 \
--decay \
--train_stats_freq 5 \
--model UNet

#--resume log_f8_cv1_l5_lw/checkpoint_latest.pth.tar_UNet_127.pth.tar
