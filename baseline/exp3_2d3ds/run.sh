#!/bin/bash
source activate
python train.py \
--batch-size 16 \
--test-batch-size 16 \
--epochs 200 \
--data_folder data_small \
--fold 3 \
--log_dir log/log_unet_f4_cv3_rgbd \
--decay \
--train_stats_freq 5 \
--model UNet \
--in_ch rgbd \
--lr 1e-3 \
--feat 4

# FCN8s, UNet, ResNetDUCHDC

