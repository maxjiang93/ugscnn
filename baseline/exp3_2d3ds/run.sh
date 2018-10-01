#!/bin/bash
MESHFILES=../../mesh_files
DATADIR=data

if [ ! -d $MESHFILES ]; then
    echo "[!] Mesh files do not exist..."
    exit
fi

# generate data
if [ ! -d $DATADIR ]; then
    echo "[!] Data files do not exist. Preparing data..."
    # download preprocessed spherical data
	wget --no-check-certificate http://island.me.berkeley.edu/ugscnn/data/2d3ds_pano_small.zip

	# setup data
	unzip 2d3ds_pano_small.zip
	mv 2d3ds_pano_small data
	rm 2d3ds_pano_small.zip
fi

# create log directory
mkdir -p logs

#source activate

python train.py \
--batch-size 16 \
--test-batch-size 16 \
--epochs 200 \
--data_folder data \
--fold 3 \
--log_dir logs/log_unet_f16_cv3_rgbd \
--decay \
--train_stats_freq 5 \
--model UNet \
--in_ch rgbd \
--lr 1e-3 \
--feat 16

# FCN8s, UNet

