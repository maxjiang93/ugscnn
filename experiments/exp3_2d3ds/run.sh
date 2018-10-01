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
	wget --no-check-certificate http://island.me.berkeley.edu/ugscnn/data/2d3ds_sphere.zip

	# setup data
	unzip 2d3ds_sphere.zip
	mv 2d3ds_sphere data
	rm 2d3ds_sphere.zip
fi

# create log directory
mkdir -p logs

#source activate

python train.py \
--batch-size 16 \
--test-batch-size 16 \
--epochs 200 \
--data_folder $DATADIR \
--max_level 5 \
--min_level 0 \
--feat 16 \
--fold 3 \
--log_dir logs/log_f16_cv3 \
--decay \
--in_ch rgbd
