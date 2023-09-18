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
    echo "Warning. Large data. Need at least 80G of extra disk space."
    # download preprocessed spherical data
	# This line requires installing gdown: pip install gdown
        gdown 'https://drive.google.com/uc?id=1DxNh7NkfR3w8qK_1N6AmJ73jf5UM3meV'

	# setup data
	unzip climate_sphere_l5.zip
	mv data_5_all data
	rm climate_sphere_l5.zip
fi

# create log directory
mkdir -p logs

python train.py \
--batch-size 256 \
--test-batch-size 256 \
--epochs 50 \
--data_folder data \
--max_level 5 \
--min_level 0 \
--feat 8 \
--log_dir log_f8_balance \
--decay \
--lr 1e-2 \
--balance
