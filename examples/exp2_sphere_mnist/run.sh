#!/bin/bash

MESHFILES=../../mesh_files

# assert mesh files exist
if [ ! -d $MESHFILES ]; then
    echo "[!] Mesh files do not exist..."
    exit
fi

# generate data
if [ ! -f mnist_ico4.gzip ]; then
    echo "[!] Data files do not exist. Preparing data..."
    python prepare_data.py --bandwidth 60 --mnist_data_folder raw_data --output_file mnist_ico4.gzip --no_rotate_train --no_rotate_test --mesh_file $MESHFILES/icosphere_4.pkl
fi

# train
python train.py --mesh_folder $MESHFILES --datafile mnist_ico4.gzip
