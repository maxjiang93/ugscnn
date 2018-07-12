#!/bin/bash

MESHFILES=/global/homes/m/maxjiang/codes/fecnn/meshcnn/mesh_files
# assert mesh files exist
if [ ! -d $MESHFILES ]; then
    echo "[!] Mesh files do not exist..."
    exit
fi

# generate data
if [ ! -f mnist_ico3.gzip ]; then
    python prepare_data.py --mnist_data_folder raw_data --output_file mnist_ico3.gzip --no_rotate_train --no_rotate_test
fi

# train
python train.py --mesh_folder $MESHFILES --no-cuda
