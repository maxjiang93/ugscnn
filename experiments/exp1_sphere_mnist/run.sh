#!/bin/bash
source activate
MESHFILES=../../mesh_files
DATAFILE=mnist_ico4.zip

# assert mesh files exist
if [ ! -d $MESHFILES ]; then
    echo "[!] Mesh files do not exist..."
    exit
fi

# generate data
if [ ! -f $DATAFILE ]; then
    echo "[!] Data files do not exist. Preparing data..."
    python prepare_data.py --bandwidth 60 --mnist_data_folder raw_data --output_file $DATAFILE --no_rotate_train --no_rotate_test --mesh_file $MESHFILES/icosphere_4.pkl
fi

# train
python train.py --mesh_folder $MESHFILES --datafile $DATAFILE --log_dir log_res_rot_adam_ft16_b64_decay --optim adam --lr 1e-2 --epochs 100 --feat 16 --decay --batch-size 64 #--dropout 
