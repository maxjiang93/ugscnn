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
    bash setup_data.sh
fi

# train
python train.py \
--log_dir logs/modelnet40_drop_ft64_b16_ty \
--model_path model.py \
--partition train \
--dataset modelnet40 \
--batch_size 16 \
--feat 64 \
--num_workers 12 \
--learning_rate 5e-3 \
--epochs 300 \
--sp_mesh_dir ../../mesh_files \
--sp_mesh_level 5 \
--data_dir $DATADIR
