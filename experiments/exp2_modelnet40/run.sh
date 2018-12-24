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
    # make data directories
    mkdir -p data
    mkdir -p data/modelnet40_test
    mkdir -p data/modelnet40_train
    # download modelnet data
    wget -P data/ https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar

    # setup modelnet40
    mkdir -p data/modelnet40
    tar -xvf data/modelnet40_manually_aligned.tar -C data/modelnet40
    find data/modelnet40 -mindepth 3 | grep off | grep train | xargs mv -t data/modelnet40_train
    find data/modelnet40 -mindepth 3 | grep off | grep test | xargs mv -t data/modelnet40_test
    rm -rf data/modelnet40
    rm data/modelnet40_manually_aligned.tar
    find data/modelnet40_train -mindepth 1 | grep annot | xargs rm
    find data/modelnet40_test -mindepth 1 | grep annot | xargs rm
fi

# train
python train.py \
--log_dir logs/modelnet40_ft16_b16 \
--model_path model.py \
--partition train \
--dataset modelnet40 \
--batch_size 16 \
--feat 16 \
--num_workers 12 \
--learning_rate 5e-3 \
--epochs 300 \
--sp_mesh_dir ../../mesh_files \
--sp_mesh_level 5 \
--data_dir $DATADIR
