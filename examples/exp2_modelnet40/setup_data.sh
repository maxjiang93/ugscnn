#!/bin/bash
# make data directories
# mkdir -p data
# mkdir -p modelnet10_test
# mkdir -p modelnet10_train
# mkdir -p modelnet40_test
# mkdir -p modelnet40_train
# download modelnet data
# wget -P data/ http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
# wget -P data/ https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar

# setup modelnet10
unzip data/ModelNet10.zip -d data/modelnet10
find data/modelnet10/ModelNet10 -mindepth 3 | grep off | grep train | xargs mv -t data/modelnet10_train
find data/modelnet10/ModelNet10 -mindepth 3 | grep off | grep test | xargs mv -t data/modelnet10_test
rm -rf data/modelnet10
rm data/ModelNet10.zip

# setup modelnet40
mkdir -p data/modelnet40
tar -xvf data/modelnet40_manually_aligned.tar -C data/modelnet40
find data/modelnet40 -mindepth 3 | grep off | grep train | xargs mv -t data/modelnet40_train
find data/modelnet40 -mindepth 3 | grep off | grep test | xargs mv -t data/modelnet40_test
rm -rf data/modelnet40
rm data/modelnet40_manually_aligned.tar