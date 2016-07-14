#!/usr/bin/env sh

./build/tools/caffe \
  train \
  --solver=examples/c3d_ucf101/c3d_ucf101_solver.prototxt \
  --gpu=1,2 \
  2>&1 | tee examples/c3d_ucf101/c3d_ucf101_train.log
