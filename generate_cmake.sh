#!/bin/bash

mkdir -p build
pushd build

cmake .. \
    -DENOKI_CUDA=1 \
    -DENOKI_AUTODIFF=1 \
    -DENOKI_PYTHON=1 \
    -DPYTHON_EXECUTABLE:FILEPATH=$CONDA_PYTHON_EXE \
    -DPYTHON_LIBRARY:FILEPATH=$CONDA_PREFIX/lib/libpython3.9.so

popd