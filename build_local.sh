#!/bin/bash

# fill in the cuda arch list
export TORCH_CUDA_ARCH_LIST="8.6"
export BUILD_CUDA_EXT=1 

python3 setup.py install