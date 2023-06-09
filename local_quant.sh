#!/bin/bash

export PY_SCRIPT=scripts/falcon_quant.py
export ORIGIN_DIR=/media/hangyu5/Home/Documents/Hugging-Face/gorilla-falcon-7b-hf-v0
export OUTPUT_DIR=/media/hangyu5/Home/Documents/Hugging-Face/gorilla-falcon-7b-hf-v0-autogptq
rm -rf $OUTPUT_DIR
mkdir $OUTPUT_DIR

python $PY_SCRIPT --pre_trained_dir=$ORIGIN_DIR --quant_dir=$OUTPUT_DIR --group_size=128