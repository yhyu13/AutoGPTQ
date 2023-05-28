#!/bin/bash

export PY_SCRIPT=scripts/falcon_quant.py
export ORIGIN_DIR=/media/hangyu5/Home/Documents/Hugging-Face/falcon-40b-instruct
export OUTPUT_DIR=/media/hangyu5/Home/Documents/Hugging-Face/falcon-40b-instruct-autogptq
rm -rf $OUTPUT_DIR
mkdir $OUTPUT_DIR

python $PY_SCRIPT --pre_trained_dir=$ORIGIN_DIR --quant_dir=$OUTPUT_DIR --group_size=-1