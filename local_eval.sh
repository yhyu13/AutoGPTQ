#!/bin/bash

export PY_SCRIPT=scripts/falcon_eval.py
export OUTPUT_DIR=/media/hangyu5/Home/Documents/Hugging-Face/falcon-7b-autogptq

python $PY_SCRIPT --quantized_model_dir=$OUTPUT_DIR