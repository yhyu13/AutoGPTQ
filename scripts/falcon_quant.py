import os
import numpy as np
import random
import torch
from transformers import AutoTokenizer, TextGenerationPipeline
from datasets import load_dataset
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

parser = argparse.ArgumentParser(description='Quantize a pre-trained GPT model')
parser.add_argument('--pre_trained_dir', type=str, help='Directory of the pre-trained model')
parser.add_argument('--quant_dir', type=str, help='Directory to save the quantized model')

args = parser.parse_args()

pretrained_model_dir = args.pre_trained_dir
quantized_model_dir = args.quant_dir

device_map = {
    0: "20GiB",
    1: "20GiB",
    "cpu": "200GiB"
}

torch.set_num_threads(40)

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    # load dataset and preprocess 
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
    return traindataset, testenc

def main():
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    
    # load un-quantized model, the model will always be force loaded into cpu
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=64,  # it is recommended to set the value to 128
        desc_act=False,  # desc_act and groupsize only works on triton
    )
    
    # get model maximum sequence length
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device_map, low_cpu_mem_usage=True)
    model_config = model.config.to_dict()
    seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
    if any([k in model_config for k in seq_len_keys]):
        for key in seq_len_keys:
            if key in model_config:
                model.seqlen = model_config[key]
                break
    else:
        print("set model.seqlen = 2048 by default")
        model.seqlen = 2048
     
    # load train dataset for quantize
    traindataset, testenc = get_wikitext2(128, 0, model.seqlen, tokenizer)

    # quantize model, the examples should be list of dict whose keys contains "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(traindataset, use_triton=False)

    # save quantized model
    model.save_quantized(quantized_model_dir, use_safetensors=True)
    
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

main()