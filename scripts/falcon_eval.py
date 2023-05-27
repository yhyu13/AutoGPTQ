import datasets
from argparse import ArgumentParser

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.eval_tasks import LanguageModelingTask
from transformers import AutoTokenizer


DATASET = "tatsu-lab/alpaca"
WITH_INPUT_TEMPLATE = "Instruction:\n{instruction}\n\nInput:\n{input}\n\nOutput:\n"
WITHOUT_INPUT_TEMPLATE = "Instruction:\n{instruction}\n\nOutput:\n"


def ds_refactor_fn(samples):
    instruction_data = samples["instruction"]
    input_data = samples["input"]
    output_data = samples["output"]

    new_samples = {"prompt": [], "output": []}
    for instruction_txt, input_txt, output_txt in zip(instruction_data, input_data, output_data):
        if input_txt:
            prompt = WITH_INPUT_TEMPLATE.format(instruction=instruction_txt, input=input_txt)
        else:
            prompt = WITHOUT_INPUT_TEMPLATE.format(instruction=instruction_txt)
        new_samples["prompt"].append(prompt)
        new_samples["output"].append(output_txt)

    return new_samples


def main():
    parser = ArgumentParser()
    parser.add_argument("--quantized_model_dir", type=str)
    parser.add_argument("--num_samples", type=int, default=20, help="how many samples will be sampled to evaluation")
    parser.add_argument("--sample_max_len", type=int, default=1024, help="max tokens for each sample")
    parser.add_argument("--block_max_len", type=int, default=2048, help="max tokens for each data block")
    parser.add_argument("--use_triton", default=False, action="store_true")
    parser.add_argument("--trust_remote_code", type=bool, default=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.quantized_model_dir,use_fast=False)
    model = AutoGPTQForCausalLM.from_quantized(args.quantized_model_dir, use_fast=False, device="cuda:1", use_triton=args.use_triton, trust_remote_code=args.trust_remote_code)
    
    prompt = "Write a story about llamas"
    prompt_template = f"### Instruction: {prompt}\n### Response:"

    tokens = tokenizer(prompt_template, return_tensors="pt").to("cuda:1").input_ids
    output = model.generate(input_ids=tokens, max_new_tokens=100, do_sample=True, temperature=0.8)
    print(tokenizer.decode(output[0]))

    """     
    task = LanguageModelingTask(
        model=model,
        tokenizer=tokenizer,
        data_name_or_path=DATASET,
        prompt_col_name="prompt",
        label_col_name="output",
        **{
            "num_samples": args.num_samples,  # how many samples will be sampled to evaluation
            "sample_max_len": args.sample_max_len,  # max tokens for each sample
            "block_max_len": args.block_max_len,  # max tokens for each data block
            "load_fn": datasets.load_dataset,  # function to load dataset
            "preprocess_fn": ds_refactor_fn,  # function to preprocess dataset
            "truncate_prompt": False  # truncate label when sample's length exceed sample_max_len
        }
    )

    print(f"eval result for quantized model: {task.run()}")
    """

if __name__ == "__main__":
    main()
