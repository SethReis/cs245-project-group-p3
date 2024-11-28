#!/bin/sh

python3 evaluate.py \
    --dataset_path "example_data/dev_data.jsonl.bz2" \
    --model_name "vanilla_baseline" \
    --llm_name "meta-llama/Llama-3.2-1B-Instruct" \
    --is_server \
    --vllm_server "http://localhost:8088/v1" \
    --max_retries 10
