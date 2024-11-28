#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

python3 generate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --split 1 \
    --model_name "vanilla_baseline" \
    --llm_name "meta-llama/Llama-3.2-1B-Instruct" \
    --is_server \
    --vllm_server "http://localhost:8088/v1"
