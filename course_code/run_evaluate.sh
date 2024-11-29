#!/bin/sh

python3 evaluate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --model_name "betterrag" \
    --llm_name "meta-llama/Llama-3.2-3B-Instruct" \
    --is_server \
    --vllm_server "http://localhost:8088/v1" \
    --max_retries 10
