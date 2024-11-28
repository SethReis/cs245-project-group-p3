#!/bin/sh

vllm serve \
    meta-llama/Llama-3.2-1B-Instruct \
    --gpu_memory_utilization=0.85 \
    --tensor_parallel_size=1 \
    --dtype="half" \
    --port=8088 \
    --enforce_eager \
    --max_model_len=4096
