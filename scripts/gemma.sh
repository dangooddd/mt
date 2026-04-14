#!/usr/bin/env bash

vllm serve data/hf/gemma-4-31B-it \
    --served-model-name gemma-4-31B-it \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --language-model-only \
    --max-model-len 4096 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --api-key "local"
