#!/usr/bin/env bash

docker run --rm --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3.5-4B \
    --served-model-name qwen3.5-4b \
    --api-key local \
    --max-model-len 16384 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.95 \
    --reasoning-parser qwen3 \
    --language-model-only
