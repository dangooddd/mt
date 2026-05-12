#!/usr/bin/env bash
set -euo pipefail

NAME=gemma4-31b-vllm
PORT=1973
IMAGE=vllm/vllm-openai:gemma4-0505-cu129
MODEL=google/gemma-4-31B-it
DRAFT_MODEL=google/gemma-4-31B-it-assistant

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
    --name "$NAME" \
    --gpus all \
    --network host \
    --ipc host \
    --shm-size 32g \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}" \
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    "$IMAGE" "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size 2 \
        --max-model-len 256K \
        --max-num-seqs 256 \
        --gpu-memory-utilization 0.95 \
        --async-scheduling \
        --language-model-only \
        --reasoning-parser gemma4 \
        --tool-call-parser gemma4 \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        --speculative-config "{\"method\":\"draft_model\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":6,\"draft_tensor_parallel_size\":1}"

echo "Started: $NAME"
echo "Endpoint: http://127.0.0.1:${PORT}/v1"
echo "Logs: docker logs -f $NAME"
