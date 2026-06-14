#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH=${DATASET_PATH:-data/datasets/opus-yandex-paracrawl-un-flores}
BASE_MODELS=(
  transformer-complete-small-unigram
  transformer-complete-small-bpe
  transformer-complete-small-wordpiece
)

BATCH_SIZE=${BATCH_SIZE:-100}
# train rows: 7,270,832; batch 100 => ceil(real epoch) = 72,709 steps.
# Use 10 virtual epochs for metrics: 7,271 * 10 = 72,710 steps,
# i.e. one full dataset pass plus one extra batch.
EPOCH_STEPS=${EPOCH_STEPS:-7271}
STEPS=${STEPS:-72710}
MAX_LENGTH=${MAX_LENGTH:-384}

LR=${LR:-0.0003}
BETA1=${BETA1:-0.9}
BETA2=${BETA2:-0.999}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
DROPOUT=${DROPOUT:-0.1}
MIN_LR=${MIN_LR:-0.00001}

run_pretrain() {
  local base_model=$1
  local suffix=$2
  local lr=$3
  local beta1=$4
  local beta2=$5
  local wd=$6
  local dropout=$7

  local experiment="${base_model}-${suffix}"
  local model_dir="data/models/experiments/${experiment}"
  mkdir -p "$model_dir"

  python experiments/render_model_config.py \
    --base-model-dir "data/models/${base_model}" \
    --output-dir "$model_dir" \
    --dropout "$dropout"

  uv run python -m mt.models.train.pretrain \
    --model-dir "$model_dir" \
    --dataset-path "$DATASET_PATH" \
    --experiment "$experiment" \
    --batch-size "$BATCH_SIZE" \
    --epoch-steps "$EPOCH_STEPS" \
    --steps "$STEPS" \
    --max-lr "$lr" \
    --min-lr "$MIN_LR" \
    --adam-beta1 "$beta1" \
    --adam-beta2 "$beta2" \
    --weight-decay "$wd" \
    --max-length "$MAX_LENGTH"
}
