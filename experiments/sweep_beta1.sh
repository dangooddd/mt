#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

for base_model in "${BASE_MODELS[@]}"; do
  for beta1 in 0.85 0.9 0.95; do
    run_pretrain "$base_model" "b1${beta1}" "$LR" "$beta1" "$BETA2" "$WEIGHT_DECAY" "$DROPOUT"
  done
done
