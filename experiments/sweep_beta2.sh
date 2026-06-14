#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

for base_model in "${BASE_MODELS[@]}"; do
  for beta2 in 0.95 0.98 0.999; do
    run_pretrain "$base_model" "b2${beta2}" "$LR" "$BETA1" "$beta2" "$WEIGHT_DECAY" "$DROPOUT"
  done
done
