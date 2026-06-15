#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

for lr in 0.003 0.0003 0.00003; do
  run_pretrain "$BASE_MODEL" "lr${lr}" "$lr" "$BETA1" "$BETA2" "$WEIGHT_DECAY" "$DROPOUT"
done
