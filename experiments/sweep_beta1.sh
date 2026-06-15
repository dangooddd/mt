#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

for beta1 in 0.85 0.9 0.95; do
  run_pretrain "$BASE_MODEL" "b1${beta1}" "$LR" "$beta1" "$BETA2" "$WEIGHT_DECAY" "$DROPOUT"
done
