#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

for dropout in 0.05 0.1 0.2; do
  run_pretrain "$BASE_MODEL" "do${dropout}" "$LR" "$BETA1" "$BETA2" "$WEIGHT_DECAY" "$dropout"
done
