#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

for base_model in "${BASE_MODELS[@]}"; do
  for wd in 0.1 0.01 0.005; do
    run_pretrain "$base_model" "wd${wd}" "$LR" "$BETA1" "$BETA2" "$wd" "$DROPOUT"
  done
done
