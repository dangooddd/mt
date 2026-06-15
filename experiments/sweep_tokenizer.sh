#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

for base_model in "${TOKENIZER_MODELS[@]}"; do
  run_pretrain "$base_model" "tokenizer" "$LR" "$BETA1" "$BETA2" "$WEIGHT_DECAY" "$DROPOUT"
done
