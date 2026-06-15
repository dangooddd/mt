#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

for wd in 0.1 0.01 0.005; do
  run_pretrain "$BASE_MODEL" "wd${wd}" "$LR" "$BETA1" "$BETA2" "$wd" "$DROPOUT"
done
