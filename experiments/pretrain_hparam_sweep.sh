#!/usr/bin/env bash
set -euo pipefail

# Tokenizer comparison: unigram vs bpe vs wordpiece.
"$(dirname "$0")/sweep_tokenizer.sh"

# One-dimensional hyperparameter sweeps on unigram only.
"$(dirname "$0")/sweep_lr.sh"
"$(dirname "$0")/sweep_beta1.sh"
"$(dirname "$0")/sweep_beta2.sh"
"$(dirname "$0")/sweep_weight_decay.sh"
"$(dirname "$0")/sweep_dropout.sh"
