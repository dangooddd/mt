#!/usr/bin/env bash
set -euo pipefail

# Runs separate one-dimensional sweeps with default values for other hyperparameters.
"$(dirname "$0")/sweep_lr.sh"
"$(dirname "$0")/sweep_beta1.sh"
"$(dirname "$0")/sweep_beta2.sh"
"$(dirname "$0")/sweep_weight_decay.sh"
"$(dirname "$0")/sweep_dropout.sh"
