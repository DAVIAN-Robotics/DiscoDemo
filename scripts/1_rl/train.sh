#!/usr/bin/env bash
# Stage 1: train an RL demonstration generator (RFCL + FlashSAC, plus the METRA skill objective for DiscoDemo).
#
#   bash scripts/1_rl/train.sh <task> <discodemo|prfcl> [hydra overrides ...]
#
# Config: libs/FlashSAC/configs/fr3.yaml + task/<task>.yaml + method/<method>.yaml.
# Human demos: data/demo_banks/<task>/human_demos.h5. Checkpoints go to libs/FlashSAC/models/<task>/<method>-<task>/;
# resume with agent_load_path=<step dir>.
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK=${1:?task}; METHOD=${2:?method (discodemo|prfcl)}; shift 2
args=(--overrides "task=$TASK" --overrides "method=$METHOD")
for o in "$@"; do args+=(--overrides "$o"); done
exec python libs/FlashSAC/train_rfcl_robolab.py "${args[@]}"
