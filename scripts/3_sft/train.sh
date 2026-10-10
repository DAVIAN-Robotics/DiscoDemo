#!/usr/bin/env bash
# Stage 3: pi0.5 delta-action SFT on a generated dataset, then evaluation of the final checkpoint in Isaac Sim.
#
#   bash scripts/3_sft/train.sh <task> <dataset_dir> <output_dir> [seed=1000]
#
# Paper recipe: from DAVIAN-Robotics/pi05_droid_jointpos, 20k steps, batch 16 on one GPU, chunk 15,
# lr 2.5e-5 -> 2.5e-6 (cosine, 1k warmup), evaluation over 50 episodes with the training seed.
# WANDB=false disables wandb logging.
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK=${1:?task}; DATA=${2:?dataset_dir}; OUT=${3:?output_dir}; SEED=${4:-1000}
STEPS=20000
accelerate launch --num_processes 1 --mixed_precision bf16 -m sft.train \
  --task "$TASK" --dataset.repo_id "local/$TASK" --dataset.root "$DATA" \
  --policy.type pi05 --policy.pretrained_path DAVIAN-Robotics/pi05_droid_jointpos --policy.push_to_hub false \
  --policy.dtype bfloat16 --policy.gradient_checkpointing true \
  --policy.chunk_size 15 --policy.n_action_steps 15 --policy.max_state_dim 8 \
  --policy.optimizer_lr 2.5e-5 --policy.scheduler_warmup_steps 1000 \
  --policy.scheduler_decay_steps "$STEPS" --policy.scheduler_decay_lr 2.5e-6 \
  --steps "$STEPS" --batch_size 16 --num_workers 8 --tolerance_s 0.01 --seed "$SEED" \
  --log_freq 100 --save_freq 1000 --eval_freq 0 \
  --wandb.enable "${WANDB:-true}" --wandb.project DiscoDemo \
  --rename_map '{"observation.images.over_shoulder_left_camera": "observation.images.base_0_rgb", "observation.images.wrist_cam": "observation.images.left_wrist_0_rgb"}' \
  --output_dir "$OUT" --job_name "$(basename "$OUT")"
