#!/usr/bin/env bash
# Stage 2: roll out a trained generator, filter the successes, render observations, export a LeRobot v3.0 dataset.
#
#   bash scripts/2_gendata/generate.sh <task> <ckpt_dir> <config_yaml> <out_dir> [n_keep=3000] [n_shards=4]
#
#   ckpt_dir     generator checkpoint step dir (actor.pt + normalizer.pt), e.g. a Stage1_RL repo from the hub
#   config_yaml  the generator's config.yaml
#
# Steps: collect successful rollouts (camera-free states + contact forces) -> safety signals -> export filter and
# n_keep trajectories from a fixed permutation -> render the kept trajectories in shards -> convert -> merge ->
# q01/q99 stats -> constant-frame-rate videos.
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK=${1:?task}; CKPT=${2:?ckpt_dir}; CFG=${3:?config_yaml}; OUT=${4:?out_dir}
N_KEEP=${5:-3000}; N_SHARDS=${6:-4}
NUM_ENVS=${NUM_ENVS:-2048}       # collection envs (no rendering)
RENDER_ENVS=${RENDER_ENVS:-32}   # trajectories rendered in parallel

# Successes to collect before filtering, as for the paper datasets (the export filter keeps about 54% / 49% / 24% /
# 79% of the DiscoDemo rollouts). Resumable with --resume-from-existing.
case "$TASK" in
  pnp_banana|stack_cube|fmb_sqcircle) POOL=10000 ;; fmb_round) POOL=32000 ;;
  *) echo "unknown task: $TASK (pnp_banana | stack_cube | fmb_round | fmb_sqcircle)" >&2; exit 1 ;;
esac
python libs/RoboLab/scripts/collect_rl_states.py --ckpt-dir "$CKPT" --config-yaml "$CFG" \
  --num-envs "$NUM_ENVS" --target-k "${TARGET_K:-$POOL}" --out-dir "$OUT/states" --resume-from-existing
python libs/RoboLab/scripts/score_collision.py score --states-dir "$OUT/states" --out "$OUT/scores.json"
python libs/RoboLab/scripts/score_collision.py select --scores "$OUT/scores.json" \
  --n-keep "$N_KEEP" --seed 0 --out "$OUT/keep.json"
python libs/RoboLab/scripts/repartition_states_shards.py --src-dir "$OUT/states" --out-dir "$OUT/shards" \
  --n-shards "$N_SHARDS" --keep-json "$OUT/keep.json"

shards=()
for ((i = 0; i < N_SHARDS; i++)); do
  python libs/RoboLab/scripts/render_states_to_lerobot.py --task "$TASK" \
    --states-dir "$OUT/shards/shard_$i" --out-dir "$OUT/render/r$i" --num-envs "$RENDER_ENVS" --resume
  python libs/RoboLab/scripts/convert_raw_to_lerobot.py "$OUT/render/r$i" "$OUT/lerobot/r$i"
  shards+=("$OUT/lerobot/r$i")
done
python libs/RoboLab/scripts/aggregate_shards.py --out "$OUT/dataset" "${shards[@]}"
python libs/RoboLab/scripts/add_quantile_stats.py --root "$OUT/dataset"
python libs/RoboLab/scripts/normalize_videos_cfr.py "$OUT/dataset"
echo "dataset: $OUT/dataset"
