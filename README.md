# DiscoDemo

**DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning**

[Project page](https://davian-robotics.github.io/DiscoDemo/) ·
[Models and datasets](https://huggingface.co/collections/DAVIAN-Robotics/discodemo-6ac696e70d028fddf0c3dfeb) ·
Paper (coming soon)

DiscoDemo trains a reinforcement-learning demonstration generator from a small set of human demonstrations,
samples diverse task-solving behaviors from it, filters them with simulator safety signals, and uses the
resulting dataset to fine-tune a vision-language-action policy. This repository contains the full pipeline
on a real-to-sim Franka FR3 workcell in Isaac Sim, for four tasks:

| Task | Description |
|---|---|
| `pnp_banana` | pick up a banana and place it in a bowl |
| `stack_cube` | stack a red cube on a blue cube |
| `fmb_round` | insert a round peg into an FMB board |
| `fmb_sqcircle` | insert a square-circle peg into an FMB board |

## Pipeline

| Stage | What it does | Entry point |
|---|---|---|
| 0. Real-to-sim | FR3 workcell (calibrated cameras, gripper delay, system identification), task scenes and success predicates, human teleoperation demonstrations replayed in simulation | `libs/real2sim`, `data/demo_banks` |
| 1. RL generator | reverse-curriculum RL (RFCL) with SAC from 20 demonstrations, plus a METRA skill objective (DiscoDemo) and a contact / jamming penalty | `scripts/1_rl/train.sh` |
| 2. Data generation | roll out the generator with a random skill per episode, keep successful rollouts that pass the export filter, render camera observations, export LeRobot v3.0 | `scripts/2_gendata/generate.sh` |
| 3. SFT | delta-action fine-tuning of pi0.5 (`DAVIAN-Robotics/pi05_droid_jointpos`) and evaluation in simulation | `scripts/3_sft/train.sh`, `sft/sim_eval.py` |
| Analysis | trajectory diversity and dataset coverage (Vendi score), generation yield, 2D end-effector trajectory overlays | `scripts/analysis`, `libs/real2sim/scripts` |

The export filter removes a successful rollout if the illegal contact force exceeds 50 N in one step or 20 N for
three consecutive steps, if the peg jams with more than 15 N (insertion tasks), or if the container (bowl, lower
cube or board) moved by 1 cm or more; 3,000 episodes are then drawn with a fixed permutation. On the insertion tasks, success also requires the board to stay within 1 cm and 5 degrees
of its initial pose.

## Installation

Requirements: Linux, an NVIDIA RTX-capable GPU, Python 3.11, [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/DAVIAN-Robotics/DiscoDemo.git
cd DiscoDemo
uv sync                      # Isaac Sim 5.0, Isaac Lab 2.2, PyTorch 2.9 (cu128), the local packages
uv sync --extra analysis     # optional: forward kinematics for the coverage analysis
source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=Y
```

All commands below are run from the repository root on a single GPU. Training logs go to Weights & Biases
(project `DiscoDemo`); pass `logger_type=tensorboard` (RL, logs under `runs/`) or `WANDB=false` (SFT) to turn it off.

## Usage

### 1. Train an RL generator

```bash
bash scripts/1_rl/train.sh pnp_banana discodemo     # DiscoDemo
bash scripts/1_rl/train.sh pnp_banana prfcl         # P-RFCL baseline (no skill objective)
```

The configuration is `libs/FlashSAC/configs/fr3.yaml` (shared hyperparameters) composed with `task/<task>.yaml`
and `method/<method>.yaml`. Extra arguments are Hydra overrides, e.g. `seed=1`, or
`agent_load_path=<checkpoint step dir>` to resume (the replay buffer and the curriculum are restored too).
Checkpoints are written to `libs/FlashSAC/models/`.

### 2. Generate a dataset

```bash
bash scripts/2_gendata/generate.sh pnp_banana <ckpt_dir> <config.yaml> out/pnp_banana_discodemo
```

`<ckpt_dir>` is a generator checkpoint (`actor.pt` and `normalizer.pt`) and `<config.yaml>` its training
config, either from step 1 or from a `DiscoDemo-Stage1_RL-*` model on the hub. The script collects
successful rollouts, scores them, applies the export filter, renders the selected 3,000 episodes and writes a
LeRobot dataset to `out/pnp_banana_discodemo/dataset`. Collection and rendering resume where they stopped.

### 3. Fine-tune and evaluate pi0.5

```bash
bash scripts/3_sft/train.sh pnp_banana out/pnp_banana_discodemo/dataset out/sft_pnp_banana
python -m sft.sim_eval <checkpoint> --task pnp_banana --seed 1000 --out-dir out/eval_pnp_banana
```

Training runs 20k steps with batch size 16 and then evaluates the final checkpoint on 50 episodes (success
rate and mean time to success). `sft.sim_eval` evaluates any checkpoint, including the
`DiscoDemo-Stage3_SFT-*` models on the hub; `--episode-videos` also writes one video per episode.

### Analysis

- **Diversity and coverage.** `scripts/analysis/diversity_vendi.py` (fixed-scene trajectory diversity) and
  `scripts/analysis/coverage_vendi.py` (dataset coverage) embed object and end-effector pose trajectories
  and report Vendi scores; `scripts/analysis/yield_gen.py` reports generation yield. Inputs are given as a
  JSON spec; see the docstring of each script.
- **2D end-effector overlays.** `libs/real2sim/scripts/rollout_fixed_init.py` rolls out a generator from the
  fixed scenes of the paper figures (`libs/real2sim/assets/fixed_scenes/<task>.json`); with `--export-view` it
  renders the scene background and camera instead. `libs/real2sim/scripts/render_eef_2d.py` draws the
  trajectories on that background.

## Tests

```bash
uv run pytest       # CPU tests of the curriculum, export filter, action space, SFT labels and metrics
```

## Repository layout

```
libs/FlashSAC   SAC agent, RFCL reverse curriculum, METRA skills, RL entry point and configs
libs/RoboLab    Isaac Lab environment layer (FR3 robot, RL registration, initial poses, contact signals),
                data-generation scripts, robot and object assets
libs/real2sim   FR3 workcell, cameras, task definitions and scenes
libs/fmb        FMB board and pegs, peg-in-hole predicate
libs/lerobot    LeRobot (modified: pi0.5 image padding, bounded video-decoder cache)
sft             pi0.5 fine-tuning, inference wrapper and simulation evaluation
data/demo_banks human demonstrations replayed in simulation (20 per task)
scripts         stage scripts and analysis
docs            changes to the bundled upstream code
```

## License

The code in this repository is released under the Apache License 2.0. Bundled third-party code and assets
keep their own licenses; see [NOTICE](NOTICE).

## Citation

```bibtex
@article{park2026discodemo,
  title   = {DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning},
  author  = {Park, Minho and Kim, Kinam and Kim, Donghu and Lee, Byungkun and Hwang, Dongyoon and Shin, Yongjae and Hyung, Junha and Lee, Hojoon and Choo, Jaegul},
  journal = {arXiv preprint},
  year    = {2026}
}
```
