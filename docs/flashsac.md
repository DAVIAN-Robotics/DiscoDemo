# FlashSAC (`libs/FlashSAC`)

Upstream: FlashSAC by Holiday Robotics (MIT License, `libs/FlashSAC/LICENSE`). We keep the SAC agent (networks,
distributional critic, schedulers, reward normalization, replay buffer) and drop the multi-backend environment layer.
Everything for DiscoDemo is added around it.

## Changed upstream files

- `flash_rl/agents/flashSAC/agent.py`
  - Observations pass through a bounds normalizer (`agents/utils/normalizer.py`) before the actor and the buffer;
    the bounds come from the task's workcell window (`flash_rl/envs/robolab_obs_bounds.py`) and are saved as
    `normalizer.pt`.
  - METRA skill objective: `set_skill_modules` attaches the skill representation, the buffer stores an episode id
    per transition, and the intrinsic reward is added in `update()` after reward normalization; its state is saved as
    `skill_state.pt`.
  - Learning-rate schedules advance with environment steps (`update(env_step)`).
  - `load_actor` loads only the actor and the normalizer (inference); `load` restores everything for resuming.
- `flash_rl/agents/flashSAC/update.py`: the critic update also returns the fraction of TD targets outside the value
  support (logged), and the actor update logs Q statistics.
- `flash_rl/agents/utils/network.py`: `Network.load` accepts checkpoints saved with or without `torch.compile`.
- `flash_rl/agents/utils/reward_normalization.py`: `reset_running_returns()` for a change of the number of envs.
- `flash_rl/buffers/torch_buffer.py`
  - Storage is allocated in blocks of 2^24 rows as the buffer fills, so GPU memory and the saved buffer grow with the
    stored transitions (about 0.4 KB each for the FR3 tasks) instead of the configured capacity. The paper capacity
    (100M) is never reached by a 50M-step run.
  - Optional per-transition episode id (METRA).
- `flash_rl/buffers/__init__.py`: the numpy buffer and the buffer factory are removed.
- `flash_rl/common/logger.py`: wandb runs are named after `exp_name` and can be resumed by id; evaluation results
  are logged on their own step axis (`eval/env_step`) because they arrive after training has moved on.
- `flash_rl/envs/__init__.py`: the environment backends (DMC, MuJoCo, HumanoidBench, MetaWorld, MyoSuite, D4RL,
  Isaac Lab, ManiSkill, Genesis, ...) are removed; the FR3 environment is built by `flash_rl/envs/robolab.py`.
- `flash_rl/types.py`: no jax dependency.
- Docstrings and type annotations only: `flash_rl/__init__.py`, `agents/__init__.py`, `agents/base_agent.py`,
  `agents/flashSAC/__init__.py`, `agents/flashSAC/layer.py`, `agents/flashSAC/network.py`, `agents/utils/__init__.py`,
  `agents/utils/distribution.py`, `agents/utils/scheduler.py`, `buffers/base_buffer.py`, `common/__init__.py`.
- `pyproject.toml`: packages `flash_rl` only; dependencies are in the root `pyproject.toml`.

Removed upstream content: the other entry points, configs and launch scripts, the environment backends and wrappers,
the numpy buffer, the random agent, docs and CI.

## Added (DiscoDemo)

- `train_rfcl_robolab.py`: training entry point (reverse curriculum, then forward phase; evaluation in a subprocess;
  resume).
- `configs/fr3.yaml`, `configs/task/<task>.yaml`, `configs/method/{discodemo,prfcl}.yaml`.
- `flash_rl/rfcl/`: reverse curriculum (a PyTorch reimplementation of RFCL, Tao et al., ICLR 2024), demonstration
  buffer, export-filter signals, evaluation worker.
- `flash_rl/agents/utils/metra.py`: METRA skill objective (Park et al., ICLR 2024).
- `flash_rl/envs/robolab.py`, `robolab_obs_bounds.py`, `skill_z.py`: the FR3 vector environment, the observation window
  and the skill latent.

## Action space

The relative arm action moves each joint by at most `max_joint_velocity * 0.05 s` per control step (20 Hz), smoothed
by the velocity / acceleration / jerk limits in `jerk_limited_action` of the task config.
