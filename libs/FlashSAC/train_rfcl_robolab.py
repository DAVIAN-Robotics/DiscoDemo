"""DiscoDemo generator training: RFCL reverse curriculum + FlashSAC on the FR3 RoboLab tasks.

1. Reverse phase: every episode starts from a demonstration state (in-sim teleport through
   ``ReverseCurriculumIsaacEnv``); each demonstration's start frontier moves back as the policy
   masters it.
2. Forward phase: once ``rfcl.solved_frac_threshold`` of the demonstrations are solved, the reverse
   env is closed and a plain env with randomized initial states is built in the same process. The
   demonstration transitions stay mixed into every batch.

With ``skill.alpha > 0`` (DiscoDemo) every episode carries a skill z and successful episodes earn
the METRA intrinsic reward; ``skill.alpha = 0`` is the P-RFCL baseline.

Evaluation runs in a subprocess (``flash_rl/rfcl/eval_worker_robolab.py``) that loads each
checkpoint. Isaac's AppLauncher starts inside the first ``make_robolab_rl_env`` call, so modules
that import ``isaaclab``/``robolab`` at top level are imported after it.

    python libs/FlashSAC/train_rfcl_robolab.py --overrides task=pnp_banana --overrides method=discodemo
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.agents.utils.metra import MetraRepresentation, SkillGateTable
from flash_rl.common import create_logger
from flash_rl.envs.robolab import env_kwargs_from_cfg, make_robolab_rl_env
from flash_rl.envs.robolab_obs_bounds import (
    ObsBoundsSpec,
    arm_joint_limits_from_env,
    demo_outside_fraction,
    format_bounds_table,
    resolve_obs_bounds,
    term_layout_from_env,
)
from flash_rl.envs.skill_z import SkillZConcatWrapper, augment_demo_transitions_with_z
from flash_rl.rfcl.checkpoint_state import restore_reverse_curriculum_state, serialize_reverse_curriculum_state
from flash_rl.rfcl.demo_buffer import DemoBuffer
from flash_rl.rfcl.eval_request import eval_harvest_deadline_s, eval_result_path, write_eval_request
from flash_rl.rfcl.mixed_buffer import MixedReplayBuffer
from flash_rl.rfcl.reverse_curriculum_isaac import ReverseCurriculumIsaacEnv

# libs/FlashSAC/train_rfcl_robolab.py -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_path(path: str) -> str:
    """Absolute path; relative paths are taken from the repository root."""
    p = Path(os.path.expanduser(path))
    return str(p if p.is_absolute() else (_REPO_ROOT / p).resolve())


def _phi_dims(env: Any) -> list[int]:
    """Observation indices fed to the METRA representation: every policy term except ``prev_action``."""
    om = env.unwrapped.observation_manager
    names = list(om.active_terms["policy"])
    dims = [int(np.prod(d)) for d in om.group_obs_term_dim["policy"]]
    assert names[-1] == "prev_action", names
    return list(range(sum(dims[:-1])))


def _save_checkpoint(
    agent: Any, train_env: Any, ckpt_dir: str, interaction_step: int, env_step: int, run_id: str, forward_active: bool
) -> str:
    """Save the agent, replay buffer and ``meta.json`` (steps, phase, logger run and curriculum state)."""
    agent.save(ckpt_dir)
    agent.save_replay_buffer(ckpt_dir)
    meta: dict[str, Any] = {
        "interaction_step": interaction_step,
        "env_step": env_step,
        "forward_active": forward_active,
        "run_id": run_id,
    }
    if not forward_active:
        meta.update(serialize_reverse_curriculum_state(train_env))
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] checkpoint saved -> {ckpt_dir}", flush=True)
    return ckpt_dir


def _prune_replay_buffers(save_dir: str, keep_dir: str) -> None:
    """Delete every ``step<N>/replay_buffer.pt`` except ``keep_dir``'s; ``*_reverse_end`` snapshots stay."""
    keep = os.path.join(keep_dir, "replay_buffer.pt")
    for d in os.listdir(save_dir):
        p = os.path.join(save_dir, d, "replay_buffer.pt")
        if d.startswith("step") and not d.endswith("_reverse_end") and p != keep and os.path.exists(p):
            os.remove(p)


class _EvalWorker:
    """Eval subprocess: request checkpoints, collect results and log them on the eval axis."""

    def __init__(self, save_dir: str, cfg: Any):
        self.ipc = os.path.join(save_dir, "eval_ipc")
        os.makedirs(self.ipc)
        self.log_path = os.path.join(save_dir, "eval_worker.log")
        self._log = open(self.log_path, "w")  # noqa: SIM115  # held by the worker until finish()
        self.proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).parent / "flash_rl" / "rfcl" / "eval_worker_robolab.py"),
                "--config-yaml",
                os.path.join(save_dir, "config.yaml"),
                "--ipc-dir",
                self.ipc,
                "--num-envs",
                str(int(cfg.eval_worker.num_envs)),
                "--seed",
                str(int(cfg.seed) + 1000),
                "--parent-pid",
                str(os.getpid()),
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            cwd=str(_REPO_ROOT),
        )
        print(f"[train] eval worker pid={self.proc.pid} (log: {self.log_path})", flush=True)

    def request(self, env_step: int, ckpt_dir: str) -> None:
        """Ask for an evaluation of ``ckpt_dir``; raises if the worker has died."""
        if self.proc.poll() is not None:
            raise RuntimeError(f"eval worker died (exit={self.proc.returncode}); log: {self.log_path}")
        write_eval_request(self.ipc, env_step, ckpt_dir)

    def drain(self, logger: Any) -> None:
        """Log every finished result, oldest first."""
        files = sorted(
            (f for f in os.listdir(self.ipc) if f.startswith("result_") and f.endswith(".json")),
            key=lambda f: int(f.split("_")[1].split(".")[0]),
        )
        for name in files:
            path = os.path.join(self.ipc, name)
            with open(path) as f:
                res = json.load(f)
            os.remove(path)
            metrics = {"eval/success_rate": float(res["success_rate"])}
            if res["time_to_success_s"] is not None:
                metrics["eval/time_to_success_s"] = float(res["time_to_success_s"])
            logger.log_eval(metrics, int(res["env_step"]), res["video"])
            print(f"[train] eval env_step={res['env_step']}: {metrics}", flush=True)

    def finish(self, logger: Any, env_step: int, max_episode_steps: int) -> None:
        """Wait for the result of ``env_step``, log the remaining results and stop the worker."""
        deadline = time.time() + eval_harvest_deadline_s(max_episode_steps)
        while not os.path.exists(eval_result_path(self.ipc, env_step)):
            if self.proc.poll() is not None:
                raise RuntimeError(f"eval worker died (exit={self.proc.returncode}); log: {self.log_path}")
            if time.time() >= deadline:
                print(f"[train] WARNING: final eval did not arrive in time (log: {self.log_path})", flush=True)
                break
            time.sleep(5.0)
        self.drain(logger)
        open(os.path.join(self.ipc, "stop"), "w").close()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._log.close()


def run(args: argparse.Namespace) -> None:
    """Full training run: config -> demos -> agent -> reverse phase -> forward phase."""
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)
    # The eval worker reads the saved config from another process, so paths become absolute.
    cfg.rfcl.demo_path = _repo_path(str(cfg.rfcl.demo_path))
    cfg.env.task = _repo_path(str(cfg.env.task))
    cfg.env.workcell = _repo_path(str(cfg.env.workcell))
    resume_meta: dict[str, Any] | None = None
    if cfg.agent_load_path is not None:
        cfg.agent_load_path = _repo_path(str(cfg.agent_load_path))
        with open(os.path.join(cfg.agent_load_path, "meta.json")) as f:
            resume_meta = json.load(f)
    # A resumed run reseeds from where it stopped, so it does not replay the initial states and
    # skill z sequence it already used.
    seed = int(cfg.seed) + (0 if resume_meta is None else int(resume_meta["interaction_step"]))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = "cuda:0"

    num_envs = int(cfg.num_train_envs)
    z_dim = int(cfg.skill.z_dim)
    alpha = float(cfg.skill.alpha)
    assert alpha == 0.0 or z_dim > 0, "skill.alpha > 0 needs skill.z_dim > 0"
    env_kwargs = env_kwargs_from_cfg(cfg)
    safety_penalty = OmegaConf.to_container(cfg.safety_penalty, resolve=True)
    rev = cfg.rfcl.reverse
    # The episode budget must leave room to hold success after starting at a demo's last frame.
    assert int(rev.minimum_episode_steps) > int(cfg.success_hold_steps), "minimum_episode_steps too small"
    core_kwargs = {
        "reverse_step_size": int(rev.reverse_step_size),
        "advance_threshold": float(rev.advance_threshold),
        "frontier_window": int(rev.frontier_window),
        "minimum_episode_steps": int(rev.minimum_episode_steps),
    }

    ###############################
    # reverse env (Isaac AppLauncher starts here) and demonstrations
    ###############################
    inner = make_robolab_rl_env(
        num_envs, seed, to_numpy=False, curriculum=True, safety_penalty=safety_penalty, **env_kwargs
    )
    rate = torch.tensor(inner.max_relative_action_rate, dtype=torch.float32)
    max_episode_steps = int(inner.max_episode_steps)

    from flash_rl.rfcl.demo_dataset_robolab import extract_demo_transitions_robolab, load_demo_states_robolab

    demos = load_demo_states_robolab(cfg.rfcl.demo_path, int(cfg.rfcl.truncate_after_success))
    print(f"[train] {len(demos)} demos, lengths {[d['total_steps'] for d in demos]}", flush=True)
    # Demo observations are regenerated by teleporting the env to every demo frame.
    extract_env = ReverseCurriculumIsaacEnv(
        inner, demos, max_relative_action_rate=rate, fixed_frontier=True, **core_kwargs
    )
    demo_transitions = extract_demo_transitions_robolab(demos, extract_env, device)
    if z_dim > 0:
        # Demo z are fixed by cfg.seed so a resumed run sees the same demonstration buffer.
        demo_transitions = augment_demo_transitions_with_z(demo_transitions, z_dim, seed=int(cfg.seed))

    # A new wrapper attaches a fresh curriculum to the same env.
    train_env: Any = ReverseCurriculumIsaacEnv(
        inner, demos, max_relative_action_rate=rate, to_numpy=True, **core_kwargs
    )
    if z_dim > 0:
        train_env = SkillZConcatWrapper(train_env, z_dim, seed=seed)

    ###############################
    # agent, observation bounds, METRA, demo buffer
    ###############################
    _, env_info = train_env.reset()
    agent: Any = create_agent(train_env.observation_space, train_env.action_space, env_info, cfg.agent)

    layout = term_layout_from_env(inner)
    low, high = resolve_obs_bounds(
        layout, ObsBoundsSpec.from_config(cfg.obs_bounds), joint_limits=arm_joint_limits_from_env(inner)
    )
    demo_obs = demo_transitions["observation"]
    outside = demo_outside_fraction(demo_obs[:, : low.numel()], low, high)
    print("[train] observation bounds\n" + format_bounds_table(layout, low, high, outside), flush=True)
    # A dim that clips more than 5% of the demo observations means a wrong box.
    over = [(i, float(outside[i])) for i in range(low.numel()) if float(outside[i]) > 0.05]
    assert not over, f"obs_bounds: demo observations fall outside the bounds in dims {over}"
    # Skill z dims use the window [-1, 1] (values beyond it are clipped).
    agent._normalizer.set_obs_bounds(torch.cat([low, -torch.ones(z_dim)]), torch.cat([high, torch.ones(z_dim)]))
    for key in ("observation", "next_observation"):
        demo_transitions[key] = agent._normalizer.normalize_obs(demo_transitions[key])

    if alpha > 0:
        phi_dims = _phi_dims(inner)
        gate_table = SkillGateTable(device=str(agent._device))
        metra = MetraRepresentation(
            obs_sub_dim=len(phi_dims),
            z_dim=z_dim,
            hidden=[int(h) for h in cfg.skill.phi_hidden],
            lr=float(cfg.skill.phi_lr),
            dual_slack=float(cfg.skill.dual_slack),
            lam_init=float(cfg.skill.lam_init),
            device=str(agent._device),
        )
        agent.set_skill_modules(metra, gate_table, alpha=alpha, z_dim=z_dim, phi_dims=torch.tensor(phi_dims))

    demo_buffer = DemoBuffer(
        train_env.observation_space,
        train_env.action_space,
        sample_batch_size=int(cfg.agent.sample_batch_size),
        device_type=str(cfg.agent.buffer_device_type),
        n_step=int(cfg.n_step),
        gamma=float(cfg.gamma),
    )
    demo_buffer.fill_from_transitions(demo_transitions)
    agent._replay_buffer = MixedReplayBuffer(agent._replay_buffer, demo_buffer, float(cfg.rfcl.demo_ratio))
    print(f"[train] demo buffer: {len(demo_buffer)} transitions, demo_ratio={cfg.rfcl.demo_ratio}", flush=True)

    ###############################
    # resume, logger, save dir, eval worker
    ###############################
    # A resumed run continues the logger run recorded in its checkpoint.
    logger = create_logger(cfg, resume_id=None if resume_meta is None else resume_meta["run_id"])

    save_dir = _repo_path(str(cfg.save_path).replace("TIMESTAMP", datetime.now().strftime("%m%d-%H%M%S")))
    os.makedirs(save_dir, exist_ok=False)  # two runs must not share a directory
    OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))
    eval_worker = _EvalWorker(save_dir, cfg)

    env_step = 0
    interaction_step = 0
    forward_active = False
    if resume_meta is not None:
        agent.load(cfg.agent_load_path)
        agent.load_replay_buffer(cfg.agent_load_path)
        env_step = int(resume_meta["env_step"])
        interaction_step = int(resume_meta["interaction_step"])
        forward_active = bool(resume_meta["forward_active"])
        if not forward_active:
            restore_reverse_curriculum_state(train_env, resume_meta)
        print(
            f"[train] resumed from {cfg.agent_load_path}: env_step={env_step} "
            f"phase={'forward' if forward_active else 'reverse'}",
            flush=True,
        )

    def build_forward_env() -> Any:
        """Close the reverse env and build the forward env (randomized initial states) in the same app."""
        inner.envs.close()
        gc.collect()
        torch.cuda.empty_cache()
        env: Any = make_robolab_rl_env(num_envs, seed, curriculum=False, safety_penalty=safety_penalty, **env_kwargs)
        if z_dim > 0:
            env = SkillZConcatWrapper(env, z_dim, seed=seed + 1)
        # Running returns and cached action noise belong to the old env instances.
        agent.reward_normalizer.reset_running_returns()
        agent.reset_action_noise_state(env.action_space)
        return env

    if forward_active:
        train_env = build_forward_env()

    ###############################
    # training loop
    ###############################
    iterations = int(cfg.num_env_steps) // num_envs
    log_every = max(2, iterations // 1000)
    eval_every = max(2, iterations // 100)  # 100 evaluations per run
    upi = int(cfg.updates_per_interaction_step)
    solved_frac_threshold = float(cfg.rfcl.solved_frac_threshold)
    observations, _ = train_env.reset()
    transition: dict[str, Any] | None = None
    penalty_sum: dict[str, float] = {}
    penalty_n = 0

    # Success gate (alpha > 0): per-env episode uid (0 is reserved for demos) and whether the
    # current episode has succeeded; recorded when the episode ends.
    def fresh_uids() -> np.ndarray:
        start = agent.skill_episode_uid_next
        agent.set_skill_episode_uid_next(start + num_envs)
        return np.arange(start, start + num_envs, dtype=np.int64)

    if alpha > 0:
        ep_uid = fresh_uids()
        ep_success = np.zeros(num_envs, dtype=bool)

    pbar = tqdm.tqdm(total=int(cfg.num_env_steps), initial=env_step, smoothing=0.1, mininterval=0.5)
    while env_step < int(cfg.num_env_steps):
        interaction_step += 1
        if agent.can_start_training() and transition is not None:
            actions = np.array(agent.sample_actions(interaction_step, prev_transition=transition, training=True))
        else:
            actions = np.array(train_env.action_space.sample())
        next_observations, rewards, terminateds, truncateds, infos = train_env.step(actions)
        env_step += num_envs
        pbar.update(num_envs)
        for k, v in infos["safety_penalty"].items():
            penalty_sum[k] = penalty_sum.get(k, 0.0) + float(v)
        penalty_n += 1

        done = np.asarray(terminateds, dtype=bool) | np.asarray(truncateds, dtype=bool)
        transition = {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            # Done envs bootstrap from the pre-reset observation.
            "next_observation": np.where(done[:, None], infos["final_obs"], next_observations),
        }
        if alpha > 0:
            ep_success |= np.asarray(rewards) > 0
            transition["ep_uid"] = ep_uid.copy()
        agent.process_transition(transition)
        if alpha > 0 and done.any():
            gate_table.set(torch.as_tensor(ep_uid[done]), torch.as_tensor(ep_success[done]))
            n_done = int(done.sum())
            start = agent.skill_episode_uid_next
            ep_uid[done] = np.arange(start, start + n_done, dtype=np.int64)
            agent.set_skill_episode_uid_next(start + n_done)
            ep_success[done] = False
        transition["next_observation"] = next_observations
        observations = next_observations

        if agent.can_start_training():
            for _ in range(upi):
                logger.update_metric(**agent.update(env_step=env_step))
            logger.update_metric(
                **{
                    "actor/lr": agent._actor.optimizer.param_groups[0]["lr"],
                    "critic/lr": agent._critic.optimizer.param_groups[0]["lr"],
                }
            )

        if interaction_step % log_every == 0:
            logger.update_metric(**{f"penalty/{k}": v / penalty_n for k, v in penalty_sum.items()})
            penalty_sum, penalty_n = {}, 0
            logger.update_metric(**{"curriculum/phase": 2.0 if forward_active else 1.0})
            if not forward_active:
                core = train_env.core
                rates = [core.frontier_success_rate(i) for i in range(core.num_demos)]
                logger.update_metric(
                    **{
                        "curriculum/mean_frontier": float(np.mean([md.start_step for md in core.demo_metadata])),
                        "curriculum/solved_frac": core.reverse_solved_frac,
                        "curriculum/mean_frontier_success_rate": float(np.mean(rates)),
                    }
                )
                for i, md in enumerate(core.demo_metadata):
                    logger.update_metric(
                        **{
                            f"curriculum-demo/{i}/frontier": float(md.start_step),
                            f"curriculum-demo/{i}/solved": float(md.solved),
                            f"curriculum-demo/{i}/frontier_success_rate": rates[i],
                        }
                    )
            logger.log_metric(step=env_step)
            logger.reset()

        # Every evaluation checkpoint is fully resumable (it includes the latest replay buffer).
        if interaction_step % eval_every == 0:
            ckpt = os.path.join(save_dir, f"step{interaction_step}")
            _save_checkpoint(agent, train_env, ckpt, interaction_step, env_step, logger.run_id, forward_active)
            _prune_replay_buffers(save_dir, ckpt)
            eval_worker.request(env_step, ckpt)
        if interaction_step % 25 == 0:
            eval_worker.drain(logger)

        if not forward_active and train_env.reverse_solved_frac >= solved_frac_threshold:
            print(
                f"[train] solved_frac={train_env.reverse_solved_frac:.2f} at env_step={env_step}: forward phase",
                flush=True,
            )
            # Reverse-end snapshot, kept by the buffer pruning.
            ckpt = os.path.join(save_dir, f"step{interaction_step}_reverse_end")
            _save_checkpoint(agent, train_env, ckpt, interaction_step, env_step, logger.run_id, False)
            train_env = build_forward_env()
            forward_active = True
            observations, _ = train_env.reset()
            transition = None
            if alpha > 0:
                # Interrupted episodes get no gate record.
                ep_uid = fresh_uids()
                ep_success[:] = False

    pbar.close()
    final_ckpt = os.path.join(save_dir, f"step{interaction_step}")
    if not os.path.exists(final_ckpt):
        _save_checkpoint(agent, train_env, final_ckpt, interaction_step, env_step, logger.run_id, forward_active)
        _prune_replay_buffers(save_dir, final_ckpt)
        eval_worker.request(env_step, final_ckpt)
    eval_worker.finish(logger, env_step, max_episode_steps)
    print(f"[train] done: env_step={env_step} final checkpoint {final_ckpt}", flush=True)
    if cfg.logger_type == "wandb":
        # __main__ exits with os._exit, which skips wandb's atexit hook.
        import wandb

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs")
    parser.add_argument("--config_name", default="fr3")
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()
    # os._exit skips Isaac's teardown, which can hang and keep the GPU; logs and checkpoints are
    # already flushed by run().
    try:
        run(args)
    except SystemExit as e:  # argparse --help / usage errors
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(e.code if isinstance(e.code, int) else int(e.code is not None))
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
