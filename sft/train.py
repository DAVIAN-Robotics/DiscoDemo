"""pi0.5 delta-action SFT on a generated dataset, then evaluation of the final checkpoint in Isaac Sim.

Launched by ``scripts/3_sft/train.sh`` (``accelerate launch -m sft.train ...``), which holds the paper recipe.
The evaluation (``sft/sim_eval.py``) runs in a subprocess after training, because Isaac Sim needs its own
process; its result is logged under ``eval/`` at the final step.

Resume an interrupted run with ``--resume true --config_path <output_dir>/checkpoints/last/pretrained_model/
train_config.json``.
"""

import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import NormalizerProcessorStep
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.constants import CHECKPOINTS_DIR
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import init_logging
from tqdm import tqdm

from real2sim.env_contract import TASKS
from sft.action_delta import DELTA_MASK, ActionDeltaProcessorStep, compute_delta_action_stats, policy_action_window

REPO = Path(__file__).resolve().parents[1]


@dataclass
class SFTConfig(TrainPipelineConfig):
    """LeRobot ``TrainPipelineConfig`` plus the task to evaluate and checkpoint retention."""

    # Task of the dataset; the final checkpoint is evaluated on it.
    task: str = ""
    # Evaluate the final checkpoint in Isaac Sim.
    sim_eval: bool = True
    # Step checkpoints kept on disk during training (for resume); only the last one is kept at the end.
    num_save_checkpoints: int = 3


def freeze_for_sft(policy: Any) -> None:
    """Train the vision encoder, the language model and the action expert; freeze everything else.

    The action/time projections, the multi-modal projector and the lm head keep their pretrained weights.
    This is the setting the paper models were trained with.
    """
    policy.requires_grad_(False)
    pg = policy.model.paligemma_with_expert
    pg.paligemma.vision_tower.requires_grad_(True)
    pg.paligemma.language_model.requires_grad_(True)
    pg.gemma_expert.model.requires_grad_(True)
    policy.get_optim_params = lambda: [p for p in policy.parameters() if p.requires_grad]


def policy_features(cfg: SFTConfig, dataset: Any) -> tuple[dict, dict]:
    """Policy input and output features after ``rename_map``, in the pretrained checkpoint's input order.

    pi0.5 assigns camera slots by order. Checkpoint cameras missing from the dataset are allowed (pi0.5 pads
    them with a masked image); any other missing input is an error.
    """
    from huggingface_hub import hf_hub_download

    renamed = {cfg.rename_map.get(k, k): v for k, v in dataset_to_policy_features(dataset.meta.features).items()}
    pp = Path(str(cfg.policy.pretrained_path))
    cfg_path = pp / "config.json" if pp.is_dir() else Path(hf_hub_download(str(pp), "config.json"))
    order = list(json.loads(cfg_path.read_text())["input_features"])
    missing = [k for k in order if k not in renamed and "observation.images" not in k]
    if missing:
        raise ValueError(f"checkpoint inputs {missing} are not in the dataset after rename_map: {list(renamed)}")
    inputs = {k: renamed[k] for k in order if k in renamed}
    return inputs, {"action": renamed["action"]}


def prune_checkpoints(output_dir: Path, keep: int) -> None:
    """Delete all but the newest ``keep`` step checkpoints."""
    ckpts = sorted((p for p in (output_dir / CHECKPOINTS_DIR).iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    for p in ckpts[:-keep]:
        logging.info(f"Deleting older checkpoint: {p}")
        shutil.rmtree(p)


def update_policy(
    tracker: Any, policy: Any, batch: Any, optimizer: Any, lr_scheduler: Any, grad_clip_norm: float, acc: Accelerator
) -> tuple[Any, dict]:
    """One optimizer step."""
    start = time.perf_counter()
    policy.train()
    with acc.autocast():
        loss, output_dict = policy.forward(batch)
    acc.backward(loss)
    grad_norm = acc.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()
    tracker.loss = loss.item()
    tracker.grad_norm = grad_norm.item()
    tracker.lr = optimizer.param_groups[0]["lr"]
    tracker.update_s = time.perf_counter() - start
    return tracker, output_dict


def run_sim_eval(cfg: SFTConfig, wandb_logger: WandBLogger | None) -> None:
    """Evaluate the final checkpoint in a subprocess and log the result at the final step."""
    ckpt = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, cfg.steps) / "pretrained_model"
    out = cfg.output_dir / "sim_eval"
    cmd = [sys.executable, "-m", "sft.sim_eval", str(ckpt), "--task", cfg.task, "--seed", str(cfg.seed)]
    subprocess.run([*cmd, "--out-dir", str(out)], cwd=REPO, check=True)
    result = json.loads((out / "result.json").read_text())
    logging.info(f"[sim_eval] success_rate={result['success_rate']:.3f} time={result['time_to_success_s']}")
    if wandb_logger is not None:
        metrics = {k: result[k] for k in ("success_rate", "time_to_success_s") if result[k] is not None}
        wandb_logger.log_dict(metrics, step=cfg.steps, mode="eval")
        wandb_logger.log_video(str(out / "video.mp4"), cfg.steps, mode="eval")


@parser.wrap()
def train(cfg: SFTConfig) -> None:
    """Load the dataset, build pi0.5 and its processors, train, then evaluate the final checkpoint.

    Parameters
    ----------
    cfg : SFTConfig
        Parsed configuration.
    """
    if cfg.policy is None or cfg.policy.type != "pi05" or not cfg.policy.pretrained_path:
        raise ValueError("sft.train fine-tunes a pretrained pi05 policy (--policy.type pi05 --policy.pretrained_path)")
    if cfg.task not in TASKS:
        raise ValueError(f"--task must be one of {TASKS}, got {cfg.task!r}")
    if not cfg.rename_map:
        raise ValueError("--rename_map must map the dataset cameras to the checkpoint's camera keys")
    cfg.validate()

    acc = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    init_logging(accelerator=acc)
    is_main = acc.is_main_process
    logging.info(pformat(cfg.to_dict()))
    wandb_logger = WandBLogger(cfg) if cfg.wandb.enable and is_main else None
    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=acc)

    dataset = make_dataset(cfg)
    cfg.policy.input_features, cfg.policy.output_features = policy_features(cfg, dataset)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    freeze_for_sft(policy)
    acc.wait_for_everyone()

    # A fresh run normalizes actions with delta statistics; a resumed run restores its processors (with the
    # delta step) from the checkpoint.
    processor_kwargs = {}
    if not cfg.resume:
        horizon, offset = policy_action_window(cfg.policy)
        dataset.meta.stats["action"] = compute_delta_action_stats(dataset, horizon=horizon, offset=offset)
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path if cfg.resume else None, **processor_kwargs
    )
    if preprocessor.steps[0].rename_map == {}:
        preprocessor.steps[0].rename_map = cfg.rename_map
    if not cfg.resume:
        # Absolute -> delta conversion runs in raw space, right before the normalizer.
        norm_idx = next(i for i, s in enumerate(preprocessor.steps) if isinstance(s, NormalizerProcessorStep))
        preprocessor.steps.insert(norm_idx, ActionDeltaProcessorStep(mask=DELTA_MASK))

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=acc.device.type != "cpu",
        drop_last=True,
    )
    policy, optimizer, dataloader, lr_scheduler = acc.prepare(policy, optimizer, dataloader, lr_scheduler)
    dl_iter = cycle(dataloader)
    acc.wait_for_everyone()

    meters = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "data_s": AverageMeter("data_s", ":.3f"),
        "save_s": AverageMeter("save_s", ":.3f"),
    }
    tracker = MetricsTracker(
        cfg.batch_size * acc.num_processes, dataset.num_frames, dataset.num_episodes, meters, initial_step=step
    )
    pbar = tqdm(total=cfg.steps, initial=step, disable=not is_main, desc="Training")
    while step < cfg.steps:
        t0 = time.time()
        batch = preprocessor(next(dl_iter))
        tracker.data_s = time.time() - t0
        tracker, output_dict = update_policy(
            tracker, policy, batch, optimizer, lr_scheduler, cfg.optimizer.grad_clip_norm, acc
        )
        step += 1
        pbar.update(1)
        tracker.step()

        if is_main and cfg.log_freq > 0 and step % cfg.log_freq == 0:
            logging.info(tracker)
            if wandb_logger is not None:
                wandb_logger.log_dict({**tracker.to_dict(), **(output_dict or {})}, step)
            tracker.reset_averages()
        if is_main and cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
            t0 = time.time()
            ckpt_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                ckpt_dir, step, cfg, acc.unwrap_model(policy), optimizer, lr_scheduler, preprocessor, postprocessor
            )
            update_last_checkpoint(ckpt_dir)
            prune_checkpoints(cfg.output_dir, cfg.num_save_checkpoints)
            tracker.save_s = time.time() - t0
        acc.wait_for_everyone()
    pbar.close()

    acc.end_training()
    if is_main:
        prune_checkpoints(cfg.output_dir, 1)
        if cfg.sim_eval:
            # Release the training GPU memory before Isaac Sim starts.
            del policy, optimizer, dataloader, dl_iter
            acc.free_memory()
            run_sim_eval(cfg, wandb_logger)


if __name__ == "__main__":
    train()
