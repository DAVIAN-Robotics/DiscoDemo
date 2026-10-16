"""Logger factory."""

from typing import Any

from flash_rl.common.logger import TensorboardTrainerLogger, WandbTrainerLogger  # noqa

TrainerLogger = WandbTrainerLogger | TensorboardTrainerLogger


def create_logger(cfg: Any, resume_id: str | None = None) -> TrainerLogger:
    """Create a wandb or tensorboard logger according to ``cfg.logger_type``.

    ``resume_id`` (the ``run_id`` saved in a checkpoint) continues that run.
    """
    if cfg.logger_type == "wandb":
        return WandbTrainerLogger(cfg, resume_id)
    if cfg.logger_type == "tensorboard":
        return TensorboardTrainerLogger(cfg, resume_id)
    raise ValueError(f"unknown logger_type {cfg.logger_type!r}")


__all__ = [
    "WandbTrainerLogger",
    "TensorboardTrainerLogger",
    "create_logger",
]
