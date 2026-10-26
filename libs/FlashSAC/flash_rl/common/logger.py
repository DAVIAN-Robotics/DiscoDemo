"""Training loggers (wandb / tensorboard) and metric averaging."""

from typing import Any

import numpy as np
from omegaconf import OmegaConf


class WandbTrainerLogger:
    """Logs scalar and video metrics to wandb.

    Evaluation results arrive after training has moved on, so ``eval/*`` metrics use their own
    step axis ``eval/env_step`` (the env step of the evaluated checkpoint).
    """

    def __init__(self, cfg: Any, resume_id: str | None = None):
        import wandb

        self._wandb = wandb
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, throw_on_missing=True)
        # resume_id appends to an existing run, so a resumed run is not split across runs.
        wandb.init(
            project=cfg.project_name,
            entity=cfg.entity_name,
            group=cfg.group_name,
            name=cfg.exp_name,
            config=dict_cfg,  # type: ignore
            id=resume_id,
            resume="allow" if resume_id is not None else None,
        )
        wandb.define_metric("eval/env_step")
        wandb.define_metric("eval/*", step_metric="eval/env_step")
        self.media_dict: dict[str, Any] = {}
        self.reset()

    @property
    def run_id(self) -> str:
        """wandb run id (stored in checkpoints so a resume continues the run)."""
        return str(self._wandb.run.id)

    def log_eval(self, metrics: dict[str, float], env_step: int, video_path: str | None) -> None:
        """Log one evaluation result on the ``eval/env_step`` axis."""
        data: dict[str, Any] = {"eval/env_step": env_step, **metrics}
        if video_path is not None:
            data["eval/video"] = self._wandb.Video(video_path, format="mp4")
        self._wandb.log(data)

    def update_metric(self, **kwargs: Any) -> None:
        """Accumulate scalars into averages; keep 5D arrays as videos and anything else as media."""
        for k, v in kwargs.items():
            if isinstance(v, (float, int)):
                self.average_meter_dict.update(k, v)
            elif isinstance(v, np.ndarray) and v.ndim == 5:
                self.media_dict[k] = self._wandb.Video(v, fps=30, format="mp4")
            else:
                self.media_dict[k] = v

    def log_metric(self, step: int) -> None:
        """Flush averaged scalars and media to wandb at ``step``."""
        log_data = {}
        log_data.update(self.average_meter_dict.averages())
        log_data.update(self.media_dict)
        self._wandb.log(log_data, step=step)

    def reset(self) -> None:
        """Clear the meters and media buffer."""
        self.average_meter_dict = AverageMeterDict()
        self.media_dict.clear()


class TensorboardTrainerLogger:
    """Logs scalar and video (GIF) metrics to TensorBoard."""

    def __init__(self, cfg: Any, resume_id: str | None = None):
        from datetime import datetime

        from torch.utils.tensorboard import SummaryWriter

        # A resumed run writes to the directory of the run it continues.
        self.run_id = resume_id or f"seed{cfg.seed}_{datetime.now().strftime('%m%d_%H%M%S')}"
        log_dir = f"runs/{cfg.group_name}/{cfg.exp_name}/{self.run_id}"
        self.writer = SummaryWriter(log_dir=log_dir)  # type: ignore[no-untyped-call]
        self.writer.add_text("config", OmegaConf.to_yaml(cfg))  # type: ignore[no-untyped-call]
        self.media_dict: dict[str, Any] = {}
        self.reset()

    def update_metric(self, **kwargs: Any) -> None:
        """Accumulate scalars into averages; keep anything else as media."""
        for k, v in kwargs.items():
            if isinstance(v, (float, int)):
                self.average_meter_dict.update(k, v)
            else:
                self.media_dict[k] = v

    def log_eval(self, metrics: dict[str, float], env_step: int, video_path: str | None) -> None:
        """Write one evaluation result at ``env_step`` (the video stays at ``video_path``)."""
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, global_step=env_step)  # type: ignore[no-untyped-call]
        self.writer.flush()  # type: ignore[no-untyped-call]

    def log_metric(self, step: int) -> None:
        """Write averaged scalars and 5D videos (as GIFs) to TensorBoard."""
        for k, v in self.average_meter_dict.averages().items():
            self.writer.add_scalar(k, v, global_step=step)  # type: ignore[no-untyped-call]
        for k, v in self.media_dict.items():
            if isinstance(v, np.ndarray) and v.ndim == 5:
                # v: (B, T, C, H, W) uint8 — encode to animated gif, bypassing moviepy
                # TensorBoard's frontend has no native mp4 video summary; GIF via Image summary is the only path.
                import io

                from PIL import Image
                from tensorboard.compat.proto.summary_pb2 import Summary

                frames = v[0].transpose(0, 2, 3, 1)  # (T, H, W, C)
                frames_pil = [Image.fromarray(f) for f in frames]
                buf = io.BytesIO()
                frames_pil[0].save(
                    buf,
                    format="GIF",
                    save_all=True,
                    append_images=frames_pil[1:],
                    duration=33,
                    loop=0,
                )
                _, h, w, _ = frames.shape
                image_summary = Summary.Image(encoded_image_string=buf.getvalue(), height=h, width=w)
                summary = Summary(value=[Summary.Value(tag=k, image=image_summary)])
                assert self.writer.file_writer is not None
                self.writer.file_writer.add_summary(summary, global_step=step)
        self.writer.flush()  # type: ignore[no-untyped-call]

    def reset(self) -> None:
        """Clear the meters and media buffer."""
        self.average_meter_dict = AverageMeterDict()
        self.media_dict.clear()


class AverageMeter:
    """Tracks and calculates the average and current values of a series of numbers."""

    def __init__(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Add ``val`` with weight ``n`` to the running average."""
        # TODO: description for using n
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format: str) -> str:
        """Format as ``"val (avg)"``."""
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class AverageMeterDict:
    """A dict of named AverageMeters."""

    def __init__(self, meters: dict[str, AverageMeter] | None = None):
        self.meters = meters if meters else {}

    def __getitem__(self, key: str) -> AverageMeter:
        """Return the meter for ``name`` (a zeroed temporary meter if missing)."""
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name: str, value: float, n: int = 1) -> None:
        """Update the meter ``name``, creating it if needed."""
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self) -> None:
        """Reset all meters."""
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string: str = "{}") -> dict[str, float]:
        """Return ``{formatted name: current value}``."""
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string: str = "{}") -> dict[str, float]:
        """Return ``{formatted name: average}``."""
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}
