"""Batched pi0.5 inference for DROID joint-position checkpoints (base model or an SFT checkpoint).

Adds the DROID I/O that LeRobot lacks: the 8-dim state (7 joints in the command frame + gripper closure),
delta -> absolute joints, and a binary gripper command. Each predicted chunk is executed open loop for
``n_action_steps`` steps; all envs replan together.
"""

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

# Registers ActionDeltaProcessorStep, which SFT checkpoints serialize in their preprocessor.
from sft.action_delta import DELTA_MASK

#: DROID state: 7 arm joints + gripper closure; the state tokenizer must discretize exactly these.
STATE_DIM = 8


class Pi05Policy:
    """pi0.5 policy over a batch of envs.

    Parameters
    ----------
    checkpoint : str
        Local directory or hub repo with ``config.json``, ``model.safetensors`` and the processors.
    device : str
        Torch device.
    prompt : str
        Task instruction.
    """

    def __init__(self, checkpoint: str, device: str, prompt: str):
        policy = PI05Policy.from_pretrained(checkpoint)
        # embed_tokens is tied to lm_head in Gemma and is not stored in the checkpoint.
        pal = policy.model.paligemma_with_expert.paligemma
        pal.model.language_model.embed_tokens.weight.data.copy_(pal.lm_head.weight.data)
        policy.config.device = device
        self.policy = policy.to(device).eval()
        self.device = device
        self.prompt = prompt
        self.n_action_steps = int(policy.config.n_action_steps)

        # Exterior camera = the image key without "wrist"; wrist camera = the left wrist key.
        img_keys = [k for k in policy.config.input_features if k.startswith("observation.images")]
        (self.key_exterior,) = [k for k in img_keys if "wrist" not in k]
        (self.key_wrist,) = [k for k in img_keys if "left_wrist" in k]

        self.preprocessor, self.postprocessor = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
        (state_step,) = [
            s for s in self.preprocessor.steps if type(s).__name__ == "Pi05PrepareStateTokenizerProcessorStep"
        ]
        state_step.max_state_dim = STATE_DIM
        self._chunk: np.ndarray | None = None  # [B, n_action_steps, 8]
        self._t = 0

    def reset(self) -> None:
        """Drop the queued actions (episode start)."""
        self._chunk = None

    @torch.no_grad()
    def act(self, exterior: np.ndarray, wrist: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Next action for every env.

        Parameters
        ----------
        exterior, wrist : numpy.ndarray
            ``[B, H, W, 3]`` uint8 camera images.
        state : numpy.ndarray
            ``[B, 8]`` joints (command frame) + gripper closure in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray
            ``[B, 8]`` absolute joint targets (command frame) + gripper command in ``{0, 1}``.
        """
        if self._chunk is None or self._t == self.n_action_steps:
            # Contiguous NCHW float in [0, 1]: the image resize kernel depends on the memory layout.
            batch = {
                self.key_exterior: torch.from_numpy(exterior.astype(np.float32) / 255.0)
                .permute(0, 3, 1, 2)
                .contiguous(),
                self.key_wrist: torch.from_numpy(wrist.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous(),
                "observation.state": torch.from_numpy(state.astype(np.float32)),
                "task": [self.prompt] * len(state),
            }
            proc = self.preprocessor(batch)
            proc = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in proc.items()}
            chunk = self.postprocessor(self.policy.predict_action_chunk(proc))  # [B, H, 32]
            chunk = chunk.float().cpu().numpy()[:, : self.n_action_steps, :STATE_DIM].copy()
            # Delta -> absolute: the state at prediction time is the base of the whole chunk.
            mask = np.asarray(DELTA_MASK)
            chunk += np.where(mask, state.astype(np.float32), 0.0)[:, None, :]
            chunk[..., -1] = (chunk[..., -1] > 0.5).astype(np.float32)
            self._chunk, self._t = chunk, 0
        a = self._chunk[:, self._t]
        self._t += 1
        return a
