import math
from types import SimpleNamespace

import torch

from real2sim.predicates.board_still import board_still


def make_env():
    data = SimpleNamespace(root_pos_w=torch.zeros(3, 3), root_quat_w=torch.tensor([[1.0, 0, 0, 0]] * 3))
    return SimpleNamespace(scene={"board": SimpleNamespace(data=data)}, episode_length_buf=torch.ones(3))


def test_reference_is_the_first_step_and_limits_apply():
    env = make_env()
    data = env.scene["board"].data
    data.root_pos_w[:, 0] = torch.tensor([0.5, 0.5, 0.5])
    assert board_still(env, "board", 0.01, 5.0).all()  # first step: reference = current pose
    env.episode_length_buf = torch.full((3,), 10.0)
    data.root_pos_w[1, 1] += 0.02  # pushed 2 cm
    half = math.radians(10.0) / 2  # turned 10 deg about z
    data.root_quat_w[2] = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])
    assert board_still(env, "board", 0.01, 5.0).tolist() == [True, False, False]
    env.episode_length_buf = torch.tensor([10.0, 1.0, 10.0])  # env 1 starts a new episode
    assert board_still(env, "board", 0.01, 5.0).tolist() == [True, True, False]
