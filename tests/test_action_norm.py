import math

import torch

from robolab.robots.action_norm import (
    ARM_J7_INDEX,
    JerkLimitedJointCommand,
    command_to_physical,
    drop_ineffective_gripper_requests,
    relative_applied_target,
    relative_delta_demo_arm,
)


def test_command_frame():
    q = torch.zeros(7)
    p = command_to_physical(q)
    assert math.isclose(float(p[ARM_J7_INDEX]), math.pi / 4, rel_tol=1e-6)
    assert float(p[:ARM_J7_INDEX].abs().sum()) == 0.0


def test_relative_label_inverts_the_action_term():
    r = torch.full((7,), 0.02)
    q_cur = torch.linspace(-0.5, 0.5, 7)
    a = torch.linspace(-0.9, 0.9, 7)
    target = relative_applied_target(a * r, q_cur, q_cur - 1, q_cur + 1)
    q_cmd = target.clone()
    q_cmd[ARM_J7_INDEX] -= math.pi / 4
    assert torch.allclose(relative_delta_demo_arm(q_cmd, q_cur, r), a, atol=1e-6)


def test_jerk_filter_respects_limits():
    vmax = torch.tensor([0.2, 0.5])
    amax = torch.tensor([0.5, 1.0])
    jmax = torch.tensor([3.0, 5.0])
    f = JerkLimitedJointCommand(4, 0.05, vmax, amax, jmax)
    g = torch.Generator().manual_seed(0)
    prev_a = torch.zeros(4, 2)
    for _ in range(400):
        f.step(torch.rand(4, 2, generator=g) * 2 - 1)
        assert bool((f.velocity.abs() <= vmax + 1e-5).all())
        assert bool((f.acceleration.abs() <= amax + 1e-5).all())
        assert bool(((f.acceleration - prev_a).abs() <= jmax * 0.05 + 1e-5).all())
        prev_a = f.acceleration.clone()
    f.reset([0])
    assert float(f.velocity[0].abs().sum()) == 0.0 and float(f.acceleration[0].abs().sum()) == 0.0


def test_gripper_requests_that_never_applied_are_dropped():
    req = torch.tensor([0, 1, 0, 0, 1, 1, 1], dtype=torch.float32)
    app = torch.tensor([0, 0, 0, 0, 0, 1, 1], dtype=torch.float32)
    # The one-step close at t=1 was never applied; the run from t=4 was applied (from t=5).
    assert drop_ineffective_gripper_requests(req, app).tolist() == [0, 0, 0, 0, 1, 1, 1]
