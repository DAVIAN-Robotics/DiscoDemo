import numpy as np

from flash_rl.rfcl import collision_metrics as cm

ROBOT = ["panda_hand", "panda_leftfinger"]
SCENE = ["table", "banana"]
MASK = cm.illegal_pair_mask(ROBOT, SCENE, {("panda_leftfinger", "banana")})


def signals(illegal, container=None, axial=None, lateral=None):
    pf = np.zeros((len(illegal), 2, 2))
    pf[:, 0, 0] = illegal  # hand x table: illegal
    pf[:, 1, 1] = 500.0  # finger x banana: allowed, never counted
    cxy = np.zeros((len(illegal) + 1, 2)) if container is None else np.asarray(container)
    return cm.trajectory_signals(pf, MASK, cxy, axial, lateral)


def test_mask():
    assert MASK.tolist() == [[True, True], [True, False]]


def test_peak_and_sustained_thresholds():
    assert cm.export_filter_drops(signals([0, 60, 0, 0]))
    assert not cm.export_filter_drops(signals([0, 45, 0, 0]))
    assert cm.export_filter_drops(signals([25, 25, 25, 0]))  # 3 steps above 20 N
    assert not cm.export_filter_drops(signals([25, 25, 0, 25]))


def test_force_cap_and_integral():
    s = signals([300.0, 0.0])
    assert s["contact_peak"] == cm.FORCE_CAP
    # The integral is unclipped (force x control period).
    assert np.isclose(s["illegal_contact"], 300.0 * cm.CONTROL_DT)


def test_jam_deadbands():
    s = signals([0, 0], axial=[5.0, 8.0], lateral=[15.0, 19.0])
    assert np.isclose(s["jam_norm"], 5.0)  # sqrt(3^2 + 4^2)
    assert not cm.export_filter_drops(s)
    assert cm.export_filter_drops(signals([0, 0], axial=[30.0, 0.0], lateral=[0.0, 0.0]))


def test_container_displacement():
    s = signals([0, 0], container=[[0.5, 0.0], [0.5, 0.004], [0.508, 0.006]])
    assert np.isclose(s["container_disp"], 0.01)
    assert cm.export_filter_drops(s)
