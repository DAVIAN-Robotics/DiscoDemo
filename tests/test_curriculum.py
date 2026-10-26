import numpy as np
import pytest

from flash_rl.rfcl.curriculum_core import CurriculumCore


def make(lengths=(40, 30), window=4, stride=16):
    return CurriculumCore(
        8,
        list(lengths),
        "cpu",
        reverse_step_size=stride,
        advance_threshold=0.75,
        frontier_window=window,
        minimum_episode_steps=30,
    )


def test_frontier_starts_at_last_frame_and_moves_back():
    core = make()
    assert [md.start_step for md in core.demo_metadata] == [39, 29]
    for _ in range(3):
        core.record_outcome(1, 39, 0)
    core.record_outcome(0, 39, 0)
    assert core.frontier_success_rate(0) == 0.75
    assert core.step_curriculum()
    assert core.demo_metadata[0].start_step == 39 - 16
    # The window restarts at the new frontier.
    assert core.frontier_success_rate(0) == 0.0


def test_only_frontier_starts_count():
    core = make()
    for _ in range(4):
        core.record_outcome(1, 38, 0)
    assert core.frontier_success_rate(0) == 0.0
    assert not core.step_curriculum()


def test_solved_after_frontier_reaches_zero():
    core = make(lengths=(10,), stride=16)
    for _ in range(4):
        core.record_outcome(1, 9, 0)
    core.step_curriculum()
    assert core.demo_metadata[0].start_step == 0
    for _ in range(4):
        core.record_outcome(1, 0, 0)
    core.step_curriculum()
    assert core.demo_metadata[0].solved
    assert core.reverse_solved_frac == 1.0


def test_start_states_are_geometric_from_frontier():
    core = make(lengths=(200,))
    core.demo_metadata[0].start_step = 100
    core.mark_active_dirty()
    starts = np.array([core.assign_reset(0)[1] for _ in range(4000)])
    assert starts.min() >= 100
    assert abs(np.mean(starts == 100) - 0.5) < 0.04
    assert abs(np.mean(starts == 101) - 0.25) < 0.04


def test_time_limit_and_state_round_trip():
    core = make()
    assert core.compute_dynamic_timelimit_from_remaining(12) == 42
    core.record_outcome(1, 39, 0)
    other = make()
    other.set_frontier_state(core.frontier_state())
    assert other.frontier_state() == [[1], []]
    with pytest.raises(ValueError):
        other.set_frontier_state([[1]])
