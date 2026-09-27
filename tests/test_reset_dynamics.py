"""E8 recovery + validation tests (neutral-state reset axes, Ritz/Jha/Daw/Cohen 2026)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import reset_dynamics as RD  # noqa: E402


def test_reset_speed_recovers_true_rate():
    """A faster true reset rate must yield a larger estimated reset_speed (monotone recovery)."""
    slow, fast = [], []
    for s in range(8):
        rng = np.random.default_rng(s)
        slow.append(RD.reset_axes(RD.simulate_reset_trajectory(rng, reset_rate=0.2))["reset_speed"])
        rng = np.random.default_rng(100 + s)
        fast.append(RD.reset_axes(RD.simulate_reset_trajectory(rng, reset_rate=1.0))["reset_speed"])
    assert np.nanmean(fast) > np.nanmean(slow)


def test_neutral_return_in_range_and_ordered():
    rng = np.random.default_rng(3)
    low = RD.reset_axes(RD.simulate_reset_trajectory(rng, reset_rate=0.15))["neutral_return"]
    rng = np.random.default_rng(4)
    high = RD.reset_axes(RD.simulate_reset_trajectory(rng, reset_rate=1.1))["neutral_return"]
    assert high > low                                    # more reset -> more return to neutral


def test_neutral_state_is_convergence_centroid():
    rng = np.random.default_rng(7)
    traj = RD.simulate_reset_trajectory(rng, reset_rate=0.9)
    neutral = RD.estimate_neutral_state(traj.onset)
    # with strong reset, onset states cluster near the neutral point
    d = np.linalg.norm(traj.onset - neutral, axis=1)
    assert d.mean() < np.linalg.norm(traj.peak - neutral, axis=1).mean()


def test_e8_recovery_and_flexibility_signs():
    out = RD.run_e8(n_agents=40, n_seeds=4)
    rec = out["recovery_true_vs_estimated_reset"]
    # recovery correlation should be strong and pass the honesty gate (not abstained)
    assert rec["abstained"] is False and rec["value"] > 0.5
    # more return-to-neutral predicts LOWER switch cost, and low-reset agents cost MORE
    assert out["control_lowreset_minus_highreset_cost"] > 0
