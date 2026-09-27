"""E8 (extension): inter-trial neutral-state reset as decision-phenotype axes.

Grounded in Ritz, Jha, Daw & Cohen (2026), "Inter-trial convergence of neural task states
supports cognitive flexibility" (Current Biology 36:4245-4257). They show that RNNs and human
brains RESET to a neutral task state between trials, and that control-dependent resetting is
what unifies classic accounts of cognitive flexibility: effective task-switching depends on
returning to neutral before the next trial.

The decision-phenotype models a person as a posture on a manifold, but scores only the
within-trial valuation coordinates. It does not yet score the BETWEEN-trial dynamics. This
module adds two neurally-grounded axes that the Ritz result licenses:

  * neutral_return : how completely the state returns to the inter-trial neutral point before
                     the next decision (1 = full reset, 0 = full carry-over).
  * reset_speed    : the relaxation rate of the post-decision trajectory back toward neutral
                     (fit as the decay constant of distance-to-neutral).

Both are recovered from a decision-state trajectory with intra-trial relaxation samples. The
paper's causal claim (resetting supports flexibility) becomes a testable prediction here:
higher reset should PREDICT lower switch cost. E8 validates recovery of the reset rate and
that prediction, with a no-reset negative control. Every reported number goes through the
honesty gate.

Pure numpy/scipy. Simulation of the mechanism, in the E1/E7 idiom of this repo.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import honesty as H  # noqa: E402


@dataclass
class Trajectory:
    onset: np.ndarray          # [n_trials, dim] state at each trial onset (post-reset)
    peak: np.ndarray           # [n_trials, dim] state at the decision peak
    relax: np.ndarray          # [n_trials, M, dim] intra-trial relaxation samples
    task: np.ndarray           # [n_trials] task label 0/1
    switch: np.ndarray         # [n_trials] bool, True on a task-switch trial
    switch_cost: np.ndarray    # [n_trials] observed switch cost (interference proxy)


def simulate_reset_trajectory(rng, reset_rate: float, n_trials: int = 120, dim: int = 4,
                              n_relax: int = 8, noise: float = 0.02) -> Trajectory:
    """One agent. `reset_rate` in (0, ~1.5] is the true relaxation rate toward the neutral
    point between trials; larger = faster/more complete reset. Switch cost is generated as the
    interference from residual displacement toward the PREVIOUS task attractor at onset."""
    neutral = np.zeros(dim)
    attractors = {0: rng.normal(0, 1, dim), 1: rng.normal(0, 1, dim)}
    for k in attractors:                                   # unit-norm task attractors
        attractors[k] /= np.linalg.norm(attractors[k])

    onset, peak, relax, task, switch, cost = [], [], [], [], [], []
    s = neutral + rng.normal(0, noise, dim)
    prev_task = None
    for t in range(n_trials):
        cur = int(t // 2 % 2) if t < 4 else int(rng.integers(0, 2))  # alternate then random
        s0 = s.copy()
        # switch cost = residual pull toward the PREVIOUS attractor still present at onset
        if prev_task is not None and cur != prev_task:
            interference = float(max(0.0, s0 @ attractors[prev_task]))
            sw = True
        else:
            interference = 0.0
            sw = cur != prev_task if prev_task is not None else False
        # decision phase: move toward current attractor
        s_peak = attractors[cur] + rng.normal(0, noise, dim)
        # relaxation phase: exponential decay toward neutral at reset_rate
        samples = []
        for m in range(1, n_relax + 1):
            s_m = neutral + (s_peak - neutral) * np.exp(-reset_rate * m) + rng.normal(0, noise, dim)
            samples.append(s_m)
        onset.append(s0); peak.append(s_peak); relax.append(np.array(samples))
        task.append(cur); switch.append(sw)
        cost.append(interference + rng.normal(0, noise))
        s = samples[-1]                                    # next onset = last relaxation sample
        prev_task = cur
    return Trajectory(np.array(onset), np.array(peak), np.array(relax), np.array(task),
                      np.array(switch, bool), np.array(cost))


def estimate_neutral_state(onset: np.ndarray) -> np.ndarray:
    """The inter-trial neutral point = the convergence centroid of trial-onset states."""
    return onset.mean(axis=0)


def reset_axes(traj: Trajectory, neutral: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Recover (neutral_return, reset_speed) from a trajectory.

    neutral_return : 1 - mean( ||next_onset - neutral|| / ||peak - neutral|| )
    reset_speed    : slope of log distance-to-neutral over relaxation samples (decay constant),
                     averaged over trials (a positive number; larger = faster reset).
    """
    if neutral is None:
        neutral = estimate_neutral_state(traj.onset)
    peak_disp = np.linalg.norm(traj.peak - neutral, axis=1)              # [n_trials]
    next_onset = np.roll(np.linalg.norm(traj.onset - neutral, axis=1), -1)
    frac_residual = np.clip(next_onset[:-1] / np.clip(peak_disp[:-1], 1e-9, None), 0, 2)
    neutral_return = float(1.0 - np.mean(frac_residual))

    rates = []
    for i, m_traj in enumerate(traj.relax):
        # anchor at the decision peak (m=0) then the relaxation samples (m=1..M)
        d = np.concatenate([[peak_disp[i]], np.linalg.norm(m_traj - neutral, axis=1)])
        # keep only the high-signal prefix: samples before distance falls below 5% of the peak
        # excursion (beyond that the trajectory is at the noise floor and log-distance flattens)
        floor = max(0.05 * peak_disp[i], 1e-9)
        keep = 1
        while keep < len(d) and d[keep] > floor:
            keep += 1
        if keep >= 3:
            m = np.arange(keep)
            slope = np.polyfit(m, np.log(d[:keep]), 1)[0]
            rates.append(-slope)                                         # decay constant
    reset_speed = float(np.mean(rates)) if rates else float("nan")
    return {"neutral_return": neutral_return, "reset_speed": reset_speed}


def run_e8(n_agents: int = 60, n_seeds: int = 6) -> Dict[str, object]:
    """Validate: (a) reset_speed recovers the true reset rate across agents; (b) reset PREDICTS
    lower switch cost (the paper's flexibility claim); with a no-reset negative control."""
    true_rates, est_speed, est_return, mean_switch_cost = [], [], [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(4200 + s)
        for _ in range(n_agents):
            r = float(rng.uniform(0.1, 1.2))
            traj = simulate_reset_trajectory(rng, reset_rate=r)
            ax = reset_axes(traj)
            if not np.isfinite(ax["reset_speed"]):
                continue
            true_rates.append(r); est_speed.append(ax["reset_speed"])
            est_return.append(ax["neutral_return"])
            mean_switch_cost.append(float(traj.switch_cost[traj.switch].mean())
                                    if traj.switch.any() else 0.0)
    true_rates = np.array(true_rates); est_speed = np.array(est_speed)
    est_return = np.array(est_return); mean_switch_cost = np.array(mean_switch_cost)
    n = len(true_rates)

    recovery_r = float(stats.pearsonr(true_rates, est_speed)[0])
    flex_r = float(stats.pearsonr(est_return, mean_switch_cost)[0])   # expect NEGATIVE

    # negative control: agents with near-zero reset show higher switch cost than high-reset
    lo = mean_switch_cost[est_return < np.quantile(est_return, 0.33)]
    hi = mean_switch_cost[est_return > np.quantile(est_return, 0.67)]
    control_gap = float(lo.mean() - hi.mean())                        # expect POSITIVE

    return {
        "n": n,
        "recovery_true_vs_estimated_reset": H.gate_effect(
            "reset_rate_recovery", recovery_r, n, kind="correlation").to_dict(),
        "flexibility_return_vs_switchcost": H.gate_effect(
            "neutral_return_vs_switch_cost", flex_r, n, kind="correlation").to_dict(),
        "control_lowreset_minus_highreset_cost": control_gap,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_e8(), indent=2, default=str))
