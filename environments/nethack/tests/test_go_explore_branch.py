"""Step 6: go_explore exercises the engine's branch() primitive.

``run_go_explore(branch_probe=True)`` probes each returned cell's continuation
divergence via ``EngineEnv.branch()`` and biases selection by it — a real
research path that actually invokes the fork's snapshot/branch differentiator
(previously branch() was reachable only from tests). Drives a real engine (slow).
"""
from __future__ import annotations
import pathlib
import sys

# approaches/ lives at the Hub repo root; nethack_core is the engine package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from approaches.go_explore.go_explore import run_go_explore, _branch_divergence
from nethack_core.engine_env import EngineEnv


def test_branch_probe_is_exercised_and_run_is_healthy():
    result, env = run_go_explore(
        iterations=6, explore_steps=4, seed=2, verbose=False,
        branch_probe=True, branch_n=4, branch_horizon=12,
    )
    try:
        # engine.branch() was invoked once per iteration (the real research path)
        assert result.n_branch_probes == 6
        assert 0.0 <= result.mean_branch_divergence <= 1.0
        # the search still functions
        assert result.n_cells >= 1
        assert result.max_depth >= 1
    finally:
        env.close()


def test_branch_probe_off_by_default():
    result, env = run_go_explore(
        iterations=4, explore_steps=4, seed=2, verbose=False,
    )
    try:
        assert result.n_branch_probes == 0
        assert result.mean_branch_divergence == 0.0
    finally:
        env.close()


def test_branch_divergence_helper_is_bounded():
    env = EngineEnv()
    env.reset(seeds=(2, 2))
    h = env.snapshot()
    try:
        d = _branch_divergence(env, h, n=4, horizon=10)
        assert 0.0 <= d <= 1.0
    finally:
        env.free_snapshot(h)
