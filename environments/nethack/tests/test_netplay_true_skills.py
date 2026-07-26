"""Tests for the vendored NetPlay skill layer (skill_set="netplay_true").

Upstream: github.com/CommanderCero/NetPlay @ 6acb90d865411f28d440e042372d9034f971e54a
See environments/nethack/vendor/PROVENANCE.md.
"""

import pathlib
import sys

import pytest

_ENV_NETHACK = pathlib.Path(__file__).resolve().parents[1]
if str(_ENV_NETHACK) not in sys.path:
    sys.path.insert(0, str(_ENV_NETHACK))

from nethack_harness.tools.skills import SkillResult, registry  # noqa: E402


# --------------------------------------------------------------------------
# Regression: the registry must not drop closed-loop bookkeeping.
# --------------------------------------------------------------------------

def test_registry_preserves_pre_executed_when_stripping_unknown_args():
    """A closed-loop skill called with a stray arg must not be re-executed.

    `SkillRegistry.call` filters kwargs it cannot pass through, and used to
    rebuild the SkillResult with only four fields -- silently dropping
    `pre_executed`/`pre_reward`/`final_obs`/`pre_terminated`/`pre_truncated`.
    For a skill that already stepped the env (every NetPlay skill does), losing
    `pre_executed` makes the harness replay `actions` on top of the steps the
    skill already took.
    """
    sentinel = object()

    @registry.register("_test_closed_loop", schema={"description": "", "parameters": {}})
    def _skill(env, obs):
        return SkillResult(
            actions=[],
            feedback="done",
            pre_executed=True,
            pre_reward=3.5,
            final_obs=sentinel,
            pre_terminated=True,
            pre_truncated=True,
        )

    try:
        res = registry.call("_test_closed_loop", None, None, bogus_arg=1)
        assert res.pre_executed is True
        assert res.pre_reward == 3.5
        assert res.final_obs is sentinel
        assert res.pre_terminated is True
        assert res.pre_truncated is True
        assert "ignored unknown args" in res.feedback
    finally:
        registry._skills.pop("_test_closed_loop", None)
        registry._schemas.pop("_test_closed_loop", None)
