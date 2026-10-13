"""`verifiers` where it is installed; otherwise the three names the game box runs.

The residency game image runs `python -m nethack_toolset.game`, which builds the
env with `nethack.load_environment` and then only calls `setup_state`,
`_apply_tool_call` and `_render_obs_text`. None of that is a verifiers rollout,
so that image does not install verifiers. The only `vf.*` code that still runs
there is the env's base class, the `vf.Rubric(...)` built in `load_environment`
and the `@vf.reward` decorators in `helpers`; without verifiers those resolve to
the stand-ins below, which keep what that path reads and nothing else.

Wherever verifiers is installed, `vf` is the verifiers module itself and
nothing in this file is used.
"""

from __future__ import annotations

try:
    import verifiers as vf
except ModuleNotFoundError as exc:
    # Only a missing verifiers selects the stand-ins: an installed verifiers that
    # fails to import must still fail loudly.
    if exc.name != "verifiers":
        raise
    HAVE_VERIFIERS = False
else:
    HAVE_VERIFIERS = True


if not HAVE_VERIFIERS:

    class StatefulToolEnv:
        """The part of verifiers' env base that `NetHackVerifiersEnv` uses outside a rollout.

        As in `verifiers.Environment`, every keyword argument becomes an
        attribute, so an env_args key the env does not name is kept, not refused.
        """

        def __init__(self, *, max_turns: int = 10, **kwargs) -> None:
            self.max_turns = max_turns
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Rubric:
        """Holds the reward functions; nothing scores them without a verifiers rollout."""

        def __init__(self, funcs=None, weights=None, **kwargs) -> None:
            self.funcs = list(funcs or [])
            self.weights = weights

    def reward(func=None, weight: float = 1.0, priority: int = 0, stage: str = "rollout"):
        """Marks a reward function the way `verifiers.reward` does, and returns it unchanged."""

        def decorator(f):
            f.reward = True
            f.reward_priority = priority
            f.reward_stage = stage
            f.reward_weight = weight
            return f

        return decorator if func is None else decorator(func)

    class _Verifiers:
        """`vf` without verifiers: the stand-ins above, and a clear error for anything else."""

        StatefulToolEnv = StatefulToolEnv
        Rubric = Rubric
        reward = staticmethod(reward)

        def __getattr__(self, name: str):
            raise AttributeError(
                f"vf.{name} needs verifiers, which is not installed here; "
                "only the game box (nethack_toolset.game) runs without it"
            )

    vf = _Verifiers()


__all__ = ["HAVE_VERIFIERS", "vf"]
