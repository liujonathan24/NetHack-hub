"""`verifiers` where it is installed; otherwise stand-ins for the few `vf` names
the game box uses (`StatefulToolEnv`, `Rubric`, `reward`), since it plays games
without a verifiers rollout.
"""

from __future__ import annotations

try:
    import verifiers as vf
except ModuleNotFoundError as exc:
    # An installed verifiers that fails to import must still fail loudly.
    if exc.name != "verifiers":
        raise
    HAVE_VERIFIERS = False
else:
    HAVE_VERIFIERS = True


if not HAVE_VERIFIERS:

    class StatefulToolEnv:
        """As in `verifiers.Environment`, every keyword argument becomes an attribute."""

        def __init__(self, *, max_turns: int = 10, **kwargs) -> None:
            self.max_turns = max_turns
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Rubric:
        def __init__(self, funcs=None, weights=None, **kwargs) -> None:
            self.funcs = list(funcs or [])
            self.weights = weights

    def reward(func=None, weight: float = 1.0, priority: int = 0, stage: str = "rollout"):
        """Sets the same attributes as `verifiers.reward`."""

        def decorator(f):
            f.reward = True
            f.reward_priority = priority
            f.reward_stage = stage
            f.reward_weight = weight
            return f

        return decorator if func is None else decorator(func)

    class _Verifiers:
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
