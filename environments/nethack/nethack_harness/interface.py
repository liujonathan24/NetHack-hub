"""Typed Action layer over the engine's raw ``NetHackInterface``.

This lives in the Hub (not the engine) because the typed action set and its
dispatch are derived from the Hub's skill registry — the engine stays a pure
substrate (see ``nethack_interface``, layer 1). ``TypedNetHackInterface``
subclasses the engine's ``NetHackInterface`` and adds an ``Action`` branch to
``step`` that runs a skill via the registry; ``RawAction`` / bare-int stepping
is inherited unchanged.

Consumers that want typed actions import from here::

    from nethack_interface import RawAction, Observation          # engine
    from nethack_harness.interface import Action, action_spec, TypedNetHackInterface
"""
from __future__ import annotations
from dataclasses import dataclass, field

from nethack_interface import NetHackInterface  # engine (layer 1)


@dataclass
class Action:
    name: str
    args: dict = field(default_factory=dict)


def action_spec() -> dict:
    """name -> arg schema, sourced from the live skill registry (no drift)."""
    from nethack_harness.tools.skills import registry
    return dict(registry.all_schemas())


class TypedNetHackInterface(NetHackInterface):
    """Engine raw interface + typed ``Action`` / skill dispatch.

    ``step(Action(...))`` runs the named skill through the registry and steps the
    resulting NLE actions (behavioral parity with the harness's env_response,
    which calls ``_to_action_indices`` before stepping). ``step(RawAction(i))``
    and ``step(int)`` fall through to the engine base class.
    """

    def step(self, action):
        if isinstance(action, Action):
            from nethack_harness.tools.skills import registry
            from nethack_harness.helpers import _to_action_indices

            res = registry.call(action.name, self._env, self._structured, **action.args)
            total = 0.0
            term = trunc = False
            info = {"feedback": res.feedback}
            for idx in _to_action_indices(self._env, res.actions):
                self._raw, r, term, trunc, _info2 = self._env.step(idx)
                total += float(r)
                if term or trunc:
                    break
            return self._shape(), total, bool(term or trunc), info
        return super().step(action)
