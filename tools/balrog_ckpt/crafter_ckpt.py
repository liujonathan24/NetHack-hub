"""Snapshot/restore helpers for BALROG's Crafter env, plus the determinism patch
that makes restored copies replay identically.

Why a patch is needed
---------------------
crafter.Env.step() calls _balance_chunk(chunk, objs) for every chunk, where
`objs` is a Python *set* of object instances (World._chunks is a
defaultdict(set)). _balance_object() then does

    creatures = [obj for obj in objs if isinstance(obj, cls)]
    ...
    obj = creatures[random.randint(0, len(creatures))]   # despawn

Set iteration order depends on id()-based hashes, so any restored copy (pickle,
deepcopy, or an explicit world copy) orders `creatures` differently from the
original -- and from every other restore. The RNG stream is identical, but the
*object* picked for despawn (and the spawn/despawn decisions that follow) is
not, so branches diverge after a few steps. (The same effect makes stock
Crafter non-reproducible across processes.)

The patch sorts `objs` by a stable key (position + class name; at most one
object per cell) before the original balancing code runs. It does not change
the RNG stream, only makes an already-arbitrary choice deterministic.
"""
from __future__ import annotations

import copy
import pickle

import crafter

_PATCHED = False


def apply_determinism_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    orig = crafter.Env._balance_chunk

    def _balance_chunk(self, chunk, objs):
        objs = sorted(objs, key=lambda o: (int(o.pos[0]), int(o.pos[1]), type(o).__name__))
        return orig(self, chunk, objs)

    crafter.Env._balance_chunk = _balance_chunk
    _PATCHED = True


# ---------------------------------------------------------------------------
# wrapper-stack navigation (BALROG): EnvWrapper -> GymV21CompatibilityV0 ->
# CrafterLanguageWrapper -> crafter.Env
# ---------------------------------------------------------------------------
def inner_crafter(env):
    lang = env.env.gym_env
    return lang, lang.env


_SKIP = ("_np_random",)  # gym 0.23 RandomNumberGenerator; unused by crafter, un-picklable on numpy>=2


def _wrapper_state(env, lang) -> dict:
    return {
        "score_tracker": lang.score_tracker,
        "achievements": copy.deepcopy(lang.achievements),
        "failed_candidates": list(env.failed_candidates),
    }


def _set_wrapper_state(env, lang, s: dict) -> None:
    lang.score_tracker = s["score_tracker"]
    lang.achievements = copy.deepcopy(s["achievements"])
    env.failed_candidates = list(s["failed_candidates"])


# --- A: deepcopy of the inner env (with _np_random stripped) ------------------
def snapshot_deepcopy(env):
    lang, core = inner_crafter(env)
    state = {k: v for k, v in core.__dict__.items() if k not in _SKIP}
    return {"core": copy.deepcopy(state), "wrap": _wrapper_state(env, lang)}


def restore_deepcopy(env, snap) -> None:
    lang, core = inner_crafter(env)
    keep = {k: v for k, v in core.__dict__.items() if k in _SKIP}
    core.__dict__.clear()
    core.__dict__.update(copy.deepcopy(snap["core"]))  # copy again: saved snapshot stays pristine
    core.__dict__.update(keep)
    _set_wrapper_state(env, lang, snap["wrap"])


# --- B: pickle bytes of the inner env, restored in place -----------------------
def snapshot_pickle(env) -> bytes:
    lang, core = inner_crafter(env)
    state = {k: v for k, v in core.__dict__.items() if k not in _SKIP}
    return pickle.dumps({"core": state, "wrap": _wrapper_state(env, lang)}, protocol=pickle.HIGHEST_PROTOCOL)


def restore_pickle(env, blob: bytes) -> None:
    lang, core = inner_crafter(env)
    payload = pickle.loads(blob)
    keep = {k: v for k, v in core.__dict__.items() if k in _SKIP}
    core.__dict__.clear()
    core.__dict__.update(payload["core"])
    core.__dict__.update(keep)
    _set_wrapper_state(env, lang, payload["wrap"])


# --- C: explicit component copy (world + player + RNG + counters) --------------
def snapshot_components(env) -> dict:
    lang, core = inner_crafter(env)
    return {
        "world": copy.deepcopy(core._world),  # objects (incl. player) + chunks + np RandomState
        "rng": core._world.random.get_state(),
        "step": core._step,
        "episode": core._episode,
        "last_health": core._last_health,
        "unlocked": set(core._unlocked),
        "wrap": _wrapper_state(env, lang),
    }


def restore_components(env, snap: dict) -> None:
    from crafter import objects

    lang, core = inner_crafter(env)
    world = copy.deepcopy(snap["world"])
    world.random.set_state(snap["rng"])
    core._world = world
    players = [o for o in world._objects if isinstance(o, objects.Player)]
    assert len(players) == 1, players
    core._player = players[0]
    core._step = snap["step"]
    core._episode = snap["episode"]
    core._last_health = snap["last_health"]
    core._unlocked = set(snap["unlocked"])
    core._local_view._world = world  # views keep a world reference
    core._sem_view._world = world
    _set_wrapper_state(env, lang, snap["wrap"])


def state_digest(core) -> str:
    """Canonical digest of a crafter.Env's game state (independent of object
    identity / set iteration order, unlike pickle bytes)."""
    import hashlib

    import numpy as np

    h = hashlib.sha256()
    w = core._world
    h.update(np.ascontiguousarray(w._mat_map).tobytes())
    h.update(np.ascontiguousarray(w._obj_map).tobytes())
    h.update(repr(float(w.daylight)).encode())
    st = w.random.get_state()
    h.update(st[0].encode() + np.ascontiguousarray(st[1]).tobytes() + repr(st[2:]).encode())
    from crafter import objects as _objects

    def canon(v):  # object references (e.g. Zombie.player) would repr() as memory addresses
        if isinstance(v, _objects.Object):
            return (type(v).__name__, int(v.pos[0]), int(v.pos[1]))
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return sorted((k, canon(x)) for k, x in v.items())
        if isinstance(v, (list, tuple)):
            return [canon(x) for x in v]
        return v

    objs = []
    for o in w._objects:
        if o is None:
            continue
        fields = {k: canon(v) for k, v in vars(o).items() if k not in ("world", "random")}
        objs.append((type(o).__name__, int(o.pos[0]), int(o.pos[1]), repr(sorted(fields.items(), key=lambda kv: kv[0]))))
    for item in sorted(objs):
        h.update(repr(item).encode())
    h.update(repr((core._step, core._episode, core._last_health, sorted(core._unlocked))).encode())
    return h.hexdigest()[:16]


APPROACHES = {
    "A_deepcopy_inner": (snapshot_deepcopy, restore_deepcopy),
    "B_pickle_inner_inplace": (snapshot_pickle, restore_pickle),
    "C_explicit_world_player_rng": (snapshot_components, restore_components),
}
