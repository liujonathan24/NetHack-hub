"""Per-game checkpoint adapters over BALROG's own env stack.

Every adapter exposes the same contract to the PAE loop:

    make()                      build BALROG's env (make_env, untouched)
    reset(seed) -> obs, info    fresh episode
    step(action) -> obs, r, term, trunc, info
    env_snapshot() -> dict      whatever restores the ENV state
    env_restore(snap) -> obs    put the env back, return the observation the
                                player was looking at at snapshot time
    progression() -> float      BALROG's own metric (env.get_stats())
    aux_progress() -> float     dense, game-specific exploration proxy used only
                                to decide *when* to checkpoint and what to show
                                the orchestrator; never reported as the metric
    summary() -> str            one line of state for the orchestrator ledger

The restore methods are the ones verified in ``tools/balrog_ckpt``
(see its REPORT.md): MiniHack = replay of the action prefix from the episode's
*effective* seeds, Crafter = pickle of the inner env plus the determinism
patch, TextWorld = Jericho get_state/set_state for the .z8 game and replay for
the Glulx ones.
"""
from __future__ import annotations

import copy
import hashlib
import random
from typing import Any

import numpy as np


def obs_digest(obs: dict) -> str:
    h = hashlib.sha256()

    def feed(v):
        if isinstance(v, np.ndarray):
            h.update(str(v.dtype).encode() + str(v.shape).encode() + np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode())
                feed(v[k])
        elif v is None:
            h.update(b"None")
        else:
            h.update(repr(v).encode())

    feed(obs)
    return h.hexdigest()[:16]


class BaseAdapter:
    env_name = ""
    #: label for the dense exploration proxy (NOT a reported metric)
    aux_label: str = "none"
    #: engine patches this adapter installs. Applied IDENTICALLY in every arm
    #: (base included) so the arms stay comparable; disclosed in summary.json.
    env_patches: tuple[str, ...] = ()

    def __init__(self, task: str, cfg):
        self.task = task
        self.cfg = cfg
        self.env = None
        self.seed = None
        self.steps = 0          # env steps since the last reset
        self.total_env_steps = 0  # every env step ever taken, replays included

    # -- lifecycle ---------------------------------------------------------
    def make(self):
        from balrog.environments import make_env

        self.env = make_env(self.env_name, self.task, self.cfg)
        return self.env

    def reset(self, seed: int):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        obs, info = self.env.reset(seed=seed)
        self.steps = 0
        self.total_env_steps += 1
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self.steps += 1
        self.total_env_steps += 1
        return out

    @property
    def max_steps(self):
        return self.env.max_steps

    def instruction_prompt(self, obs):
        return self.env.get_instruction_prompt()

    def progression(self) -> float:
        return float(self.env.get_stats().get("progression", 0.0))

    def aux_progress(self) -> float:
        return 0.0

    def summary(self) -> str:
        return ""

    def state_digest(self):
        """Optional canonical digest of the env state, for restore parity checks."""
        return None

    def env_snapshot(self) -> dict:
        raise NotImplementedError

    def env_restore(self, snap) -> Any:
        raise NotImplementedError

    def close(self):
        if self.env is not None:
            self.env.close()


# ---------------------------------------------------------------------------
# MiniHack: replay from the episode's effective seeds
# ---------------------------------------------------------------------------
class MiniHackAdapter(BaseAdapter):
    """Restore = reset with the effective (core, disp, reseed) seeds + replay.

    Verified byte-identical in tools/balrog_ckpt/verify_minihack_replay.py
    (Quest-Easy, Boxoban-Medium; 3/3 replays).  Two traps handled here:
      * BALROG's NLELanguageWrapper latches ``done`` and reset() never clears it;
      * ``env.reset(seed=s)`` draws ``disp`` from SystemRandom, so the seed the
        evaluator passes does not by itself pin the episode - read the effective
        seeds back with get_seeds() and re-apply them before every replay.
    """

    env_name = "minihack"
    aux_label = "distinct (dlvl,x,y) cells seen"

    def __init__(self, task, cfg):
        super().__init__(task, cfg)
        self.effective_seeds = None
        self.log: list[dict] = []   # {"action": validated, "completion": raw}
        self._visited: set = set()
        self._max_depth = 0

    def _inner(self):
        return self.env.env.gym_env.unwrapped

    def _lang(self):
        e = self.env.env.gym_env
        while e is not None and not ("done" in vars(e) and "progress" in vars(e)):
            e = vars(e).get("env")
        return e

    def instruction_prompt(self, obs):
        return self.env.get_instruction_prompt()

    def reset(self, seed: int):
        obs, info = super().reset(seed)
        self.effective_seeds = [int(x) if not isinstance(x, bool) else x for x in self._inner().get_seeds()[:3]]
        self.log = []
        self._visited = set()
        self._max_depth = 0
        self._track(obs)
        return obs, info

    def _full_reset(self):
        """reset() + clear BALROG's sticky done latch."""
        out = self.env.reset()
        lw = self._lang()
        if lw is not None:
            lw.done = False
        return out

    def _track(self, obs):
        raw = obs.get("obs") or {}
        bl = raw.get("blstats")
        if bl is not None:
            x, y, depth = int(bl[0]), int(bl[1]), int(bl[12])
            self._visited.add((depth, x, y))
            self._max_depth = max(self._max_depth, depth)

    def record(self, validated, completion):
        self.log.append({"action": validated, "completion": completion})

    def step(self, action):
        out = super().step(action)
        self._track(out[0])
        return out

    def env_snapshot(self) -> dict:
        return {
            "kind": "replay",
            "seed": self.seed,
            "effective_seeds": list(self.effective_seeds),
            "log": copy.deepcopy(self.log),
        }

    def env_restore(self, snap: dict):
        """Replay the prefix; returns (obs, info, per-step observation digests)."""
        core, disp, reseed = snap["effective_seeds"]
        random.seed(snap["seed"])
        np.random.seed(snap["seed"])
        self._inner().seed(int(core), int(disp), bool(reseed))
        obs, info = self._full_reset()
        self.steps = 0
        self.total_env_steps += 1
        self.seed = snap["seed"]
        self.effective_seeds = list(snap["effective_seeds"])
        self._visited = set()
        self._max_depth = 0
        self._track(obs)
        digests = [obs_digest(obs)]
        feedback = self.cfg.eval.feedback_on_invalid_action
        for entry in snap["log"]:
            obs, reward, term, trunc, info = self.step(entry["action"])
            # the evaluator rewrites the observation text when the model's raw
            # completion was not a legal action; replay must do the same or the
            # restored prompt would differ from the one the player saw.
            if feedback and entry["action"] != entry["completion"]:
                obs["text"]["long_term_context"] = (
                    f"\n\nYour previous output did not contain a valid action. "
                    f"Defaulted to action: {entry['action']}\n\nObservation:\n"
                    + obs["text"]["long_term_context"]
                )
            digests.append(obs_digest(obs))
        self.log = copy.deepcopy(snap["log"])
        return obs, info, digests

    def aux_progress(self) -> float:
        return float(len(self._visited))

    def summary(self) -> str:
        raw = self.env.env.gym_env.last_obs if hasattr(self.env.env.gym_env, "last_obs") else None
        bl = None
        try:
            inner = self._inner()
            obs = inner.last_observation
            bl = obs[inner._observation_keys.index("blstats")]
        except Exception:
            pass
        if bl is None:
            return f"step {self.steps}, cells seen {len(self._visited)}"
        return (
            f"pos=({int(bl[0])},{int(bl[1])}) dlvl={int(bl[12])} hp={int(bl[10])}/{int(bl[11])} "
            f"time={int(bl[20])} cells_seen={len(self._visited)}"
        )


# ---------------------------------------------------------------------------
# Crafter: pickle of the inner env (plus the determinism patch)
# ---------------------------------------------------------------------------
class CrafterAdapter(BaseAdapter):
    """Crafter, snapshotted by pickling the inner ``crafter.Env``.

    ``crafter_ckpt.apply_determinism_patch`` sorts the per-chunk object set
    before ``_balance_object`` picks a creature to despawn.  Without it stock
    Crafter is not reproducible across processes and a restored copy diverges
    from the original within ~30 steps (balrog_ckpt/REPORT.md: 1/5 seeds pass
    unpatched, 5/5 patched).  It reorders which object an already-arbitrary
    choice lands on; it does not touch the RNG stream, rewards or achievements.
    It is applied in EVERY arm, --base-only included, so base and PAE play the
    same game, and is disclosed as ``env_patches`` in summary.json.
    """

    env_name = "crafter"
    aux_label = "achievements unlocked (of 22)"
    env_patches = ("crafter_balance_chunk_sorted",)

    def __init__(self, task, cfg):
        super().__init__(task, cfg)
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "balrog_ckpt"))
        import crafter_ckpt

        crafter_ckpt.apply_determinism_patch()
        self._ck = crafter_ckpt

    def env_snapshot(self) -> dict:
        return {"kind": "pickle", "blob": self._ck.snapshot_pickle(self.env), "steps": self.steps}

    def env_restore(self, snap: dict):
        self._ck.restore_pickle(self.env, snap["blob"])
        self.steps = snap["steps"]
        return None, None, None  # the observation is carried in the checkpoint

    def state_digest(self):
        return self._ck.state_digest(self._ck.inner_crafter(self.env)[1])

    def aux_progress(self) -> float:
        return float(self.env.env.gym_env.score_tracker)

    def summary(self) -> str:
        lang = self.env.env.gym_env
        done = sorted(k for k, v in (getattr(lang, "achievements", None) or {}).items() if v)
        return f"step {self.steps}, achievements {len(done)}: {', '.join(done) or 'none'}"


# ---------------------------------------------------------------------------
# TextWorld: Jericho save/restore (.z8) or replay (.ulx)
# ---------------------------------------------------------------------------
class TextWorldAdapter(BaseAdapter):
    env_name = "textworld"
    aux_label = "score / max_score (same as progression)"
    _SKIP = {
        "_wrapped_env", "_game", "_inform7", "_jericho", "_gamefile", "gamefile",
        "request_infos", "_tracked_infos", "_process", "_names_struct",
        "_last_backend", "max_episode_steps", "_seed",
    }

    def __init__(self, task, cfg):
        super().__init__(task, cfg)
        self.log: list[dict] = []

    def _chain(self):
        gymenv = self.env.env.gym_env.env
        e = gymenv.batch_env.envs[0]
        out = []
        while e is not None:
            out.append(e)
            e = getattr(e, "_wrapped_env", None)
        return gymenv, out

    def _native(self) -> bool:
        _, objs = self._chain()
        return hasattr(objs[-1], "_jericho")

    def reset(self, seed):
        out = super().reset(seed)
        self.log = []
        return out

    def record(self, validated, completion):
        self.log.append({"action": validated, "completion": completion})

    def env_snapshot(self) -> dict:
        if not self._native():
            return {"kind": "replay", "seed": self.seed, "log": copy.deepcopy(self.log)}
        gymenv, objs = self._chain()
        return {
            "kind": "jericho",
            "jericho": objs[-1]._jericho.get_state(),
            "wrappers": copy.deepcopy([{k: v for k, v in vars(o).items() if k not in self._SKIP} for o in objs]),
            "gym": copy.deepcopy(
                {"last_commands": gymenv.last_commands, "obs": gymenv.obs, "batch_last": gymenv.batch_env.last}
            ),
            "progression": self.env.env.gym_env.progression,
            "failed_candidates": list(self.env.failed_candidates),
            "steps": self.steps,
            "log": copy.deepcopy(self.log),
        }

    def env_restore(self, snap: dict):
        if snap["kind"] == "replay":
            random.seed(snap["seed"])
            np.random.seed(snap["seed"])
            obs, info = self.env.reset(seed=snap["seed"])
            self.steps = 0
            digests = [obs_digest(obs)]
            for entry in snap["log"]:
                obs, reward, term, trunc, info = self.step(entry["action"])
                digests.append(obs_digest(obs))
            self.log = copy.deepcopy(snap["log"])
            return obs, info, digests
        gymenv, objs = self._chain()
        objs[-1]._jericho.set_state(snap["jericho"])
        for o, saved in zip(objs, copy.deepcopy(snap["wrappers"])):
            for k, v in saved.items():
                setattr(o, k, v)
        g = copy.deepcopy(snap["gym"])
        gymenv.last_commands, gymenv.obs = g["last_commands"], g["obs"]
        gymenv.batch_env.last = g["batch_last"]
        self.env.env.gym_env.progression = snap["progression"]
        self.env.failed_candidates = list(snap["failed_candidates"])
        self.steps = snap["steps"]
        self.log = copy.deepcopy(snap["log"])
        return None, None, None

    def state_digest(self):
        if not self._native():
            return None
        _, objs = self._chain()
        import pickle as _pickle

        st = objs[-1]._jericho.get_state()
        return hashlib.sha256(_pickle.dumps(st, protocol=4)).hexdigest()[:16]

    def aux_progress(self) -> float:
        return float(self.env.env.gym_env.progression)

    def summary(self) -> str:
        return f"step {self.steps}, score fraction {self.env.env.gym_env.progression:.2f}"


ADAPTERS = {"minihack": MiniHackAdapter, "crafter": CrafterAdapter, "textworld": TextWorldAdapter}


def make_adapter(env_name: str, task: str, cfg):
    return ADAPTERS[env_name](task, cfg)
