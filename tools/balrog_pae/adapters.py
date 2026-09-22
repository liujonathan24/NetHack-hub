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
        elif hasattr(v, "tobytes") and hasattr(v, "mode") and hasattr(v, "size"):
            # PIL Image (the VLM observation). repr() would embed the object's
            # memory address, making digests differ between processes - and
            # between two restores in the same process - for identical pixels.
            h.update(str(v.mode).encode() + repr(tuple(v.size)).encode() + v.tobytes())
        elif isinstance(v, (list, tuple)):
            for item in v:
                feed(item)
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
    env_patches = ("crafter_balance_chunk_sorted", "crafter_seed_pinned_to_episode_seed")

    def __init__(self, task, cfg):
        super().__init__(task, cfg)
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "balrog_ckpt"))
        import crafter_ckpt

        crafter_ckpt.apply_determinism_patch()
        self._ck = crafter_ckpt

    def reset(self, seed: int):
        """Reset to the world named by ``seed`` (disclosed patch #2).

        BALROG ships ``envs.crafter_kwargs.seed: null``, so ``crafter.Env.__init__``
        draws its world seed from the *global* numpy RNG at construction time; and
        ``CrafterLanguageWrapper.reset()`` takes no seed argument, so the seed the
        evaluator passes never reaches Crafter at all.  Stock BALROG therefore
        plays a different, uncontrolled world in every process - which would make
        "base seed k" and "PAE seed k" different episodes and void the paired
        comparison this experiment rests on.

        Crafter derives the world seed as ``hash((_seed, _episode)) % (2**31-1)``;
        tuples of ints hash identically in every CPython process (PYTHONHASHSEED
        only perturbs str/bytes), so pinning ``_seed = seed`` and ``_episode = 0``
        immediately before the reset makes ``--seed k`` name one fixed world.
        Applied in EVERY arm, ``--base-only`` included, and disclosed in
        ``summary.json`` as ``crafter_seed_pinned_to_episode_seed``.
        """
        _, core = self._ck.inner_crafter(self.env)
        core._seed = int(seed)
        core._episode = 0
        return super().reset(seed)

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
    """BALROG's three TextWorld games.

    Stack: ``EnvWrapper > GymV21CompatibilityV0 > TextWorldWrapper >
    TextworldGymEnv > SyncBatchEnv > Filter > Limit > GenericEnvironment >
    TWInform7 > StateTracking > Inform7Data > GameData > {JerichoEnv (.z8) |
    GitGlulxEnv (.ulx)}``.

    Restore (verified in ``games/textworld/verify_restore.py``):

    * ``the_cooking_game`` (.z8) -- native: ``jericho.FrotzEnv.get_state()/
      set_state()`` plus a deepcopy of every Python wrapper's mutable state
      (``Limit.nb_steps``, the Inform7 state tracker, ``SyncBatchEnv.last`` --
      a done latch that short-circuits ``step()`` if forgotten).
    * ``treasure_hunter`` / ``coin_collector`` (.ulx) -- replay: ``GitGlulxEnv``
      is a ``git-glulx-ml`` subprocess over a socket with no state export
      (``copy()`` -> ``NotImplementedError``; in-game ``save`` kills the
      interpreter). The games are deterministic, so reset + replay of the
      command prefix reproduces the state exactly.

    Seeding. BALROG's ``TextWorldFactory`` picks the *game file* with
    ``env_ids[task][seed % 25]`` when ``envs.env_kwargs.seed`` is set, and
    otherwise cycles a global counter -- which would make "seed 0" and "seed 1"
    the same game in a fresh process. This adapter therefore binds the game
    file to the episode seed: ``make()``/``reset()`` (re)build the env with
    ``envs.env_kwargs.seed = seed``. Nothing else about the env is touched.
    ``env.reset(seed=)`` itself only seeds the gym wrapper's unused RNG; the
    games carry no run-time randomness.
    """

    env_name = "textworld"
    aux_label = "score / max_score (live)"
    #: ``env_patches`` is a property below: the live-progression change is
    #: disclosed there, and BALROG's own ``get_stats()["progression"]`` rides
    #: along so it lands in summary.json (which the loop writes at episode end).
    _SKIP = {
        "_wrapped_env", "_game", "_inform7", "_jericho", "_gamefile", "gamefile",
        "request_infos", "_tracked_infos", "_process", "_names_struct",
        "_last_backend", "max_episode_steps", "_seed",
    }

    def __init__(self, task, cfg):
        super().__init__(task, cfg)
        self.log: list[dict] = []
        self._built_seed = None
        self._score = 0.0
        self._max_score = 1.0
        self._won = False

    # -- env construction, with the game file bound to the seed -------------
    def _build(self, seed: int):
        from balrog.environments import make_env

        if self.env is not None:
            try:
                self.env.close()
            except Exception:  # noqa: BLE001
                pass
        self.cfg.envs.env_kwargs.seed = int(seed)
        self.env = make_env(self.env_name, self.task, self.cfg)
        self._built_seed = int(seed)
        return self.env

    def make(self):
        seed = self.cfg.envs.env_kwargs.get("seed", None)
        return self._build(0 if seed is None else int(seed))

    def gamefile(self) -> str:
        """Absolute path of the game file this env is bound to (after reset)."""
        try:
            _, objs = self._chain()
        except Exception:  # noqa: BLE001
            return ""
        base = objs[-1]
        return str(getattr(base, "gamefile", None) or getattr(base, "_gamefile", "") or "")

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
        if self._built_seed != int(seed):
            self._build(seed)
        out = super().reset(seed)
        self.log = []
        self._score, self._max_score, self._won = 0.0, 1.0, False
        return out

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        self._track(info)
        if (term or trunc) and isinstance(info, dict):
            # lands in attempts.jsonl -> the per-episode record of BALROG's own
            # number next to the live one the loop reports.
            info.setdefault(
                "end_status",
                f"score={self._score:.0f}/{self._max_score:.0f} won={self._won} "
                f"balrog_get_stats_progression={self.balrog_progression():.6f}",
            )
        return obs, reward, term, trunc, info

    def _track(self, info):
        if not info:
            return
        if info.get("max_score"):
            self._max_score = float(info["max_score"])
        if info.get("score") is not None:
            self._score = float(info["score"])
        self._won = bool(info.get("won"))

    def record(self, validated, completion):
        self.log.append({"action": validated, "completion": completion})

    # -- checkpointing ------------------------------------------------------
    def env_snapshot(self) -> dict:
        common = {
            "seed": self.seed,
            "steps": self.steps,
            "log": copy.deepcopy(self.log),
            "score": self._score,
            "max_score": self._max_score,
            "won": self._won,
        }
        if not self._native():
            return {"kind": "replay", **common}
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
            **common,
        }

    def env_restore(self, snap: dict):
        if snap["kind"] == "replay":
            if self._built_seed != int(snap["seed"]):
                self._build(snap["seed"])
            random.seed(snap["seed"])
            np.random.seed(snap["seed"])
            obs, info = self.env.reset(seed=snap["seed"])
            self.steps = 0
            self.total_env_steps += 1
            self.seed = snap["seed"]
            self._score, self._max_score, self._won = 0.0, 1.0, False
            digests = [obs_digest(obs)]
            feedback = self.cfg.eval.feedback_on_invalid_action
            for entry in snap["log"]:
                obs, reward, term, trunc, info = self.step(entry["action"])
                # TextWorld's language_action_space accepts everything, so this
                # branch is currently dead; kept so the replayed prompt would
                # still match if BALROG ever constrained the action space.
                if feedback and entry["action"] != entry["completion"]:
                    obs["text"]["long_term_context"] = (
                        f"\n\nYour previous output did not contain a valid action. "
                        f"Defaulted to action: {entry['action']}\n\nObservation:\n"
                        + obs["text"]["long_term_context"]
                    )
                digests.append(obs_digest(obs))
            self.log = copy.deepcopy(snap["log"])
            self._restore_scalars(snap)
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
        self._restore_scalars(snap)
        return None, None, None

    def _restore_scalars(self, snap):
        self._score = float(snap.get("score", 0.0))
        self._max_score = float(snap.get("max_score", 1.0) or 1.0)
        self._won = bool(snap.get("won", False))
        self.steps = int(snap.get("steps", self.steps))

    def state_digest(self):
        if not self._native():
            return None
        import pickle as _pickle

        _, objs = self._chain()
        st = objs[-1]._jericho.get_state()
        return hashlib.sha256(_pickle.dumps(st, protocol=4)).hexdigest()[:16]

    # -- progress -----------------------------------------------------------
    def progression(self) -> float:
        """BALROG's progression formula, evaluated at **every** step.

        ``TextWorldWrapper.step`` computes
        ``max(score / max_score, 1.0 if won else 0.0)`` but only inside
        ``if done:``, so ``get_stats()["progression"]`` reads 0.0 for the whole
        episode and is written exactly once.  That is fine for BALROG (it makes
        one env per episode and reads the number at the end) but it blinds a
        checkpointed loop: ``pae.py``'s checkpoint trigger and plateau guard
        both read this method, so with the stock behaviour no TextWorld score
        gain would ever fire a checkpoint or reset the plateau.

        This returns the *same formula on the same numbers*, live.  At ``done``
        the two are equal by construction, and
        ``games/textworld/test_progression_parity.py`` asserts it on a winning
        trace of each game.  The deviation is disclosed in ``env_patches`` and
        applies identically in every arm, ``--base-only`` included.
        """
        return max(float(self._score) / float(self._max_score or 1.0),
                   1.0 if self._won else 0.0)

    def balrog_progression(self) -> float:
        """BALROG's own ``get_stats()["progression"]``, untouched."""
        try:
            return float(self.env.get_stats().get("progression", 0.0))
        except Exception:  # noqa: BLE001
            return 0.0

    @property
    def env_patches(self) -> tuple[str, ...]:
        return (
            "textworld_live_progression",
            f"balrog_get_stats_progression={self.balrog_progression():.6f}",
        )

    def aux_progress(self) -> float:
        """Dense exploration proxy: the same live score fraction."""
        return float(self._score) / float(self._max_score or 1.0)

    def summary(self) -> str:
        return (
            f"step {self.steps}/{self.max_steps}, score {self._score:.0f}/{self._max_score:.0f} "
            f"({self.aux_progress():.2f}), won={self._won}"
        )


ADAPTERS = {"minihack": MiniHackAdapter, "crafter": CrafterAdapter, "textworld": TextWorldAdapter}


def make_adapter(env_name: str, task: str, cfg):
    return ADAPTERS[env_name](task, cfg)
