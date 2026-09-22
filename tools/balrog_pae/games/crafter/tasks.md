# Crafter — progress semantics, caps, and the PAE checkpoint trigger

Everything below is read off the BALROG checkout this experiment drives
(`/root/nld/gen-pae/BALROG`, commit `b7afe79`) and the PAE loop in
`tools/balrog_pae/`. Nothing here is an estimate unless it says "measured".

## 1. What "progression" is

`CrafterLanguageWrapper.get_stats()`
(`balrog/environments/crafter/env.py:345`) returns

```python
{"score": self.score_tracker,
 "progression": float(self.score_tracker) / 22.0,
 "achievements": self.achievements}
```

and `score_tracker` is recomputed from scratch on **every** step
(`update_progress`, `env.py:341`):

```python
self.score_tracker = 0 + sum([1.0 for k, v in info["achievements"].items() if v > 0])
```

`info["achievements"]` is `crafter`'s own per-achievement **counter** dict, so
`score_tracker` is the number of the 22 achievements unlocked **at least once**
this episode.

* **progression = (achievements unlocked) / 22**, a value in
  `{0, 1/22, 2/22, …, 1}`; one achievement = **0.04545**.
* It is monotone non-decreasing within an episode (counters never decrease), so
  "progression went up" ⟺ "a new achievement was unlocked".
* `progression == 1.0` (all 22) is what the PAE loop treats as `solved`; in
  practice it is unreachable for an LLM agent.
* The 22 achievements are exactly the list BALROG puts in the instruction
  prompt (`balrog/environments/crafter/__init__.py`): collect wood/stone/coal/
  iron/diamond/drink/sapling, place table/stone/furnace/plant, make wood|stone|
  iron pickaxe and sword, eat cow, eat plant, defeat zombie, defeat skeleton,
  wake up.
* Caveat for the paper: this is **not** the Crafter paper's "score" (the
  geometric mean of per-achievement success rates across episodes). BALROG's
  Crafter number is the mean of this per-episode achievement *fraction*. We
  report BALROG's, unchanged.

`achievements` is `None` until the first `step()` (only `reset()` has run), so
`get_stats()["achievements"]` is `None` at t0 while `progression` is already
`0.0` — handled by the adapter, which reads `progression` only.

## 2. Episode length cap

| quantity | value | where |
|---|--:|---|
| `eval.max_steps_per_episode` | `null` | `balrog/config/config.yaml` |
| `envs.crafter_kwargs.max_episode_steps` | **2000** | `balrog/config/config.yaml` |
| `CrafterLanguageWrapper.max_steps` | **2000** | set from the above, `env.py:282` |
| `crafter.Env(length=…)` | 10000 (crafter's default; BALROG never passes it) | `crafter/env.py` |
| `CrafterLanguageWrapper.default_steps` | 10000 | **dead attribute** — referenced nowhere in BALROG |

BALROG's evaluator (`balrog/evaluator.py:280`) takes
`max_steps_per_episode = env.max_steps if cfg.eval.max_steps_per_episode is None
else cfg.eval.max_steps_per_episode` and loops `for step in
range(max_steps_per_episode)`.

**The BALROG default cap for Crafter is therefore 2000 agent steps**, verified
at runtime (`adapter.max_steps == 2000`). The inner `crafter.Env` truncation at
`_step >= 10000` exists but 2000 binds first, so it never fires; the "10000"
in the earlier note is crafter's own default and the dead
`default_steps` attribute, not BALROG's cap.

Step bookkeeping detail: `CrafterLanguageWrapper.reset()` internally issues one
`Noop` (`self._step_impl(0)`) to build the first observation, so the inner
`crafter.Env._step` is already `1` when the episode is at agent-step 0. This
offsets the inner counter by one; it does not affect the 2000-step agent
horizon, which the evaluator (and `pae.Run.play`) counts itself.

Measured episode lengths on the pinned seed-0 world, **uniform-random policy**,
30 action seeds: `56, 130, 131, 131, 140, 141, 142, 145, 146, 146, 147, 148,
148, 155, 155, 156, 157, 158, 162, 169, 172, 175, 179, 188, 191, 192, 196, 199,
204, 210` — median ≈ 155, all deaths (starvation / zombie). This brackets the
earlier "150-350 steps to death" estimate from below. The LLM base episode is in
`calibration.json`.

## 3. Seeding (a bug found and fixed here — disclosed)

BALROG ships `envs.crafter_kwargs.seed: null`, so `crafter.Env.__init__` does
`seed = np.random.randint(0, 2**31-1) if seed is None else seed` — it draws its
world seed from the **global numpy RNG at construction time**. And
`CrafterLanguageWrapper.reset()` takes **no seed argument**, so the seed the
evaluator (or the PAE loop) passes to `env.reset(seed=k)` never reaches Crafter.

Consequence for stock BALROG: every process plays a different, uncontrolled
Crafter world. Measured — three processes, all asking for seed 0:
`_seed = 570254935 / 59658325 / 1185730523`, three different worlds. That would
make "base, seed k" and "PAE, seed k" *different episodes* and void the paired
comparison this experiment is built on.

`CrafterAdapter.reset()` therefore pins it: `crafter` derives the world seed as
`hash((self._seed, self._episode)) % (2**31 - 1)`, and tuples of ints hash
identically in every CPython process (`PYTHONHASHSEED` perturbs only str/bytes),
so setting `_seed = seed` and `_episode = 0` immediately before the reset makes
`--seed k` name one fixed world. Verified cross-process: seed 0 →
`state_digest f9ee440fd993c332` in every process, seed 1 → `c72b651d9e638d50`.

Applied in **every** arm, `--base-only` included, and disclosed in
`summary.json` as `env_patches: ["crafter_balance_chunk_sorted",
"crafter_seed_pinned_to_episode_seed"]`.

## 4. Determinism patch (`crafter_balance_chunk_sorted`)

`crafter.Env.step` calls `_balance_chunk(chunk, objs)` every 10 inner steps,
where `objs` is a Python **set**; `_balance_object` then indexes into
`[obj for obj in objs if isinstance(obj, cls)]` with a random draw. Set order is
`id()`-derived, so a pickle-restored copy orders it differently from the
original and the despawn lands on a different creature — same RNG stream,
different game. `balrog_ckpt/crafter_ckpt.apply_determinism_patch()` sorts
`objs` by `(x, y, class name)` first. It reorders an already-arbitrary choice
and touches neither the RNG stream, nor rewards, nor achievements.

Quantified in `restore_verification.json` (see §6). **Decision (taken): applied
in both arms, disclosed.** `CrafterAdapter.__init__` applies it, and the adapter
is constructed for `--base-only` too, so the base arm gets it as well.

## 5. The PAE checkpoint trigger

`pae.Run.play` (unowned by this task — quoted, not changed):

```python
advanced = prog > last_prog
if advanced or ((step + 1) % self.cfg.checkpoint_every == 0):
    self._save_checkpoint(..., "progression" if advanced else "periodic", ...)
```

plus one checkpoint at episode start (step 0, reason `episode start`). With
`--checkpoint-every 10` (the default) this is exactly:

> **a checkpoint every 10 steps, and a checkpoint on any new achievement.**

For Crafter "progression increased" ⟺ "a new achievement" (§1), so the two
trigger conditions are the intended ones with no adapter-side special-casing.
Notes:

* Checkpoints are written only when `not done`, so the death step itself is
  never a checkpoint; the last resumable point is at most 9 steps before death
  (or the last achievement).
* `resumable()` drops any checkpoint with `step >= step_cap` — a checkpoint at
  the horizon would resume with zero steps left.
* The dense `aux` signal for Crafter is `score_tracker`, i.e. **22 × the
  reported progression**. It drives nothing (not checkpointing, not stopping,
  not the metric); for Crafter it is not even independent of progression, so it
  is only a readability convenience in the orchestrator's ledger (an integer
  count instead of a fraction).
* Restore method: pickle of the inner `crafter.Env.__dict__` minus gym's
  `_np_random`, restored **in place**, plus the wrapper's `score_tracker`,
  `achievements` and `failed_candidates`. Measured restore cost **0.40-0.55 ms**
  — free next to a ~2 s LLM call. No replay is needed, so Crafter has no
  invalid-action-feedback replay hazard (unlike MiniHack).

## 6. Restore verification (artifact: `restore_verification.json`)

Protocol: one base episode on the pinned seed-0 world, 210 scripted actions;
`env_snapshot()` at steps **10, 50, 150**; each snapshot restored and replayed
with the **same 50 actions**; every restored step compared to the original
continuation on observation text, full observation (text + rendered image + raw
array), canonical world state, score, progression, achievements, reward and
done. Both arms start from the same world (`f9ee440fd993c332`), so the only
difference between them is the patch.

| arm | processes | restored steps identical | per-process totals | first divergence offsets |
|---|--:|--:|---|---|
| **patched** | 5 | **750 / 750** (3 ckpts × 50 steps × 5 processes) | 150,150,150,150,150 | none, anywhere |
| unpatched | 5 | 252 / 750 | 16,64,54,64,54 | +8 … +38 steps after restore |

Every restore itself was exact in both arms — `state_digest_after_restore ==
state_digest_at_snapshot` in 30/30 cases. The divergence is in what happens
*next*: unpatched, the restored branch stops matching the original continuation
within 8-38 steps.

The unpatched arm was **never** identical on all three checkpoints in any of 5
processes, and the offset at which it broke moved from process to process (for
the step-10 checkpoint: +8, +8, +8, +28, +38; for step-50: +8, +38, +38, +8, +8;
for step-150: none, +18, +8, +28, +8) — the failure is `id()`-order dependent, so
the *unpatched* behaviour is not itself reproducible, which is the whole
problem. It is also not a hidden bookkeeping difference: at the step-10
checkpoint the divergence reaches the agent's own observation — canonical world
state at +8, rendered image and observation **text** at +24. The patched arm was
identical on every field, every step, every checkpoint, every process.

## 7. Budget risk (open, flagged)

Base Crafter episodes end on **death**, measured above at 56-210 steps under a
random policy and reported in `calibration.json` for the LLM. PAE resumes from a
checkpoint taken *before* death and plays on under the same 2000-step horizon,
so the committed trajectory can only get longer, attempt after attempt. With
N=10 attempts an uncapped PAE run can approach 2000 committed steps where the
base episode cost ~250, i.e. an ~8× cost blow-up per seed.

Mitigation used here and in `launch.sh`: **always pass `--max-steps` explicitly**
to the PAE arm. `launch.sh` defaults the PAE arm to `--max-steps 400` (≈ the
measured base episode length plus headroom) while the base arm runs at BALROG's
own 2000, and it prints the projected cost before it launches anything.
`committed_steps` and `total_env_steps` in `summary.json` are both reported so
the two accountings stay auditable.
