# Vendored NetPlay skill layer — provenance

## Upstream

| | |
|---|---|
| Repository | <https://github.com/CommanderCero/NetPlay> |
| Commit | `6acb90d865411f28d440e042372d9034f971e54a` (default branch `main`, 2024-11-04) |
| License | MIT — preserved verbatim at `netplay/LICENSE` |
| Paper | Jeurissen, Perez-Liebana, Gow, Cakmak, Kwan, "Playing NetHack with LLMs: Potential & Limitations as Zero-Shot Agents", CoG 2024, arXiv:2403.00690 |

Fetched with `curl https://raw.githubusercontent.com/CommanderCero/NetPlay/<SHA>/<path>`.

**Why vendored rather than reimplemented.** Our own `netplay` skill set is an
*approximation* of NetPlay's action surface, and the differences are behavioural,
not cosmetic: our `attack(direction)` is a single directional bump where
NetPlay's `melee_attack(x, y)` pursues a target until it dies, and our
`explore_and_descend` caps its search where NetPlay's `explore_level` runs until
exploration is provably exhausted. To compare against NetPlay's published
numbers we need NetPlay's actual code, so this is a port, not a rewrite.

**Fidelity rule applied throughout:** adapt imports and the engine seam; change
no behavioural line. Where upstream looks buggy, it is left as-is and noted
under "Upstream quirks" below — a silent improvement would make our numbers
incomparable to theirs.

## Fidelity summary

Eleven of the twelve vendored behavioural files are **byte-identical** to
upstream. Verify at any time with:

```bash
diff vendor/netplay/nethack_agent/skills.py <(curl -sSL \
  https://raw.githubusercontent.com/CommanderCero/NetPlay/6acb90d865411f28d440e042372d9034f971e54a/netplay/nethack_agent/skills.py)
```

| Vendored file | Upstream path | Diff |
|---|---|---|
| `netplay/nethack_agent/skills.py` | same | **identical** |
| `netplay/nethack_agent/tracking.py` | same | **identical** |
| `netplay/nethack_agent/describe.py` | same | **identical** |
| `netplay/nethack_utils/glyphs.py` | same | **identical** |
| `netplay/nethack_utils/monster.py` | same | **identical** |
| `netplay/nethack_utils/monflag.py` | same | **identical** |
| `netplay/nethack_utils/screen_symbols.py` | same | **identical** |
| `netplay/core/skill.py` | same | **identical** |
| `netplay/core/skill_repository.py` | same | **identical** |
| `netplay/core/descriptor.py` | same | **identical** |
| `netplay/core/agent_base.py` | same | **identical** |
| `netplay/nethack_agent/pathfinding.py` | same | 1 line — see below |

This was achieved by putting `vendor/` on `sys.path` as a **package root**, so
upstream's own absolute imports (`netplay.*`, `nle.*`, `nle_language_wrapper`,
`gradio`) resolve without editing a single import line. The adaptation lives in
shims *around* the vendored code rather than *inside* it.

## The one changed behavioural line

`netplay/nethack_agent/pathfinding.py`, in `nethack_bfs`:

```python
-        y, x = buf[index]
+        y, x = int(buf[index][0]), int(buf[index][1])
```

`buf` is `np.uint32`, so upstream unpacks numpy uint32 scalars and then computes
`y + dy` with `dy == -1`. Under numpy 1.x that promoted to a signed integer and
the following `if py < 0` bounds check worked. NumPy 2's NEP 50 promotion rules
instead raise `OverflowError: Python integer -1 out of bounds for uint32`, so
upstream's BFS cannot run at all on numpy 2 (our venv has 2.5.1). Casting to
Python `int` restores exactly the numpy 1.x arithmetic the function was written
against; traversal order, distances and paths are unchanged.

This is an environment-compatibility fix, not a behavioural improvement.

## Files replaced or reduced (and why)

### `netplay/nethack_agent/agent.py` — ADAPTED

Upstream this is the **LLM agent**: an `AgentMemory` rolling chat buffer, a
`_solve_task` loop calling an LLM skill-selector, gradio/minihack renderers.
It imports langchain, openai, nle_language_wrapper, gradio and minihack.

We reuse NetPlay's *skill layer*, not its agent loop — our harness supplies the
policy (the LLM scaffold under evaluation) and calls skills one at a time. Kept:

- `finish_task_skill` — byte-for-byte (upstream 38–44)
- `_execute_skill` — byte-for-byte (upstream 215–238)
- `_skip_more_messages` — byte-for-byte (upstream 189–196)
- `_update_objects` — byte-for-byte (upstream 198–213)

The three `_`-prefixed methods are **not** LLM plumbing — they are the skill
*execution contract*, and they materially bound behaviour: a skill is stopped
after `max_skill_gamesteps` (100) in-game turns and interrupted on a dungeon
level change, teleport, newly-seen glyph or low health. `explore_level` runs
until exploration is exhausted, so without this cap it would not be the action
NetPlay's agent actually took.

Replaced: `NetHackAgent.__init__` only. Every attribute `skills.py` touches on
the agent — `blstats`, `current_level`, `step`, `get_path_to`, `distance_to`,
`get_distance_map`, `get_walkable_mask`, `waiting_for_popup`,
`current_game_message`, `set_current_room`, `avoid_monsters` — is defined by the
byte-identical `netplay/core/agent_base.py`, so dropping the LLM plumbing
changes no skill semantics.

### `netplay/nethack_utils/nle_wrapper.py` — REDUCED

`RawKeyPress` (upstream 12–134) is reproduced **byte-for-byte**;
`render_ascii_map` and `NethackGymnasiumWrapper` (upstream 136–273) are removed.
That class *is* the NLE/MiniHack engine seam this port replaces — it constructs
`nle.env.base.NLE` / `minihack.MiniHack` and adapts NLE's 4-tuple step. Keeping
it would add unusable code plus hard dependencies on `nle`, `minihack` and PIL.
`skills.py` imports only `RawKeyPress` from this module.

### `netplay/__init__.py` — REDUCED

Upstream is a single `create_llm_agent()` factory wiring the skill repository to
langchain/openai, the gradio renderer and the NLE env. Not reproduced. The part
that defines the **action surface** — upstream lines 9–18 — is preserved
verbatim as `NETPLAY_SKILL_REPOSITORY` in
`nethack_harness/tools/netplay_true.py`: same eight entries, same order.

### Not ported

`descriptors.py`, `skill_selection.py`, `agent_base.py`'s renderer subclasses,
and the whole `autoascend/` tree. These are observation-rendering and
LLM-prompting concerns — our harness owns those, and this task is about the
*action surface*. Note `skills.py` has **zero** `autoascend` references; that
package is vendored in upstream's repo but the LLM-facing skill layer never
calls into it.

## How the `nle` dependency was resolved

`nle` is **not installed**, and was **not installed** — `.venv-cli-eval` stays
stock (the `test_site_packages_loader_is_stock` assertion is unaffected).

Installing it was rejected for two reasons. First, `nle` ships its *own* NetHack
engine as a compiled extension; adding it puts a second, differently-built
NetHack in the process next to the fork we actually benchmark, which invites
exactly the kind of silent divergence this repo has been bitten by before.
Second, it is unnecessary: the vendored skill layer uses `nle.nethack` only for
**constants and pure glyph arithmetic**, never to drive an engine.

So `vendor/nle/` is a shim re-exporting our engine's equivalents:

| Symbol group | Source |
|---|---|
| `actions.*` (`Command`, `CompassDirection`, …) | `nethack_core.actions`, which mirrors NLE's enums verbatim (same names, same keystroke ints) |
| glyph offsets, `glyph_is_*`, `glyph_to_mon` | `nethack_core.glyphs` |
| `GLYPH_{EXPLODE,ZAP,SWALLOW,WARNING}_OFF`, `glyph_is/to_{swallow,warning}`, `glyph_to_{obj,cmap,pet}` | derived with the exact arithmetic from the fork's `include/display.h`; asserted at import against `nethack_core`'s independently-derived `GLYPH_STATUE_OFF` / `MAX_GLYPH` |
| `objclass()`, `objdescr.from_idx()`, `*_CLASS` | `nle_shim_data/objects_table.py`, generated from the fork's `src/objects.c` |
| `permonst().mname` | `nethack_core.glyphs.MONSTER_NAMES` |

`vendor/nle/` is **appended** to `sys.path`, never prepended, so a real `nle`
install would take precedence if one were ever added.

### The generated object table

`nethack_core` exposes no object-class table, but `glyphs.py` needs one:
`G.BOULDERS` feeds `Level.get_walkable_mask(treat_boulder_unwalkable=True)`, so
getting it wrong would silently corrupt every path the agent plans.

`tools/gen_object_table.py` parses the fork's `src/objects.c` — the same source
NetHack's own `makedefs` reads — mapping each class-implying macro to its
`oc_class`. Two subtleties are handled explicitly:

- The `ROCK()` macro expands to `GEM_CLASS` (rocks and flint are gems). Genuine
  `ROCK_CLASS` objects (boulder, statue) are declared with raw `OBJECT(...)`.
- Five entries are guarded by `#if 0 /* DEFERRED */` and `#ifdef MAIL`, neither
  compiled into our engine. Counting them would shift every later `otyp`.

The generator refuses to emit a table unless all three checks pass: exactly 453
objects (`== nethack_core.glyphs.NUM_OBJECTS`), `BOULDER` at `otyp` 447 with
`ROCK_CLASS`, and classes monotonic in `otyp` order. Regenerate with:

```bash
python vendor/tools/gen_object_table.py \
  third_party/NetHack/src/src/objects.c \
  third_party/NetHack/src/build/include/onames.h \
  vendor/nle_shim_data/objects_table.py
```

## Other shims

| Shim | Stands in for | Used by |
|---|---|---|
| `nle/env.py` | `nle.env.NLE` | `skills.py` imports it but never references it — a dead upstream import |
| `nle_language_wrapper/` | `NLELanguageObsv.text_message` | `tracking.py`, to turn the tty top line into the game message. Upstream's is a C extension; ours reproduces that one function |
| `gradio/` | `gradio.components.Component` | `core/agent_base.py`, in an annotation evaluated at class-definition time |
| `netplay/logging/` | `StepLogger`, `AgentVideoRenderer` | `core/agent_base.py` calls these every step; our harness does its own tracing, so they are inert |

## Upstream quirks left deliberately unfixed

Left as-is for fidelity; none affect the skills we exercise.

1. `tracking.py:526` — `re.findall("^(a|an|the|\d+)...")` uses an unescaped
   `\d` in a non-raw string, so Python 3.12 emits a `SyntaxWarning`.
2. `monster.py:31` — `find()` references `MON.ALL_PETS` without importing `MON`;
   calling it would raise `NameError`. Nothing calls it.
3. `skills.py` — several failure branches `yield Step.failed(...)` without a
   following `return` (e.g. `melee_attack` when the target is unreachable,
   `explore`/`go_to`/`search_room` completion branches), so the generator keeps
   running past what reads as a terminal state.
4. `core/agent_base.py:230,236` — `get_path_to`/`distance_to` accept an
   `avoid_monsters` argument, normalise it into a local, then pass
   `self.avoid_monsters` instead of the local. The parameter is effectively
   ignored; only the agent-level flag has effect. This one is load-bearing —
   `melee_attack` passes `avoid_monsters=True` expecting monster-avoiding paths
   and does not get them.
5. `skills.py:604` — bare `except:` in `zap`.
