# The NetHack UI surface: complete inventory of interfaces an agent must answer

An LLM agent never plays NetHack directly — it plays NetHack's *interface*.
Every interface below is a state where the game clock stops and only a specific
key set does anything; every one of them is a place a rollout can wedge. We have
already paid for two of these the hard way (the inventory-menu stall, 7,497
wasted calls across three seeds; the `[- bc or ?*]` wield prompt, 90 wasted calls
in 103 turns — both documented in `prompt/interactive_state.py`). This document
enumerates the rest **before** we pay for them.

Sources: NetHack 3.6.6 as vendored in `third_party/NetHack`
(`include/wintype.h`, `include/winprocs.h`, `src/cmd.c`, `src/botl.c`,
`src/do_name.c`, `src/end.c`, `win/tty/*`, `win/rl/winrl.cc`), plus an empirical
probe of every mode against the live engine (`/root/nld/ui_probe.py`, seed 0,
Valkyrie).

---

## 1. Screen anatomy

NLE renders a fixed 24×80 terminal (`NLE_TERM_LI`/`NLE_TERM_CO`) with four
logical windows (`wintype.h`):

| window | type | where | contents |
|---|---|---|---|
| message | `NHW_MESSAGE` | row 0 | last message(s), all prompts, `--More--` |
| map | `NHW_MAP` | rows 1–21 (`ROWNO`=21) | the dungeon |
| status | `NHW_STATUS` | rows 22–23 | two-line bottom status |
| menu / text | `NHW_MENU`, `NHW_TEXT` | **drawn over the map** | inventory, help, disclosure, … |

The critical structural fact: **menus and text windows overlay the map region of
`tty_chars` but leave `chars`/`glyphs` untouched.** Any encoder that renders the
map from the glyph plane (ours does, deliberately — it kills menu bleed) is
blind to overlays by construction, which is why `detect_blocking_ui` exists.

The status line's level field is *not* always `Dlvl:N` (`botl.c
describe_level`): the Quest shows `Home N`, Ft. Ludios its dungeon name, the
endgame the plane name. Under polymorph, `Xp:N/n` becomes `HD:n`. Both matter to
anything parsing depth/XP out of tty text.

---

## 2. Blocking interfaces — measured

`misc` in the NLE observation is exactly `[in_yn_function, in_getlin,
xwaitingforspace]` (`win/rl/winrl.cc:638`). Measured behaviour, one fresh game
per row; "clock" is whether a `s` (search) keypress advances `blstats[20]`:

| # | interface | opened by | `misc` | clock | row 0 looks like |
|---|---|---|---|---|---|
| 1 | `--More--` pagination | any long/multi message | `[0,0,1]` | FROZEN | `…--More--` |
| 2 | yn confirm | `#pray`, `#quit`, `Really attack?` | `[1,0,0]` | FROZEN | `Really quit? [yn] (n)` |
| 3 | getobj item prompt | `w` `e` `r` `q` `P` `T` `a` `t` `z` … | `[1,0,0]` | FROZEN | `What do you want to wield? [- bc or ?*]` |
| 4 | getobj → menu (`?`/`*`) | `?` at a getobj prompt | `[0,0,1]` | FROZEN | `Select one item:` |
| 5 | menu, PICK_NONE | `i` (inventory) | `[0,0,1]` | FROZEN | overlay: `Weapons` … |
| 6 | menu, PICK_ANY | `D` (droptype), `m,` (pickup types) | `[0,0,1]` | FROZEN | `Drop what type of items?` |
| 7 | menu, PICK_ONE | `?` (help), `#terrain`, `C` (name what?) | `[0,0,1]` | FROZEN | `Select one item:` / `View which?` |
| 8 | text window | `^X` attrs, `\` discoveries, `^O` overview, `#enhance` | `[0,0,1]` | FROZEN | `Agent the Valkyrie's attributes:` |
| 9 | getlin free text | `E` engrave, `#annotate`, `C` call, wish | `[0,1,*]` | FROZEN | `What do you want to call this dungeon level?` |
| 10 | getdir | kick `^D`, `z` zap, `#force`, apply | `[1,0,0]` | FROZEN | `In what direction?` |
| 11 | **getpos** (travel) | `_` | `[0,0,0]` | FROZEN | `Where do you want to travel to?  (For instructions type a ?)` |
| 12 | **getpos** (farlook) | `;` `/` teleport, spell target | `[0,0,0]` | FROZEN | `Pick an object.` |
| 13 | **ext-cmd entry** | `#` | `[0,0,0]` | FROZEN | `#` (then the typed prefix) |
| 14 | **ext-cmd list** | `#?` | `[0,0,0]` | FROZEN | `# ?` |

**19 of 27 probed interfaces block the clock, and `misc` flags only 14 of
them.** The four invisible-in-`misc` modes (bold above) still announce
themselves on tty row 0 — but `getpos: farlook` says `Pick an object.` /
`Pick a monster.`, which matches none of our current markers.

There is a worse class still: **command prefixes freeze the clock while
signalling nothing in either channel.**

| probe | `misc` | row 0 | `s` advances clock? |
|---|---|---|---|
| getdir via kick `^D` | `[1,0,0]` | `In what direction?` | no (visible) |
| prefix `F` (fight) | `[0,0,0]` | *stale previous message* | **no** |
| prefix `g` (rush) | `[0,0,0]` | *stale previous message* | **no** |
| prefix `m` (no-pickup) | `[0,0,0]` | *stale previous message* | **no** |

A stray `F`/`g`/`G`/`M`/`m` leaves the game waiting for a direction with the
observation byte-identical to the previous turn — the precise input condition of
the deterministic-loop failure. It self-clears after one key, so the cost is
bounded at one call *if* the next action happens to be a direction key.

Non-blocking but stateful:

| interface | opened by | note |
|---|---|---|
| count prefix | `n`, or digits with `number_pad` | `Count: 10` on row 0; next command repeats N× |
| `^P` message recall | — | reprints history; harmless |
| `O` options | — | NLE stubs it: "The options are already set perfectly for you!" |

---

## 3. Exit semantics

- `ESC` cancels 2–9 cleanly (one press).
- `--More--` wants space/return; `ESC` also dismisses and *suppresses* the rest
  of the queued messages — cheap, but it discards information.
- Menus additionally honour their own control keys (`wintype.h`):
  `>` `<` `^` `|` page nav, `.` select-all, `-` unselect-all, `@` invert,
  `,` `\` `~` page-scoped variants, `:` search. Pagination shows `(N of M)`;
  the last page of a text window shows `(end)`.
- getpos has its own 20-key language (`cmd.c spkeys_binds`): `.`/`,`/`;`/`:`
  pick, `@` self, `$` show valid, `#` autodescribe, `m`/`M` cycle monsters,
  `o`/`O` objects, `d`/`D` doors, `x`/`X` unexplored, `z`/`Z` valid targets,
  `<`/`>` stairs, `?` help, plus `hjkl`/`HJKL` cursor movement. An agent that
  only knows "move cursor then `.`" is leaving the cheap targeting primitives
  (`>` = "put the cursor on the downstairs") unused.
- getdir accepts the 8 direction keys plus `.`/`s` (self) and `?` (help).

---

## 4. Command surface

- **Extended commands** (`cmd.c extcmdlist`, 3.6.6): 80 entries, `#adjust`
  `#annotate` `#apply` `#attributes` `#autopickup` `#call` `#cast` `#chat`
  `#close` `#conduct` `#dip` `#down` `#drop` `#droptype` `#eat` `#engrave`
  `#enhance` `#fire` `#force` `#glance` `#help` `#herecmdmenu` `#history`
  `#inventory` `#inventtype` `#invoke` `#jump` `#kick` `#known` `#knownclass`
  `#look` `#loot` `#monster` `#name` `#offer` `#open` `#options` `#overview`
  `#pay` `#pickup` `#pray` `#quaff` `#quiver` `#read` `#redraw` `#remove`
  `#ride` `#rub` `#save` `#search` `#seeall` `#see*` (7 variants) `#sit`
  `#swap` `#takeoff` `#takeoffall` `#teleport` `#terrain` `#travel` `#turn`
  `#twoweapon` `#untrap` `#up` `#version` `#versionshort` `#vanquished` `#wear`
  `#wield` `#wipe` `#zap` (+ wizard-mode/debug entries).
- Entering one is a two-stage blocking interface: `#` opens ext-cmd entry
  (invisible in `misc`), the name autocompletes, `\r` submits — and the command
  itself then usually opens *another* interface (`#pray` → yn, `#enhance` →
  menu, `#annotate` → getlin).

---

## 5. Game-lifecycle screens

- **Character creation** — `win_player_selection` / `win_askname`, plus the
  `Shall I pick a character's race, role, gender and alignment for you? [ynaq]`
  prompt visible in raw NAO ttyrecs. NLE bypasses all of it via the `character=`
  string, so our agents never see it.
- **End of game** — measured, not inferred. After `#quit` → `y` (or death) the
  engine walks a **five-screen disclosure chain**, each needing its own
  dismissal before `done` flips:
  1. inventory disclosure (`Weapons` … overlay), 2. `…'s attributes:`,
  3. `Voluntary challenges:`, 4. `The Dungeons of Doom:` overview,
  5. `Farvel Agent the Valkyrie…` tombstone → then the top-ten list and
  `done=True, how_done=13`.
  `y` does **not** advance this chain (the menus want space/ESC) — an agent that
  answers `y` to everything after death spins until the call budget ends. This
  is worth an explicit auto-dismiss in the harness: it is pure overhead on every
  single rollout that dies, which is nearly all of them.

---

## 6. Coverage gaps in our harness

`detect_blocking_ui` currently fires on `misc.any()` plus five row-0 markers
(`--more--`, `where do you want to travel to?`, `(for instructions type a`,
`(end)`, `[yn`). Against the table above:

| gap | consequence | fix |
|---|---|---|
| `getpos` farlook/`;`/`/` (`Pick an object.`, `Pick a monster.`) | silent freeze, identical obs → deterministic loop, the exact 7,497-call failure shape | add the `gloc_descr` prompt strings as markers |
| command prefixes `F`/`g`/`G`/`M`/`m` | clock frozen with **no signal at all** — not in `misc`, not on row 0 | detect via "clock did not advance and observation unchanged"; a stale-clock counter is the only channel that catches this |
| `getdir` (`In what direction?`) | visible in `misc`, but the legal answers are never named | publish the 8 direction keys as the offered answers, as `_offered_answers` does for `[...]` |
| ext-cmd entry (`#`) | same; likely under-reported because `#` is rarely reachable in our skill sets | add marker |
| menu control keys never surfaced | agent can only page with luck; multi-select drops/pickups unavailable | name `>`/`.`/`,`/`:` in the blocking-UI block, as `_offered_answers` already does for `[...]` groups |
| post-death chain | wasted tail-of-rollout calls, every dying run | auto-dismiss on `done`-pending |
| `Home N` / plane / `HD:` status variants | depth/XP parse silently wrong on Quest, endgame, polymorph | already noted in `NLD_HUMAN_DATA.md`; same fix applies to live parsing |

The general principle the measurements support: **`misc` is necessary but not
sufficient** — it covers 14 of 19 blocking modes. Anything relying on it alone
inherits five silent-freeze classes.
