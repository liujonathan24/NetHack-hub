# E16 launch blockers — what was fixed, and the evidence

Everything here is reproducible and, apart from one $0.24 rollout, free.

## The three silent hangs (Task 1)

`tools/cli_harness_eval/e16_preflight.py` exercises all three against a stub
`prime-agent`, with no inference:

    python tools/cli_harness_eval/e16_preflight.py
    # or:  tools/cli_harness_eval/run_e16.sh <RUN_DIR> preflight

`run_e16.sh <RUN_DIR> run` refuses to launch unless it passes
(`E16_SKIP_PREFLIGHT=1` overrides, knowingly).

## The end-to-end dry run (no inference)

    python outputs/e16_launchfix/dry_run.py <OUT>

Three arms, four attempts each, real engine and real archive with a stub
player: `go_explore`, `go_explore --paired-control`, and `matched_restart`
(the null).

## Prefix replay — does it reach the model? (Task 4)

The question the builder's "text replay only" note left open. Three steps:

    python outputs/e16_launchfix/prefix_make_checkpoint.py <WORK>/archive
    WORK=<WORK> outputs/e16_launchfix/prefix_run_rollout.sh          # ~$0.24
    python outputs/e16_launchfix/prefix_served_bytes.py \
        <WORK>/rollout <WORK>/ledger_text.txt QUILLFEATHER-7731

`prefix_served_bytes.result.json` is the answer from the run that was actually
made: **the replayed prefix reaches the model.** The block appears
byte-identically at offset 670 of served node 7 (the first observation), in
exactly one served node, carrying a canary string that exists nowhere else in
the harness, the wiki, the ledger table or the directive.

So H4 is testable as "lessons + prefix CONTENT transfer". What remains out of
reach is prefix continuity of SESSION — an inherited conversation and its
prompt cache — because both player scaffolds launch with sessions off. Those
are different claims and only the second is blocked.

## The two launch blockers this run found in `launch_cell.sh`

Neither was reachable from the no-inference dry run (a stub player never goes
through that script) or from the earlier model-in-the-loop sims (they passed
only a single-line `directive`). The first real resumed rollout hit both.

1. **`mapfile -t` split every multi-line `ledger_text` into one argv item per
   LINE.** Every line after the first reached the eval CLI as a bare positional
   and the run died at boot with `Unrecognized arguments: id Dlvl XL HP turn
   score ...`. The ledger is multi-line by construction, so this was not an
   edge case — it was every E16 attempt. Records are now NUL-separated.
2. **The `E16_ARGS` whitelist rejection was not fatal.** `mapfile ... < <(cmd)
   || exit 2` binds the `||` to `mapfile`, whose status is unrelated to
   `cmd`'s, so a rejected key printed its error and the launch CONTINUED with
   an empty flag array — a cell that silently lost its `resume_checkpoint`, its
   ledger and its directive, exiting 0. That is the silent-substitution class
   that invalidated E15.

Regression tests for both: `tests/test_launch_cell.py::test_e16_args_*`.

---

# What the GE-wiki pilot found (one attempt, $13.48)

The first real run of the fixed launcher completed one attempt and exposed
three more defects. Two of them were the SAME defect in
`e16_session.parse_json_mode_stdout`, and neither was reachable from anything
above because every stub in the tree — the dry run's and the unit tests' —
answered `--mode json` records on rounds the real session runs as plain text.
The preflight's stub was the one honest one, which is why the new gate lives
there.

Evidence: `/root/nld/e16_runs/gewiki_pilot/`.

## 1. The directive never reached the attempt

`orchestrator_rounds.jsonl` round 2 (`kind=round1`) recorded a 678-char reply
ending mid-rationale. The model's actual reply, in its own session file
(`orchestrator/agent/sessions/01a04847-…jsonl`, record 49), is 1,104 chars and
ends in the decision object the prompt demanded:

    {"checkpoint": "1", "directive": "Take the down-stairs on each level as
     soon as you find them and save a checkpoint at every new depth you reach.
     Do not fight any monster while your HP is below half of its maximum.", …}

Plain `--print` writes assistant text and nothing else, but the driver handed
that text to the **json record parser**, which walks stdout line by line and
drops any JSON object it does not recognise as a protocol record. The decision
line was on its own line — as instructed — so it was deleted. `parse_decision`
then found nothing, `decide()` fell back to the scripted selector, and
attempt 1 was launched with "(orchestrator produced no directive for this
attempt; play as you judge best)". $13.36 measuring the control arm under a
treatment label.

Fixed by `parse_round_stdout(stdout, json_mode)`: a text round is parsed as
text. The silent fallback is gone — `decide()` retries `directive_retries`
times and then raises `DirectiveExtractionFailed`, and `_launch_one` refuses an
empty directive on a treatment attempt as defence in depth.

## 2. Zero checkpoints despite reaching Dlvl 3

`max_dlvl: 3`, `new_checkpoints: []`, archive still one row. Diagnosis: the
design's automatic checkpoints — level entry, level-up, every 150 game turns —
**did not exist in code**. The only writer was the model's own `save` skill,
and across 187 calls the model never called it (`np_move_to` 53,
`np_press_key` 46, `request_map` 30, `np_explore_level` 25, `np_melee_attack`
16, `search` 12, `np_kick` 5, `save` 0). The `checkpoint_save` prompt-pending
guard was NOT the cause: it never ran, because nothing ever called it.

Fixed in `nethack.py` (`_maybe_auto_checkpoint`, run after every tool call). A
trigger the savepoint guard refuses — which is the common case at level entry,
since descending prints a `--More--` — is DEFERRED to the next quiescent call
rather than dropped, and the deferral is counted in the turn trace.

## 3. The opening round "degenerated" — it did not

`opening_plan.txt` is 2,656,190 chars: 18,088 non-blank lines, 589 unique, the
line "Now I have a complete picture. Let me synthesize everything." nine times
and "Here is my Round 1 discussion." 533 times. The model did none of that. Its
session file holds ONE 11,478-char plan (record 44) and that synthesis line
exactly once (record 42). `--mode json` re-emits the whole message on every
text delta, and the parser concatenated all 533 cumulative prefixes.

So there was no repetition loop — but there was also no way for the run to tell.
Degeneration detection (`detect_degeneration`) is now run on every orchestrator
reply, with a bounded retry and a hard `OpeningPlanUnusable` if the plan is
still unusable; the raw stdout of every attempt is written to
`orchestrator/raw/` either way. On the pilot's two artifacts it scores 0.033
unique-line ratio (the recorded plan) against 0.959 (the real one).

## Regression coverage

* `tests/test_e16_orchestrator.py::test_THE_PILOT_DIRECTIVE_survives_a_text_mode_round`
  — the pilot's real reply bytes, vendored as a fixture, through a real session.
* `…::test_json_mode_stream_is_not_multiplied_by_its_own_deltas`
* `…::test_degeneration_detector_flags_the_pilots_recorded_opening_plan`
* `…::test_an_invented_checkpoint_id_is_retried_and_then_RAISES`
* `environments/nethack/tests/test_e16_wiki_and_directive.py` —
  `test_auto_checkpoints_grow_the_archive_WHEN_THE_ROLLOUT_DESCENDS` drives a
  real engine down a staircase and asserts the archive grew with a restorable
  checkpoint the model never asked for.
* `e16_preflight.py::directive_extraction` — the launch gate.

## Cost: uncapped players are the expensive choice

Measured from the pilot's own trace with the repo's price table:
**97.8% of the $13.36 is re-reading accumulated context.** Input grows linearly
at ~396 tokens/call (R²=0.995, no compaction, no plateau), so cumulative cost
is quadratic in attempt length: `cost(L) = 3.36e-4·L² + 5.85e-3·L`
(R²=0.99995). $/call is therefore a function of length, not of the arm —
$0.0123/call at 10 calls, $0.0229 at 50, $0.0392 at 100, $0.0699 at 191. The
short sims' "$0.017–$0.027/call" is the same curve's first 25 calls.

At $385: **30 attempts at 187 calls, or 340 at 50, or 1,080 at 25.** Prompt
caching is already at a 99.2% hit rate and saves nothing, because
prime-inference prices cache reads at the full input rate.
