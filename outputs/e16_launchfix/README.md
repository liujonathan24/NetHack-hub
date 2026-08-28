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
