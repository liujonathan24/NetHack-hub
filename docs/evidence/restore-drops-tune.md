# A checkpoint restore silently resets every tune knob to vanilla

## Symptom

In the E16 Go-Explore runs, 18-28% of attempts in `reveal_map = 1.0` arms began
with an effectively fogged first observation -- under 50 map cells where the
tier promises a fully revealed floor:

| run            | reveal_map | attempts | start <50 cells | median start cells |
|----------------|-----------:|---------:|----------------:|-------------------:|
| treesmoke11    |        1.0 |       68 |        12 (18%) |                194 |
| treesmoke11_r2 |        1.0 |       52 |        10 (19%) |                228 |
| treesmoke11_r3 |        1.0 |       92 |        26 (28%) |                188 |
| e16_s2_r1      |        1.0 |       30 |          1 (3%) |                349 |
| e16_fog_s1_r1  |        0.0 |       26 |        10 (38%) |                 73 |

The sharpest single example: `treesmoke11_r2/a036` resumes at Dlvl 19 with
8 map cells; `a002` of the same run starts with 451.

`e16_trace_viewer.html` in this directory is a browsable reader over the full
orchestrator and player transcripts of `e16_fog_s1_r1` and `treesmoke11_r2`;
every observation block carries its own revealed-cell count, so the symptom can
be inspected in the served bytes rather than taken from this table.

## Root cause

`checkpoint_restore` (environments/nethack/nethack_harness/checkpoints.py)
rebuilds the engine with `engine_env.reset(seeds=...)` and no `tune`.
`RawEngine.start` with `tune=None` reconstructs the C context with
`tune_n == 0` -- every knob back to vanilla -- and nothing downstream of a
restore re-applies the arm's tune.

Two things made it invisible:

1. The hero's REMEMBERED map lives in the player blob and survives the
   restore. A resume onto an explored floor renders from memory and looks
   normal. Only a resume onto a fresh floor -- level-entry checkpoints, 12% of
   the archive but a frequent orchestrator choice -- came up dark.
2. `preflight_cell.py` asserts >200 revealed glyphs on the FIRST observation of
   a cell, which a fresh reset passes; no per-restore check existed.

`reveal_map` is the knob that made this visible, but every tune knob is lost
the same way (`room_density`, damage scales, ...). Generation-time knobs
happen to be inert on restore -- the checkpoint's floors are already built and
their level blobs carry the geometry -- which narrows the practical blast
radius to render/live knobs.

## Direction of the bias

The bug degrades the CONTROL arm of the fog comparison toward the treatment:
full-vision attempts intermittently played fogged, while fog arms (reveal 0.0)
have no overlay to lose and are unaffected. The measured fog penalty
(BALROG-min 7.45 vs control mean 12.35; frontier advance 31% vs 65-73%) is
therefore a lower bound; a clean control widens it.

## Fix

Capture `engine_env.get_tune()` before the reset inside `checkpoint_restore`
and re-apply it after -- from the live engine rather than a parameter, so every
caller is fixed at once and cannot pass a tune that contradicts the env it
hands in. Regression test:
`test_persistent_checkpoint.py::test_restore_preserves_the_tune`, verified to
fail against the unfixed module and pass with the fix.
