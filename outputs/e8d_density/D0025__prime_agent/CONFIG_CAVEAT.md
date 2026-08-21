# CAVEAT: ran without tune.reveal_map / auto_dismiss=false

This cell's resolved config lacks `tune.reveal_map=1.0` (full vision) and
`auto_dismiss="false"`, unlike the e7 NPCORE_v3 control it was compared to.
Found 2026-08-21 during the E10 fog audit. E10 showed vision ~doubles measured
progress, so cross-cell comparisons against the full-vision control are
confounded. Directional readings within E8 (e.g. mv_fail counts across the
density sweep) may survive; absolute BALROG comparisons do not.
