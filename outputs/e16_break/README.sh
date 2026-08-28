#!/bin/sh
# E16 adversarial verification probes (checkpoint/integrity layer).
# Run each with the launch PYTHONPATH:
#
#   export PYTHONPATH="/root/nld/zombie-fix/tools/pycompat:/root/NetHack-engine:\
# /root/nld/zombie-fix:/root/nld/zombie-fix/environments/nethack"
#   cd /root/nld/zombie-fix
#   .venv-cli-eval/bin/python outputs/e16_break/<probe>.py [case]
#
# Probes are read-only with respect to the repo; they work under /tmp.
#
#   p04  level-file BODY corruption sweep      (mode arg, e.g. zero_body)
#   p06  malformed checkpoint-directory matrix (mode arg)
#   p11  natural-play cross-process fidelity sweep (no args)
#   p12  SIGKILL a save at 4 points in the write sequence (no args)
#   p13  repeated / cross-process restores, drift, leaks (no args)
#   p14  the three flagged known risks         (case arg)
#   p15  real-death-then-restore attempt       (no args)
#   p16  #quit game-over, fully drained, then restore into that env (no args)
#   p17  does the proactive guard engage on the production env (no args)
#   p18  archive hazards: list DoS, concurrent save, path traversal (case arg)
#   p19  the two headline attacks, version-tolerant (case arg)
