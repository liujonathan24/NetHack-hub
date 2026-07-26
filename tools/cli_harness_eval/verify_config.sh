#!/usr/bin/env bash
# Print a fingerprint of the experiment-critical configuration, so a second
# session (or a future one) can confirm it is working from the same tree.
#
#   tools/cli_harness_eval/verify_config.sh
#
# Every line is something that has silently drifted at least once in this
# project. Compare the output against the EXPECTED block at the bottom.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ENG="${ENG:-/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness}"
PY=.venv-cli-eval/bin/python

hr() { printf '%s\n' "-----------------------------------------------------------"; }

hr; echo "GIT"
echo "  HEAD        $(git rev-parse --short HEAD)  $(git log -1 --format=%s | cut -c1-58)"
echo "  dirty       $(test -n "$(git status --porcelain)" && echo YES || echo no)"
for c in 20815be d3d2442 f423a4a 1c85160; do
  printf "  has %-9s %s\n" "$c" "$(git merge-base --is-ancestor $c HEAD 2>/dev/null && echo yes || echo NO)"
done

hr; echo "OBSERVATION DEFAULTS  (exp1 ran B0 uncompacted; both defaulted wrong once)"
grep -hn 'variant: str' environments/nethack/nethack.py environments/nethack/nethack_v1.py | sed 's/^/  /'
grep -hn 'compact_obs: bool' environments/nethack/nethack.py | sed 's/^/  /'

hr; echo "ARM CONFIGS"
for f in tools/cli_harness_eval/configs/*.toml; do
  printf "  %-16s variant=%-4s skill_set=%-12s trace_dir=%s\n" "$(basename "$f")" \
    "$(grep -oP '(?<=^variant = ")[^"]+' "$f" | head -1)" \
    "$(grep -oP '(?<=^skill_set = ")[^"]+' "$f" | head -1)" \
    "$(grep -q '^trace_dir = "/' "$f" && echo absolute || echo RELATIVE)"
done

hr; echo "SKILL SETS  (netplay must withhold raw move; netplay_true deliberately does not)"
PYTHONPATH="$ENG:.:environments/nethack" $PY - <<'PY' 2>/dev/null
from nethack_harness.helpers import _build_skill_adapter_callables as b
for s in ("netplay", "netplay_true"):
    n = sorted(t.__name__ for t in b(skill_set=s))
    raw = "move" in n or any(x in n for x in ("np_press_key", "np_type_text"))
    print(f"  {s:14s} {len(n):3d} tools   raw-keystroke surface: {raw}")
PY

hr; echo "OBSERVATION-LAYER REPAIR (Task 17)"
printf "  features.py present   %s\n" "$(test -f environments/nethack/nethack_harness/prompt/features.py && echo yes || echo NO)"
printf "  open-door path fix    %s\n" "$(grep -q 'open_door_tiles' environments/nethack/nethack_harness/tools/skills.py && echo yes || echo NO)"
printf "  prompt advertises move %s\n" "$(grep -qE 'move\(direction' environments/nethack/nethack_harness/prompt/rendering.py && echo YES-BAD || echo no)"

hr; echo "VENV"
printf "  verifiers   %s\n" "$(PYTHONPATH="$ENG:.:environments/nethack" $PY -c 'import verifiers;print(verifiers.__version__)' 2>/dev/null)"
printf "  stock       %s\n" "$($PY - <<'PY' 2>/dev/null
import hashlib,base64,csv,pathlib
sp=pathlib.Path(".venv-cli-eval/lib/python3.12/site-packages")
bad=0
for rec in sp.glob("verifiers-*.dist-info/RECORD"):
    for row in csv.reader(rec.open()):
        if len(row)<2 or not row[1].startswith("sha256="): continue
        f=sp/row[0]
        if not f.is_file(): continue
        d=base64.urlsafe_b64encode(hashlib.sha256(f.read_bytes()).digest()).rstrip(b"=").decode()
        if d!=row[1][7:]: bad+=1
print("yes" if bad==0 else f"NO ({bad} modified)")
PY
)"

hr; echo "TESTS"
PYTHONPATH="$ENG:.:environments/nethack" $PY -m pytest environments/nethack/tests/ tests/ -q 2>/dev/null | tail -1 | sed 's/^/  /'

hr; cat <<'EXPECTED'
EXPECTED as of run1 (2026-07-26):
  HEAD                 1c85160 or later, all four ancestor checks yes
  variant defaults     "B0" in both nethack.py and nethack_v1.py
  compact_obs default  False
  arm configs          variant=B0, skill_set=netplay, trace_dir=absolute  (all three)
  netplay               18 tools, raw-keystroke surface: False
  netplay_true          31 tools, raw-keystroke surface: True   (deliberate, faithful to upstream)
  features.py           yes      open-door fix: yes      prompt advertises move: no
  verifiers             0.2.1, stock
  tests                 292 passed
EXPECTED
