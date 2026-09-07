#!/usr/bin/env bash
# Bring a fresh box to the point where an experiment can launch.
#
#   bootstrap/bootstrap_new_box.sh
#
# This box is disposable and has no persistent volume, so bring-up is a script
# rather than a remembered sequence. Almost everything rebuilds; exactly two
# things must be carried over by hand, and both are named at the end.
#
# What this does NOT do, on purpose: it never copies a continual-harness store.
# Every experiment begins from an empty store (run_e13.sh refuses otherwise),
# and carrying one across machines would silently break that guarantee.
set -euo pipefail

HUB_REPO="${HUB_REPO:-/root/NetHack-hub}"
ENGINE_REPO="${ENGINE_REPO:-/root/NetHack-engine}"
HUB_BRANCH="${HUB_BRANCH:-exp/e8-planning-guidance}"
HUB_URL="${HUB_URL:-https://github.com/liujonathan24/NetHack-hub.git}"
ENGINE_URL="${ENGINE_URL:-https://github.com/liujonathan24/NetHack-engine.git}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n==> %s\n' "$*"; }

say "repos"
[ -d "$HUB_REPO/.git" ]    || git clone "$HUB_URL" "$HUB_REPO"
[ -d "$ENGINE_REPO/.git" ] || git clone "$ENGINE_URL" "$ENGINE_REPO"
git -C "$HUB_REPO" fetch origin --prune
git -C "$HUB_REPO" checkout "$HUB_BRANCH" 2>/dev/null \
  || git -C "$HUB_REPO" checkout -b "$HUB_BRANCH" "origin/$HUB_BRANCH"
git -C "$HUB_REPO" pull --ff-only

say "system deps, uv, venv, engine build, prime-agent (setup_sandbox.sh)"
# setup_sandbox.sh owns the heavy lifting and asserts the prime-agent version
# pin (0.3.3). That assertion is load-bearing: the harness REFUSES to launch on
# a mismatch, so if the installer has moved on, this is where you find out.
HUB_REPO="$HUB_REPO" ENGINE_REPO="$ENGINE_REPO" HUB_BRANCH="$HUB_BRANCH" \
  bash "$HUB_REPO/setup_sandbox.sh"

say "Claude Code config (settings + operational memory)"
# The memory files are the accumulated gotchas -- the daemon wedge recipe, the
# editable-install trap, git authorship. Losing them costs more than it looks.
mkdir -p "$HOME/.claude/projects/-root/memory"
cp -n "$HERE/claude-config/settings.json"       "$HOME/.claude/settings.json"       2>/dev/null || true
cp -n "$HERE/claude-config/settings.local.json" "$HOME/.claude/settings.local.json" 2>/dev/null || true
cp -n "$HERE/claude-config/memory/"*.md         "$HOME/.claude/projects/-root/memory/" 2>/dev/null || true
echo "    memory files: $(ls "$HOME/.claude/projects/-root/memory/"*.md 2>/dev/null | wc -l)"

say "verification"
export ENG="$ENGINE_REPO"
export PATH="$HOME/.local/bin:$PATH"
PY="$HUB_REPO/.venv-cli-eval/bin/python"
export PYTHONPATH="$HUB_REPO/tools/pycompat:$ENG:$HUB_REPO:$HUB_REPO/environments/nethack:$HUB_REPO/harnesses/nethack-prime-agent"

fail=0
check() { if eval "$2" >/dev/null 2>&1; then echo "    ok   $1"; else echo "    FAIL $1"; fail=1; fi; }
check "prime-agent 0.3.3"      'test "$(prime-agent --version 2>&1 | head -1)" = "0.3.3"'
check "bubblewrap present"     'command -v bwrap'
check "venv python 3.12"       "$PY --version"
check "tomllib (tier resolver needs 3.11+)" "$PY -c 'import tomllib'"
check "engine .so fresh"       "$PY $HUB_REPO/tools/cli_harness_eval/engine_provenance.py --check"
check "tier registry resolves" "$PY $HUB_REPO/tools/cli_harness_eval/tool_tiers.py flags base --arm prime_agent"
check "harness importable"     "$PY -c 'import nethack_prime_agent'"
check "prime credentials"      'test -s "$HOME/.prime/config.json"'

say "still needed by hand"
cat <<'NOTE'
  1. Prime credentials -- the ONE thing this script cannot recreate:
         prime login --headless --plain
     (or copy ~/.prime/config.json from the old box, 494 bytes)
     Then, per RUNBOOK.md and the arm configs:
         key=$(python -c "import json;print(json.load(open('$HOME/.prime/config.json'))['api_key'])")
         export PI_API_KEY="$key" PRIME_API_KEY="$key"

  2. Prior run data, if you need the reference numbers -- not in git, and the
     evidence behind every published figure:
         outputs/e10_baseline    v1 reference: median 3.54 / mean 5.34 / 14 of 15 died / dl11
         outputs/e11_gates       the descent-gate comparison
         outputs/e13/            continual-harness rounds
         outputs/e14_baseline/   the v2 baseline
     Roughly 220 MB total. rsync them into the new checkout's outputs/.

  DO NOT copy /tmp/vf-prime-agent-* -- those are continual-harness stores, and
  every experiment is designed to start from an empty one.
NOTE
[ "$fail" = 0 ] && say "bootstrap OK" || { say "bootstrap INCOMPLETE -- see FAIL lines above"; exit 1; }
