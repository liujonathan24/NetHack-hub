#!/usr/bin/env bash
# MANDATORY before any paid batch: resolve the config, mock-play one seed, and
# read back what actually ran.
#
#   preflight_cell.sh <stack> [outdir]
#
# The mock play is a real rollout on a SHORT budget (default 3 calls, 1 seed) --
# long enough to produce a resolved config.toml and a first observation, short
# enough to be noise against a 20-game batch. Override MOCK_MODEL to run it on a
# cheaper model than the batch; the point is the plumbing, not the score.
#
# It exists because every expensive mistake in this series was a config that
# claimed one thing and ran another: tune.reveal_map and auto_dismiss silently
# dropped so "control" cells ran fog'd, a stale harness package rejecting a flag,
# a trace_dir resolved against a workdir that teardown deletes. None were visible
# in the launch command; all were visible in the resolved config.toml and the
# first turn record.
set -uo pipefail

STACK="${1:?usage: preflight_cell.sh <stack> [outdir]}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${2:-$REPO/outputs/e13_preflight/$(echo "$STACK" | tr '+' '_')}"
export PYTHONPATH="${REPO}/harnesses/nethack-prime-agent${PYTHONPATH:+:${PYTHONPATH}}"
export ENG="${ENG:-/root/NetHack-engine}"
export EVAL_BIN="${EVAL_BIN:-/root/NetHack-hub/.venv-cli-eval/bin/eval}"
PY_BIN="$(dirname "${EVAL_BIN}")/python"
TIERS="$PY_BIN $REPO/tools/cli_harness_eval/tiers.py"
MOCK_CALLS="${MOCK_CALLS:-3}"
cd "$REPO"

echo "[pre  ] stack=$STACK out=$OUT"

# 1. The tier file must be able to produce this stack on this tree.
if ! $TIERS check "$STACK"; then
  echo "[pre  ] FAILED at tier check -- do not launch the batch." >&2
  exit 5
fi

# 2. Resolve.
ENV_ARGS_JSON="$($TIERS env-args "$STACK")" || exit 5
TIER_HASH="$($PY_BIN -c "import sys;sys.path.insert(0,'$REPO/tools/cli_harness_eval');import tiers;print(tiers.tier_hash())")"
echo "[pre  ] env_args: $ENV_ARGS_JSON"
echo "[pre  ] tier_hash: $TIER_HASH"

# 3. Mock-play one seed.
prime-agent shutdown >/dev/null 2>&1 || true
pkill -9 -x prime-agent 2>/dev/null || true
pkill -9 -x nethack_v1 2>/dev/null || true
rm -rf /tmp/prime-agent-0 2>/dev/null || true
sleep 2
rm -rf "$OUT"; mkdir -p "$OUT"
# shellcheck disable=SC2046
env $($TIERS flags "$STACK") ENV_ARGS="$ENV_ARGS_JSON" VARIANT=BBOX_MIN \
  ${MOCK_MODEL:+MODEL="$MOCK_MODEL"} \
  "$REPO/tools/cli_harness_eval/launch_cell.sh" prime_agent "$OUT" "$MOCK_CALLS" 1
rc=$?
if [ "$rc" != "0" ]; then
  echo "[pre  ] mock play exited $rc -- do not launch the batch." >&2
  exit 6
fi

# 4. Read it back.
"$PY_BIN" "$REPO/tools/cli_harness_eval/preflight_cell.py" "$OUT" \
  --stack "$STACK" --tier-hash "$TIER_HASH" || exit 7

echo "[pre  ] OK -- $STACK is safe to launch."
