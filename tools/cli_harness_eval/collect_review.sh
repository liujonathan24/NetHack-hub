#!/usr/bin/env bash
# Assemble everything a human needs to review one E13 run, in one directory.
#
#   collect_review.sh <outputs/e13/<run>> [DEST]
#
# Reviewing a continual-harness run means answering four questions, and each has
# a different artifact:
#
#   what did the agent DO?          per-seed transcripts (tools/transcript_dump.py)
#   what did it LEARN?              store snapshots, per round, before vs after
#   what happened to disagreement?  the merge reports
#   what was actually RUN?          resolved config.toml + run manifest
set -uo pipefail

RUN_DIR="${1:?usage: collect_review.sh <outputs/e13/<run>> [dest]}"
[ -d "$RUN_DIR" ] || { echo "collect_review: no such run dir: $RUN_DIR" >&2; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${2:-$RUN_DIR/review}"
PY_BIN="${PY_BIN:-/root/NetHack-hub/.venv-cli-eval/bin/python}"
mkdir -p "$DEST"

cp "$RUN_DIR/run_manifest.json" "$DEST/" 2>/dev/null || true
[ -d "$RUN_DIR/final" ] && cp -r "$RUN_DIR/final" "$DEST/final" 2>/dev/null || true

for round_dir in "$RUN_DIR"/round*; do
  [ -d "$round_dir" ] || continue
  r="$(basename "$round_dir")"
  mkdir -p "$DEST/$r"
  cp "$round_dir/merge_report.json" "$DEST/$r/" 2>/dev/null || true
  cp "$round_dir/orchestrator.stdout.txt" "$DEST/$r/" 2>/dev/null || true
  cp "$round_dir/orchestrator_rationale.json" "$DEST/$r/" 2>/dev/null || true
  for cell in "$round_dir"/*__prime_agent; do
    [ -d "$cell" ] || continue
    c="$(basename "$cell")"
    mkdir -p "$DEST/$r/$c"
    cp "$cell/config.toml" "$cell/eval.log" "$DEST/$r/$c/" 2>/dev/null || true
    cp "$cell"/harness_state.*.json "$DEST/$r/$c/" 2>/dev/null || true
    # Per-seed plain-text transcripts: the raw record of what the model saw and
    # did, which is what a post-mortem is actually read from.
    if [ -f "$cell/traces.jsonl" ]; then
      ( cd "$REPO" && "$PY_BIN" -m tools.transcript_dump "$cell" -o "$DEST/$r/$c/transcripts" ) \
        >/dev/null 2>&1 || echo "  (transcript dump failed for $c)"
    fi
  done
done

# A single summary so a reviewer starts with numbers, not a directory listing.
"$PY_BIN" - "$RUN_DIR" "$DEST" <<'PYS'
import json, pathlib, sys
run, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lines = ["# Review summary", ""]
man = run / "run_manifest.json"
if man.exists():
    m = json.loads(man.read_text())
    lines += [f"**{m.get('run')}** — spec `{m.get('spec_sha256_16')}`, "
              f"prompt `{m.get('reflection_prompt')}` `{m.get('prompt_sha256_16')}`",
              f"corpus seeds {m.get('corpus_seeds')} · rounds {m.get('rounds')} · "
              f"players_may_edit {m.get('players_may_edit')} · mode `{m.get('store_mode')}`", ""]
lines += ["| round | games | store entries after | added | conflicts |",
          "|---|---|---|---|---|"]
for rd in sorted(run.glob("round*")):
    cells = list(rd.glob("*__prime_agent"))
    games = 0
    for c in cells:
        t = c / "traces.jsonl"
        if t.exists():
            games += sum(1 for ln in t.read_text().splitlines() if ln.strip())
    after, added, conflicts = "—", "—", "—"
    for c in cells:
        s = c / "harness_state.after.json"
        if s.exists():
            try:
                st = json.loads(s.read_text())
                after = sum(len(v) for v in st.get("entries", {}).values())
            except Exception:
                pass
    mr = rd / "merge_report.json"
    if mr.exists():
        r = json.loads(mr.read_text())
        added, conflicts = len(r.get("added", [])), len(r.get("conflicts", []))
    lines.append(f"| {rd.name} | {games} | {after} | {added} | {conflicts} |")
final = run / "final" / "FROZEN.json"
if final.exists():
    f = json.loads(final.read_text())
    lines += ["", f"**Frozen:** `{f.get('harness_state_sha256','')[:16]}` "
              f"entries {f.get('entry_counts')}"]
lines += ["", "Transcripts: `<round>/<cell>/transcripts/` (one file per seed).",
          "Store over time: `<round>/<cell>/harness_state.{before,after}.json`.",
          "Disagreements: `<round>/merge_report.json`."]
(dest / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PYS
echo
echo "[review] assembled -> $DEST"
