#!/usr/bin/env bash
# Copy this verification's scripts, prompts, artifacts, transcripts and rollout
# evidence from the scratchpad into the repo, ready to commit.
#
# WHY THE SCRATCHPAD IS THE SOURCE OF TRUTH. The first pass of this work wrote
# straight into `outputs/e16_sim/`; at 10:31 a concurrent agent removed the
# directory (a `git clean`-shaped event -- the tree is shared and the directory
# was untracked) and took every script and transcript with it. Only the
# prime-agent session files, which live outside the repo, survived, and the
# transcripts were rebuilt from them. Work outside the shared worktree; copy in
# at commit time.
#
# Rollout traces are large; only the evidence files are staged, not `turns/`.
set -euo pipefail
SIM="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST=/root/nld/zombie-fix/outputs/e16_sim

mkdir -p "$DEST"/{scripts,artifacts,transcripts,rollouts}
cp "$SIM"/scripts/*.py "$SIM"/scripts/*.sh "$SIM"/scripts/*.txt "$DEST/scripts/"
cp "$SIM"/artifacts/*.txt "$DEST/artifacts/"
cp -r "$SIM"/artifacts/archive "$DEST/artifacts/" 2>/dev/null || true
cp -r "$SIM"/artifacts/cf "$DEST/artifacts/" 2>/dev/null || true
cp "$SIM"/transcripts/*.json "$DEST/transcripts/"
[ -f "$SIM/arm_comparison.json" ] && cp "$SIM/arm_comparison.json" "$DEST/"

for r in "$SIM"/rollouts/*/; do
  name=$(basename "$r")
  mkdir -p "$DEST/rollouts/$name"
  for f in traces.jsonl config.toml directive.txt e16_args.json \
           engine_provenance.json eval.log stall_watchdog.log \
           served_bytes_report.json compliance.json; do
    [ -f "$r$f" ] && cp "$r$f" "$DEST/rollouts/$name/"
  done
done

du -sh "$DEST"
find "$DEST" -type f | wc -l
