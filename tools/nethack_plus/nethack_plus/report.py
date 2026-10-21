"""What a run did: live progress (`status`) and results (`summary`).

A run is one row: a single-mode game, or an explore run scored by its attempt
with the highest balrog_min. The summary gives each run, then the mean of each
seed's repetitions, then the mean of the per-seed means; cost is priced from
each run's tokens (every attempt and every chooser call).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from . import prime

#: Rollouts that never played: the provider or the sandbox failed, not the model.
INFRA_STOPS = ("provider_error", "sandbox_error")


def arms(run: Path) -> list[Path]:
    return sorted(p.parent for p in run.glob("*/config.json"))


def episodes(arm: Path) -> list[dict]:
    out = []
    for path in sorted(arm.glob("vf/*/traces.jsonl")):
        out += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return out


def _score(trace: dict) -> float:
    reward = (trace.get("rewards") or {}).get("balrog_min") or {}
    return float(reward.get("score") or 0.0)


def _usages(trace: dict) -> list[dict]:
    return [c["usage"] for c in trace.get("calls") or () if c.get("usage")] + list(trace.get("extra_usage") or ())


def run_row(episode: dict, price: dict[str, float]) -> dict | None:
    """One run's row: its best attempt's numbers, its cost, and how many attempts it used."""
    traces = [t for t in episode["traces"] if t.get("stop_condition") not in INFRA_STOPS]
    if not traces:
        return None
    best = max(traces, key=_score)
    m = best.get("metrics") or {}
    explore = best.get("info", {}).get("explore") or {}
    return {
        "seed": int(episode["task"]["data"]["seed"]),
        "score": _score(best),
        "dlvl": int(m.get("max_dlvl") or 0),
        "xl": int(m.get("max_XL") or 0),
        "turns": int(m.get("game_turns") or 0),
        "calls": int(m.get("calls_graded") or 0),
        "died": bool(m.get("died")),
        "replay": bool(m.get("replay_matches_live")),
        "stop": best.get("stop_condition") or "",
        "attempt": f"{explore.get('attempt', 0)}/{len(traces)}" if explore else "",
        "cost": sum(prime.cost(u, price) for t in episode["traces"] for u in _usages(t)),
        "infra_failed": len(episode["traces"]) - len(traces),
    }


def summary(run: Path) -> str:
    manifest = json.loads((run / "manifest.json").read_text())
    price = prime.prices(manifest["model"])
    lines = [
        f"run {run.name}: {manifest['mode']}, {manifest['model']}, seeds {manifest['seeds']}, reps {manifest['reps']}"
        + (f", attempts {manifest['attempts']}" if manifest["mode"] == "explore" else "")
    ]
    for arm in arms(run):
        rows = [r for e in episodes(arm) if (r := run_row(e, price)) is not None]
        lines += ["", f"== {arm.name}: {len(rows)} runs"]
        if not rows:
            continue
        lines.append(
            f"{'seed':>4} {'score':>7} {'Dlvl':>4} {'XL':>3} {'turns':>6} {'calls':>5} {'died':>4} {'replay':>6} {'$':>7}  attempt  stop"
        )
        for r in sorted(rows, key=lambda r: r["seed"]):
            lines.append(
                f"{r['seed']:>4} {r['score']:>7.3f} {r['dlvl']:>4} {r['xl']:>3} {r['turns']:>6} {r['calls']:>5} "
                f"{'y' if r['died'] else 'n':>4} {'ok' if r['replay'] else 'NO':>6} {r['cost']:>7.2f}  "
                f"{r['attempt']:>7}  {r['stop']}"
            )
        by_seed = defaultdict(list)
        for r in rows:
            by_seed[r["seed"]].append(r)
        lines.append(
            f"{'seed':>4} {'n':>3} {'score':>7} {'Dlvl':>5} {'XL':>4} {'turns':>7} {'calls':>6} {'died':>6} {'$':>7}"
        )
        means = []
        for seed, rs in sorted(by_seed.items()):
            mean = {k: sum(r[k] for r in rs) / len(rs) for k in ("score", "dlvl", "xl", "turns", "calls")}
            means.append(mean)
            died = sum(r["died"] for r in rs)
            lines.append(
                f"{seed:>4} {len(rs):>3} {mean['score']:>7.3f} {mean['dlvl']:>5.1f} {mean['xl']:>4.1f} "
                f"{mean['turns']:>7.0f} {mean['calls']:>6.0f} {died:>3}/{len(rs):<2} {sum(r['cost'] for r in rs):>7.2f}"
            )
        overall = {k: sum(m[k] for m in means) / len(means) for k in means[0]}
        lines.append(
            f"overall ({len(means)} seeds, mean of per-seed means): score {overall['score']:.3f}  Dlvl {overall['dlvl']:.2f}  "
            f"XL {overall['xl']:.2f}  turns {overall['turns']:.0f}  calls {overall['calls']:.0f}  "
            f"total ${sum(r['cost'] for r in rows):.2f}"
        )
        failed = sum(r["infra_failed"] for r in rows)
        if failed:
            lines.append(f"({failed} attempts lost to provider or sandbox errors are left out)")
    lines += ["", "cost is priced from tokens at Prime's listed rates; the wallet is the billed truth"]
    if "balance_before" in manifest:
        lines.append(
            f"wallet: ${manifest['balance_before']:.2f} before the run, ${prime.balance():.2f} now (other spending included)"
        )
    return "\n".join(lines)


def status(run: Path, idle_after: float = 300.0) -> str:
    """One line per game seen in the tool servers' progress logs, newest state first."""
    lines = [
        f"{'arm':<9} {'seed':>4} {'att':>3} {'calls':>5} {'Dlvl':>4} {'HP':>7} {'XL':>3} {'turn':>6} {'idle':>6}  state"
    ]
    now = time.time()
    for arm in arms(run):
        for log in sorted((arm / "progress").glob("*.ndjson")):
            entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            if not entries:
                continue
            last = entries[-1]
            idle = now - last["t"]
            # The tool server runs on this host for as long as its game is being played.
            alive = Path(f"/proc/{last['pid']}").exists()
            if last.get("terminated"):
                state = "game over"
            elif not alive:
                state = "ended"
            else:
                state = "idle?" if idle > idle_after else "playing"
            hp = f"{last.get('hp', '-')}/{last.get('max_hp', '-')}"
            lines.append(
                f"{arm.name:<9} {last['seed']:>4} {last.get('attempt', 0):>3} {last.get('calls') or 0:>5} "
                f"{last.get('dlvl', '-'):>4} {hp:>7} {last.get('xl', '-'):>3} {last.get('turn', '-'):>6} "
                f"{idle / 60:>5.1f}m  {state}"
            )
    return "\n".join(lines)
