"""Extract BALROG-vs-turn curves from our own rollout traces.

Trace rows carry `status.time` (game turn), `status.depth` and
`status.experience_level`, so an agent rollout produces exactly the same curve
shape as a decoded human game -- which is what makes the three populations
comparable on one axis.
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, "/root/NetHack-hub/environments/nethack")
from nethack_harness.prompt.balrog import balrog_both  # noqa: E402

HUB = pathlib.Path("/root/nld/hub-eval")  # the exp/cli-harness-eval worktree


def cell_config(trace_file: pathlib.Path) -> dict:
    """Nearest config.toml above the trace file (cell dir), lightly parsed."""
    for up in list(trace_file.parents)[:4]:
        cfg = up / "config.toml"
        if cfg.exists():
            txt = cfg.read_text()
            def g(key):
                m = re.search(rf'^{key}\s*=\s*"([^"]*)"', txt, re.M)
                return m.group(1) if m else None
            name = up.name
            path = str(trace_file)
            arm = ("claude_code" if "claude_code" in name or "/claude_code/" in path
                   else "prime_agent" if "prime_agent" in name or "/prime_agent/" in path
                   else "control" if "control" in name or "/control/" in path
                   else "prime_agent")
            reveal = g("reveal_map")
            vision = ("on" if (reveal and float(reveal) > 0) or "vison" in name
                      else "off" if "visoff" in name or "fog" in up.parent.name
                      else None)
            return {
                "cell": name,
                "run": up.parent.name,
                "model": g("model"),
                "variant": g("variant"),
                "skill_set": g("skill_set"),
                "interface": g("interface"),
                "max_turns": g("max_turns"),
                "arm": arm,
                "vision": vision,
            }
    return {"cell": trace_file.parent.parent.name, "run": "?", "model": None,
            "arm": "prime_agent", "vision": None}


def curve_from_trace(path: pathlib.Path) -> dict | None:
    pts, max_dl, max_xp, n = [], 0, 0, 0
    calls = 0
    for line in path.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        calls += 1
        st = row.get("status")
        if not isinstance(st, dict):
            continue
        t = st.get("time")
        dl = st.get("depth")
        xp = st.get("experience_level")
        if t is None or dl is None or xp is None:
            continue
        n += 1
        max_dl, max_xp = max(max_dl, int(dl)), max(max_xp, int(xp))
        mx, mn = balrog_both(min(max_dl, 50), max_xp)
        pt = [int(t), round(mx * 100, 3), round(mn * 100, 3)]
        if pts and pts[-1][1:] == pt[1:]:
            pts[-1] = pt
        else:
            pts.append(pt)
    if not pts:
        return None
    cfg = cell_config(path)
    return {
        **cfg,
        "file": str(path.relative_to(HUB)),
        "seed": path.name.split("_")[0],
        "calls": calls,
        "n_status_rows": n,
        "turns": pts[-1][0],
        "max_dlvl": max_dl,
        "max_xp": max_xp,
        "balrog_max": pts[-1][1],
        "balrog_min": pts[-1][2],
        "c": pts,
    }


def main():
    files = sorted(HUB.glob("outputs/**/*.ndjson"))
    print(f"{len(files)} trace files", flush=True)
    out = []
    for f in files:
        try:
            c = curve_from_trace(f)
        except Exception as e:
            print("  skip", f.name, repr(e))
            continue
        if c:
            out.append(c)
    print(f"{len(out)} rollouts with a usable curve")

    print("models:", dict(collections.Counter(c["model"] for c in out)))
    print("arms:", dict(collections.Counter(c["arm"] for c in out)))
    print("vision:", dict(collections.Counter(c["vision"] for c in out)))
    by_cell = collections.Counter(f"{c['run']}/{c['cell']}" for c in out)
    print("cells:", len(by_cell))
    played = [c for c in out if c["turns"] > 1]
    print(f"rollouts that moved the clock past turn 1: {len(played)}")
    if played:
        import statistics
        print("turns   median", statistics.median([c["turns"] for c in played]),
              " max", max(c["turns"] for c in played))
        print("balrog_max mean %.2f  median %.2f  best %.2f" % (
            statistics.fmean([c["balrog_max"] for c in played]),
            statistics.median([c["balrog_max"] for c in played]),
            max(c["balrog_max"] for c in played)))
        print("balrog_min mean %.2f  median %.2f" % (
            statistics.fmean([c["balrog_min"] for c in played]),
            statistics.median([c["balrog_min"] for c in played])))
    pathlib.Path("/root/nld/agent_curves.json").write_text(json.dumps(out))
    print("wrote /root/nld/agent_curves.json")


if __name__ == "__main__":
    main()
