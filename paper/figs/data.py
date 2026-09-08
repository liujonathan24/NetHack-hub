#!/usr/bin/env python3
"""Pull per-turn records out of the E14 base-harness traces into one compact JSON.

One record per LM turn: what tool was called, whether the returned observation
carried a map, and the status line the game reported. Everything the figures
need comes from here, so the plots never touch the 50 GB trace tree directly.
"""
import json, re, os, glob, collections

from paths import E14, E14_ARMS, FIGDATA

STATUS = re.compile(
    r"HP:\s*(\d+)/(\d+)\s+AC:\s*(-?\d+)\s+Dlvl:\s*(\d+)\s+Turn:\s*(\d+)\s+XP:\s*(\d+)")
CALLRE = re.compile(r"nethack\.(\w+)\s*\(")


def interrupt_reason(text):
    """Canonical bucket for why a skill stopped, from the engine's own wording."""
    t = text.lower()
    if "no valid path" in t:
        return "no valid path"
    m = re.search(r"interrupting skill to rethink because '([^']*)", t)
    if not m:
        return "other" if "interrupt" in t else None
    r = m.group(1)
    if "health is low" in r or "hp is" in r:
        return "HP change"
    if "dungeon level" in r or "changed dung" in r:
        return "level change"
    if "teleport" in r:
        return "teleported"
    for k in ("gold", "corpse", "boulder", "statue", "fountain", "door", "stairs"):
        if k in r:
            return k
    if "appear" in r:
        return "item / monster appeared"
    return "other"


def extract_arm(reps):
    rollouts = []
    for d, rep in reps.items():
        path = os.path.join(E14, d, "traces.jsonl")
        if not os.path.exists(path):
            print("MISSING", path)
            continue
        for line in open(path):
            r = json.loads(line)
            met = r.get("metrics", {}) or {}
            pending, turns = [], []
            for n in r["nodes"]:
                m = n["message"]
                role = m.get("role")
                if role == "assistant":
                    names = []
                    for tc in (m.get("tool_calls") or []):
                        try:
                            code = json.loads(tc["arguments"]).get("code", "")
                        except Exception:
                            code = str(tc.get("arguments", ""))
                        names += CALLRE.findall(code)
                    pending = names
                elif role == "tool":
                    c = m.get("content", "") or ""
                    if not re.search(r"Dlvl[: ]", c):
                        continue
                    st = STATUS.search(c)
                    rec = {"tools": pending,
                           "has_map": "=== MAP ===" in c,
                           "interrupt": interrupt_reason(c)}
                    if st:
                        hp, mhp, ac, dl, gt, xp = (int(x) for x in st.groups())
                        rec.update(hp=hp, maxhp=mhp, dlvl=dl, gturn=gt, xp=xp)
                    turns.append(rec)
                    pending = []
            tok = sum((c.get("usage") or {}).get("prompt_tokens", 0)
                      + (c.get("usage") or {}).get("completion_tokens", 0)
                      for c in r.get("calls", []))
            gts = [t["gturn"] for t in turns if "gturn" in t]
            rollouts.append({
                "seed": r["task"]["data"].get("seed"), "rep": rep,
                "max_dlvl": int(met.get("max_dlvl_reached", 0)),
                "max_xp": max([t["xp"] for t in turns if "xp" in t] or [1]),
                "died": bool(met.get("died", 0)),
                "skill_calls": int(met.get("skill_calls", 0)),
                "tokens": tok,
                "end_gturn": max(gts) if gts else 0,
                "turns": turns,
            })
    rollouts.sort(key=lambda x: (x["seed"], x["rep"]))
    return rollouts


def summarise(rollouts):
    n = max(1, sum(len(r["turns"]) for r in rollouts))
    return {
        "rollouts": rollouts,
        "n_rollouts": len(rollouts),
        "n_turns": sum(len(r["turns"]) for r in rollouts),
        "tool_counts": dict(collections.Counter(
            t for r in rollouts for tr in r["turns"] for t in tr["tools"]).most_common()),
        "interrupt_counts": dict(collections.Counter(
            tr["interrupt"] for r in rollouts for tr in r["turns"]
            if tr["interrupt"]).most_common()),
        "map_share": round(100 * sum(tr["has_map"] for r in rollouts
                                     for tr in r["turns"]) / n, 1),
    }


def main():
    arms = {name: summarise(extract_arm(reps)) for name, reps in E14_ARMS.items()}
    out = dict(arms["base"])           # base stays at the top level for single-arm callers
    out["arms"] = arms
    json.dump(out, open(FIGDATA, "w"))

    for name, a in arms.items():
        died = sum(r["died"] for r in a["rollouts"])
        print(f"{name:>6}: {a['n_rollouts']} rollouts, {a['n_turns']} observations, "
              f"map {a['map_share']}%, deaths {died}/{a['n_rollouts']}")
        print(f"         tools {list(a['tool_counts'].items())[:4]}")
        print(f"         interrupts {list(a['interrupt_counts'].items())[:4]}")


if __name__ == "__main__":
    main()
