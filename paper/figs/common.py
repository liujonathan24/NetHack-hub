"""Shared loaders and derived series. Plot modules import from here, never from disk."""
import json, os, re, glob, collections
import numpy as np
from paths import (COMPARE, TOP10, BALROG, FIGDATA, ARM_LABELS, ASCENSION,
                   ENDGAME, ALL_GAMES, NORM_SOURCE)


def figdata():
    return json.load(open(FIGDATA))


def arm(name="base"):
    """One experimental arm's extracted rollouts and aggregate counts."""
    d = figdata()
    return d.get("arms", {}).get(name, d)


def arm_label(name):
    return ARM_LABELS.get(name, name)


def compare():
    return json.load(open(COMPARE))


def is_terminal(g):
    """The paper's human-population filter.

    A NetHack game is terminal when it ends in death or ascension, and also when
    the player escapes the dungeon or quits, since all four write a final score.
    We keep every one of them and drop only games that ended on the turn they
    began: 13,334 one-turn non-starts, 10,369 of them escapes on dungeon level 1.
    Leaves n = 46,796 of 60,130 decoded games (median BALROG-min 2.12).
    Replaces the earlier `turns >= 100` heuristic, which cut short deaths and
    kept long abandonments.
    """
    return int(g.get("turns") or 0) > 1


def balrog_table():
    """Depth and experience percentiles, as percentages."""
    d = json.load(open(BALROG))
    dl = {int(k.split(":")[1]): v * 100 for k, v in d.items() if k.startswith("Dlvl:")}
    xp = {int(k.split(":")[1]): v * 100 for k, v in d.items() if k.startswith("Xp:")}
    return dl, xp


def _invert_tables():
    """pct -> level, for each BALROG axis. The two tables share only 0.0."""
    dl, xp = balrog_table()
    D, X = {}, {}
    for k in sorted(dl):
        D[round(dl[k], 3)] = k          # deepest level carrying this percentile
    for k in sorted(xp):
        X[round(xp[k], 3)] = k
    return D, X


def _depth_xp_track(curve, D, X):
    """Running (max depth, max XL) at each stored curve point.

    A decoded game's curve stores [turn, BALROG-max, BALROG-min] rather than the
    raw (Dlvl, XL) pair, but the depth and experience percentile tables are
    disjoint apart from 0.0, so each column identifies its own axis unambiguously
    and the pair inverts exactly.
    """
    seen, cur_d, cur_x = {}, 0, 1
    for pt in curve:
        for v in (round(pt[1], 3), round(pt[2], 3)):
            if v in D and v not in X:
                cur_d = max(cur_d, D[v])
            elif v in X and v not in D:
                cur_x = max(cur_x, X[v])
        if cur_d and cur_d not in seen:
            seen[cur_d] = cur_x
    return seen


def winners_norm(maxdepth=50, minn=20, path=None):
    """Experience-for-depth norm over the NAO games that ascended.

    For every winning game, the maximum experience level it held the first time
    it stood on each depth; the norm is the median of that across winners at each
    depth. This is the population the paper's claim rests on -- trajectories that
    actually reached the Sanctum -- so the staircase is what a winning pace looks
    like, not what the median player manages.
    """
    import statistics as st
    by, n = winners_by_depth(path)
    xs = sorted(k for k in by if k <= maxdepth and len(by[k]) >= minn)
    return xs, [st.median(by[k]) for k in xs], n


def winners_by_depth(path=None):
    """{depth: [XL held on first arrival, one per ascension]} and the count.

    The distribution behind winners_norm, for callers that want a quantile
    envelope rather than only the median.
    """
    D, X = _invert_tables()
    by = collections.defaultdict(list)
    n = 0
    for g in json.load(open(path or ALL_GAMES)):
        if not is_terminal(g) or not str(g.get("death", "")).startswith("ascended"):
            continue
        n += 1
        for d, x in _depth_xp_track(g["c"], D, X).items():
            by[d].append(x)
    return by, n


def top10_norm(maxdepth=14, minn=20):
    """Median experience level at each depth reached, over the NAO top-10 games."""
    import statistics as st
    by = collections.defaultdict(list)
    for x in json.load(open(TOP10)):
        a, b = x.get("max_dlvl"), x.get("max_xp")
        if a and b:
            by[int(a)].append(int(b))
    xs = sorted(k for k in by if k <= maxdepth and len(by[k]) >= minn)
    return xs, [st.median(by[k]) for k in xs], sum(len(v) for v in by.values())


def human_norm(maxdepth=14, minn=20, source=None):
    """The experience-for-depth norm, from whichever human population is selected.

    `source="winners"` uses the NLD-NAO ascension trajectories, which is what the
    paper's text describes; `source="top10"` reproduces the original curve over
    the 12,154 DeepMind nao_top10 games.
    """
    src = source or NORM_SOURCE
    if src == "winners":
        return winners_norm(maxdepth, minn)
    if src == "top10":
        return top10_norm(maxdepth, minn)
    raise ValueError(f"unknown norm source {src!r}")


def _pct(table, v):
    ks = [k for k in table if k <= v]
    return table[max(ks)] if ks else 0.0


def e14_on_grid(g, name="base"):
    """Each rollout's running BALROG-max stepped onto grid g; NaN once it dies."""
    dl, xp = balrog_table()
    rows = []
    for r in arm(name)["rollouts"]:
        pts = sorted((t["gturn"], max(_pct(dl, t["dlvl"]), _pct(xp, t["xp"])))
                     for t in r["turns"] if "gturn" in t)
        if not pts:
            continue
        gt = np.array([p[0] for p in pts], float)
        bv = np.maximum.accumulate([p[1] for p in pts])
        idx = np.searchsorted(gt, g, side="right") - 1
        vals = np.where(idx >= 0, bv[np.clip(idx, 0, len(bv) - 1)], 0.0)
        rows.append(np.where(g <= r["end_gturn"], vals, np.nan))
    return np.array(rows)


def e14_on_grid_min(g, name="base"):
    """Each rollout's running BALROG-min stepped onto grid g; NaN once it dies.

    BALROG-min at turn t is min(depth percentile of the max Dlvl so far,
    experience percentile of the max XL so far) -- the running max is taken per
    axis first, then the min across axes.
    """
    dl, xp = balrog_table()
    rows = []
    for r in arm(name)["rollouts"]:
        pts = sorted((t["gturn"], _pct(dl, t["dlvl"]), _pct(xp, t["xp"]))
                     for t in r["turns"] if "gturn" in t)
        if not pts:
            continue
        gt = np.array([p[0] for p in pts], float)
        bv = np.minimum(np.maximum.accumulate([p[1] for p in pts]),
                        np.maximum.accumulate([p[2] for p in pts]))
        idx = np.searchsorted(gt, g, side="right") - 1
        vals = np.where(idx >= 0, bv[np.clip(idx, 0, len(bv) - 1)], 0.0)
        rows.append(np.where(g <= r["end_gturn"], vals, np.nan))
    return np.array(rows)


def arm_balrog(name="base"):
    """Final BALROG-max per rollout: the max over the depth and experience axes."""
    dl, xp = balrog_table()
    return [max(_pct(dl, r["max_dlvl"]), _pct(xp, r["max_xp"]))
            for r in arm(name)["rollouts"]]


def arm_stat(name="base"):
    """Mean BALROG and its standard error for one arm."""
    v = np.array(arm_balrog(name), float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


# ---------------- rendered-observation galleries ----------------
def _unwrap(c):
    if isinstance(c, str) and c.lstrip().startswith("{"):
        try:
            return "\n".join(p.get("text", "") for p in json.loads(c).get("content", [])
                             if isinstance(p, dict))
        except Exception:
            pass
    return c or ""


def _map_rows(rows):
    return sum(1 for r in rows
               if len(r.strip()) >= 15 and sum(r.count(c) for c in ".#|-") >= 8)


def _blocks(text, nlines):
    out = []
    for m in re.finditer(r"=== MAP[^\n]*\n", text):
        blk = text[m.end():]
        nxt = blk.find("=== ")
        blk = blk[:nxt if nxt > 0 else 2400]
        rows = [x for x in blk.split("\n") if x.strip()][:nlines]
        if rows:
            out.append((_map_rows(rows), rows))
    return out


def best_observation(path, nlines=16):
    """Richest rendered map a run ever showed, from traces.jsonl or transcripts."""
    cands = []
    if path.endswith(".jsonl") and os.path.exists(path):
        for line in open(path):
            for n in json.loads(line)["nodes"]:
                m = n["message"]
                if m.get("role") == "tool":
                    cands += _blocks(_unwrap(m.get("content", "")), nlines)
    elif os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, "*transcript*.txt")))[:5]:
            cands += _blocks(open(f, errors="ignore").read(), nlines)
    if not cands:
        return None
    score, rows = max(cands, key=lambda c: (c[0], len(c[1])))
    return rows if score >= 4 else None


# tab:continual-full: ten paired iterations, both runs, all four metrics.
# Base row is iteration 0 (the empty store on the same held-out seeds).
CURVE = {
    "mem": {
        "train_dlvl": [6.4, 8.2, 9.2, 7.4, 5, 8.4, 7.6, 5.6, 3.67, 6.8],
        "test_dlvl": [7.2, 7.4, 8, 7, 8.4, 4.6, 7.6, 8.2, 7.8, 5.8],
        "train_xl": [2.4, 2.4, 2.4, 3, 2, 1.6, 2.6, 1.4, 1.33, 3],
        "test_xl": [1.6, 2, 3.2, 3.2, 3.2, 1.6, 2.6, 2.2, 3, 2.2],
        "train_bmax": [6.14, 9.72, 13.32, 7.53, 3.36, 10.44, 7.77, 4.07, 1.89, 6.75],
        "test_bmax": [7.59, 9.55, 10.12, 8.64, 10.06, 3.15, 11.38, 11.02, 7.49, 5.51],
        "train_bmin": [1.34, 1.32, 1.34, 1.82, 1.09, 0.79, 1.99, 0.74, 0.62, 2.1],
        "test_bmin": [1.11, 0.9, 1.9, 2.11, 2.14, 0.79, 1.43, 1.27, 2.12, 1.27],
    },
    "wiki": {
        "train_dlvl": [6.8, 8.4, 4.2, 8.2, 6.2, 8.2, 6.4, 6.2, 6.2, 6],
        "test_dlvl": [7.75, 4.4, 10.6, 6.75, 8.25, 6.8, 5.4, 3.8, 5, 7],
        "train_xl": [3, 3, 1.6, 3, 2.4, 4.4, 3.2, 3.4, 3.2, 3.8],
        "test_xl": [2.75, 2.6, 3.6, 2.75, 4, 3.2, 3.4, 2.6, 3.4, 3.4],
        "train_bmax": [4.99, 11.89, 2.39, 8.92, 5.42, 9.72, 5.33, 6.65, 6.46, 6.41],
        "test_bmax": [8.02, 3.91, 16.42, 7.86, 8.68, 7.89, 3.98, 1.95, 2.85, 6.39],
        "train_bmin": [1.83, 1.83, 0.79, 2.15, 1.64, 2.61, 1.87, 1.96, 1.87, 2.44],
        "test_bmin": [2.05, 1.46, 2.28, 1.33, 2.46, 1.92, 1.93, 1.38, 2.25, 2.24],
    },
}
BASE_DL, BASE_DL_SE, BASE_XL = 8.00, 0.88, 2.33
BASE_BMAX, BASE_BMIN = 9.96, 1.32
N_ROUNDS = 10


def first_observation(path, nlines=16):
    """The observation payload a run actually showed the model.

    Encodings differ in *what* they emit, not just how a map is drawn: B0 renders an
    ASCII grid, BBOX emits feature and monster lists, BBOX_JSON emits entity JSON. So
    this returns the richest complete observation verbatim rather than hunting for a
    grid, which is what makes the panels comparable.
    """
    if not (path.endswith(".jsonl") and os.path.exists(path)):
        return None
    best, best_score = None, -1
    for line in open(path):
        for n in json.loads(line)["nodes"]:
            m = n["message"]
            if m.get("role") != "tool":
                continue
            c = _unwrap(m.get("content", ""))
            if "=== STATUS ===" not in c or "Dlvl" not in c:
                continue
            i = c.find("=== ")
            body = c[i:]
            if any(bad in body for bad in ("truncated at", "WAITING FOR INPUT",
                                           "MAP BELOW IS STALE")):
                continue
            sections = len(set(re.findall(r"=== [^=]{1,40}===", body)))
            grid = _map_rows(body.split("\n")[:nlines])
            score = sections * 100 + grid * 40 + min(len(body), 3000) / 100
            if score > best_score:
                best_score, best = score, [r for r in body.split("\n")][:nlines]
    return best


# ---------------- NAO ascensions, scored the way BALROG scores them ----------------
def balrog_specials():
    """The non-Dlvl / non-Xp achievement values: the Planes and ascension itself."""
    d = json.load(open(BALROG))
    return d.get("Astral Plane", 0.0) * 100, d.get("You ascend t", 1.0) * 100


def _lookup(table, n):
    ks = [k for k in table if k <= max(1, int(n))]
    return table[max(ks)] if ks else 0.0


def balrog_pair(dlvl, xp, planes=False, ascended=False):
    """`(max, min)` over BALROG's two progress axes, in percent.

    Faithful to `nethack_harness.prompt.balrog`: the Planes and ascension are
    terminal achievements rather than a third axis, so they are maxed into BOTH
    sides -- gating them behind the min would make a real ascension score less
    than a shallow dive. Without this credit every ascension stalls at
    Dlvl:50 = 80.68%, which is a scoring gap, not a property of the game.
    """
    astral, ascend = balrog_specials()
    if ascended:
        return ascend, ascend
    d, x = _lookup(balrog_table()[0], dlvl), _lookup(balrog_table()[1], xp)
    if planes:
        return max(d, x, astral), max(min(d, x), astral)
    return max(d, x), min(d, x)


def ascension_games(min_frac=0.9):
    """NAO games that ascended, keeping only those whose ttyrec decode is complete.

    Each game carries curve points [turn, dlvl, xp, balrog_max, balrog_min]; a game
    whose decoded turn count falls short of the xlogfile's is a truncated recording
    and is dropped.
    """
    g = json.load(open(ASCENSION))
    return [x for x in g if x["derived_turns"] >= min_frac * x["turns"]]


def endgame_states():
    """Per-game plane / ascension detections, keyed by `player/starttime`."""
    return {r["game_key"]: r for r in json.load(open(ENDGAME))}


def ascension_curves():
    """Corrected `(events, total_turns)` per ascension.

    `events` is [[turn, balrog_max, balrog_min], ...]. The stored curve's
    balrog_max/balrog_min columns are NOT used: they were written without plane
    or ascension credit, so they top out at 80.68. They are rebuilt here from the
    raw (dlvl, xp) columns plus the plane and ascension turns.
    """
    end = endgame_states()
    out = []
    for g in ascension_games():
        r = end[f"{g['player']}/{g['start']}"]
        pturn = r["first_plane_turn"] if r.get("reached_planes") else None
        aturn = r["ascension_turn"] if r.get("ascended") else None
        total = max(g["derived_turns"], aturn or 0)
        pts = [(p[0], p[1], p[2]) for p in g["curve"]]
        dl, xp = pts[-1][1], pts[-1][2]
        for extra in (pturn, aturn):
            if extra:
                pts.append((extra, dl, xp))
        pts.sort(key=lambda p: p[0])
        ev = [(t, *balrog_pair(d, x, pturn is not None and t >= pturn,
                               aturn is not None and t >= aturn)) for t, d, x in pts]
        out.append((np.array(ev, float), float(total)))
    return out


def f1_score(bmax, bmin):
    """Harmonic mean of the two BALROG percentiles; 0 when either axis is 0."""
    a, b = np.asarray(bmax, float), np.asarray(bmin, float)
    out = np.zeros_like(a + b)
    nz = (a + b) > 0
    out[nz] = 2 * a[nz] * b[nz] / (a + b)[nz]
    return out


def on_fraction(events, total, frac, col):
    """Step-function lookup of an event column at each fraction of `total` turns."""
    idx = np.searchsorted(events[:, 0], np.asarray(frac, float) * total, side="right") - 1
    return np.where(idx >= 0, events[np.clip(idx, 0, len(events) - 1), col], 0.0)
