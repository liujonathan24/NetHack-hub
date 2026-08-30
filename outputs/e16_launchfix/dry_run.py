#!/usr/bin/env python3
"""E16 no-inference dry run: the whole loop, real engine, real archive.

Runs the production Orchestrator against a STUB player that plays real engine
steps through the audited restore/save path and reports a canned summary. No
model is called anywhere -- the orchestrator's session is a scripted runner and
the player is a function -- so this exercises selection, restore, save, ingest,
lesson-writing, budget accounting, compliance scoring, pairing and every
artifact the run writes, for nothing.

Three arms, because the point of this launch fix is that the third one exists:

  A  go_explore, LLM decision seat, 4 attempts   (the method)
  B  go_explore + --paired-control, 4 attempts   (each directive vs its own
                                                  control, 2 pairs)
  C  matched_restart, 4 attempts                 (THE NULL: N tries from one
                                                  fixed state, no archive, no
                                                  selection, no directive, no
                                                  lessons)

Usage: dry_run.py <out-dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/root/nld/zombie-fix")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "environments" / "nethack"))
sys.path.insert(0, str(REPO / "tools" / "cli_harness_eval"))

import e16_orchestrator as E  # noqa: E402
import e16_session as S  # noqa: E402
from nethack_harness.checkpoints import (  # noqa: E402
    checkpoint_list, checkpoint_meta, checkpoint_restore, checkpoint_save,
)

WIKI_SRC = Path("/root/nld/e15-wiki/configs/continual/wiki")

OUTCOMES = [
    {"died": True, "stop_condition": "died", "spend": 2.0,
     "calls": 25, "player_calls": [{"name": "np_move_to",
                                    "args": {"x": 5, "y": 5, "why": "descend"}}]},
    {"died": False, "stop_condition": "harness_timeout", "spend": 3.0,
     "calls": 40, "player_calls": [{"name": "np_explore_level", "args": {}}]},
    {"died": True, "stop_condition": "died", "spend": 1.0,
     "calls": 12, "player_calls": [{"name": "np_press_key", "args": {"key": "j"}}]},
    {"died": False, "stop_condition": "agent_completed", "spend": 0.5,
     "calls": 3, "player_calls": []},
]


class StubPlayer:
    """Real engine steps, real restore, real save. No model."""

    def __init__(self):
        self.seen = []

    def __call__(self, ctx):
        self.seen.append(ctx)
        env, _ = checkpoint_restore(ctx.checkpoint_dir,
                                    fidelity_log=ctx.fidelity_log,
                                    reseed=ctx.reseed)
        for _ in range(6):
            env.step(ord("s"))
        env.modify(gold=1000 * ctx.attempt)
        env.step(27)
        saved = checkpoint_save(env, ctx.archive_dir / f"c{100 + ctx.attempt}",
                                name=f"attempt {ctx.attempt} state",
                                note="stub player checkpoint", created_by="save")
        spec = OUTCOMES[(ctx.attempt - 1) % len(OUTCOMES)]
        return E.PlayerResult(
            stop_condition=spec["stop_condition"], died=spec["died"],
            calls=spec["calls"], spend_usd=spec["spend"],
            max_dlvl=saved["dlvl"], max_xl=saved["xl"],
            summary=f"attempt {ctx.attempt}: searched six turns and stopped",
            lesson="search less, descend more",
            raw={"calls": spec["player_calls"],
                 "metrics": {"descent_count": 0, "budget_exhausted":
                             not spec["died"]}})


#: The opening plan the scripted orchestrator "writes". Long and varied enough
#: to pass `detect_degeneration` -- which is the point of it being here rather
#: than a one-liner: a dry run whose opening plan would be REFUSED by the real
#: loop is not exercising the real loop.
DRY_RUN_PLAN = """\
PLAN for this seed.

Prefer the deepest checkpoint whose HP is healthy over the deepest checkpoint
outright: resuming into a fight already lost spends the whole attempt.

The wiki's early-game rules I will steer on: flee at half HP, never melee a
floating eye, stay unburdened, do not eat old corpses.

The wall on this seed is the mid-game gap, so the archive should be dense
early -- cheap states are cheap to re-reach -- and I will re-select a state
that has been tried twice without an advance only if nothing shallower looks
better.

What I will watch: whether outcome prohibitions are followed more often than
tool prohibitions, and whether depth per attempt falls off as we go deeper.
"""


def scripted_runner(directives):
    """A stand-in for `prime-agent --print`, FAITHFUL TO THE MODE IT IS ASKED FOR.

    THE UNFAITHFULNESS THIS FIXES, and it is why this dry run passed while the
    real launch broke. This stub used to emit `--mode json` records on every
    round. The session only runs json mode on the DISCOVERY round; every round
    after it is plain text, and plain `--print` writes assistant text and
    nothing else. So the dry run drove the json record parser on rounds the
    real run drives as text, and the text path -- where the pilot's directive
    JSON was silently eaten as an unrecognised protocol record -- was never
    exercised by anything but production.
    """
    it = iter(directives)
    n = {"i": 0}

    def runner(argv, env, cwd, timeout_s):
        n["i"] += 1
        if n["i"] == 1:
            body = DRY_RUN_PLAN
        else:
            try:
                d = next(it)
            except StopIteration:
                d = "hold position and search the walls"
            body = "Choosing the deepest healthy state.\n" + json.dumps(
                {"checkpoint": "1", "directive": d, "rationale": "deepest"})
        if "--mode" not in argv or argv[argv.index("--mode") + 1] != "json":
            return body + "\n", "", 0
        return ("\n".join([
            json.dumps({"type": "session", "version": 3, "id": "dryrun-sess",
                        "cwd": cwd}),
            json.dumps({"type": "message",
                        "message": {"role": "assistant", "content": body},
                        "usage": {"prompt_tokens": 4000,
                                  "completion_tokens": 300,
                                  "cached_input_tokens": 2000}})]) + "\n", "", 0)
    return runner


def arm(out: Path, label: str, **cfgkw):
    run = out / label
    cfg = E.OrchestratorConfig(run_dir=run, wiki_src=WIKI_SRC,
                               budget_ceiling_usd=1000.0, max_attempts=4,
                               milestone_dlvl=99, milestone_dungeon=-1,
                               **cfgkw)
    session = None
    if cfg.selector == "llm":
        cfg.orchestrator_dir.mkdir(parents=True, exist_ok=True)
        session = S.PrimeAgentSession(
            work_dir=cfg.orchestrator_dir,
            agent_dir=cfg.orchestrator_dir / "agent",
            log_path=cfg.orchestrator_log,
            runner=scripted_runner([
                "descend to D3 and take the down staircase immediately",
                "do not clear this level; reach XL 2 then descend",
                "engrave Elbereth before engaging anything",
                "do not explore; go straight to the stairs",
            ]))
    player = StubPlayer()
    orch = E.Orchestrator(cfg, player, session=session)
    orch.prepare()
    E.seed_archive(cfg)
    summary = orch.run(max_attempts=4)
    return cfg, orch, player, summary


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    print("=" * 78)
    print("E16 NO-INFERENCE DRY RUN -- real engine, real archive, stub player")
    print("=" * 78)

    for label, kw in (
        ("A_go_explore", dict(selector="llm")),
        ("B_paired_control", dict(selector="llm", paired_control=True)),
        ("C_matched_restart", dict(selector="scripted",
                                   arm=E.ARM_MATCHED_RESTART,
                                   stall_attempts=2)),
    ):
        cfg, orch, player, summary = arm(out, label, **kw)
        prov = json.loads(cfg.provenance_path.read_text())
        rows = E.ledger_rows(cfg.archive_dir)
        fid = [json.loads(l) for l in
               cfg.fidelity_path.read_text().splitlines() if l.strip()]
        print(f"\n--- ARM {label} "
              f"(experiment_arm={prov['experiment_arm']}) ---")
        print(f"  attempts            : {summary['attempts']} "
              f"(died {summary['attempts_died']}, "
              f"censored {summary['attempts_censored']})")
        print(f"  censor reasons      : {summary['censor_reasons']}")
        print(f"  stop_reason         : {summary['stop_reason']}")
        print(f"  archive             : {sorted(p.name for p in checkpoint_list(cfg.archive_dir))}")
        print(f"  restores audited    : {len(fid)} (all ok: {all(r['ok'] for r in fid)})")
        print(f"  budget              : players ${summary['budget']['player_usd']:.2f} "
              f"+ orchestrator ${summary['budget']['orchestrator_usd']:.4f} "
              f"= ${summary['budget']['total_usd']:.2f}")
        print(f"  selection sources   : {summary['selection_sources']}")
        print(f"  compliance headline : {summary['directive_compliance']}")
        print(f"  compliance by clause: {summary['compliance_by_clause_kind']}")
        print(f"  directive kinds     : {summary['directive_kinds']}")
        print(f"  directive lint      : {summary['directive_lint']}")
        print(f"  instrumental ambig. : {summary['instrumental_ambiguities']}")
        mp = summary["matched_pairs"]
        print(f"  matched pairs       : enabled={mp['enabled']} "
              f"complete={mp['complete_pairs']} "
              f"incomplete={mp['incomplete_pairs']}")
        if mp["by_directive_kind"]:
            for k, v in sorted(mp["by_directive_kind"].items()):
                print(f"      kind {k:<28} n={v['pairs']} "
                      f"mean dDlvl={v['mean_dlvl_delta']:+.2f} "
                      f"treatment compliance={v['treatment_compliance']}")
        print(f"  reseed_on_restore   : {prov['reseed_on_restore']}")
        print(f"  orch cwd recorded   : {prov['orchestrator_session']['cwd']}")
        print(f"  orch TMPDIR         : {prov['orchestrator_session']['tmpdir']}")
        print(f"  orch json policy    : {prov['orchestrator_session']['json_mode_policy']}")
        print(f"  orch round deadline : {prov['orchestrator_session']['round_timeout_s']}s")
        print(f"  prefix continuity   : {prov['prefix_continuity']} "
              f"(served-bytes verified: "
              f"{prov['prefix_continuity_verified_in_served_bytes']})")
        print(f"  best state          : {summary.get('best_state')}")
        report[label] = {
            "attempts": summary["attempts"],
            "stop_reason": summary["stop_reason"],
            "experiment_arm": prov["experiment_arm"],
            "budget": summary["budget"],
            "matched_pairs": mp,
            "compliance_by_clause_kind": summary["compliance_by_clause_kind"],
            "directive_kinds": summary["directive_kinds"],
            "directive_lint": summary["directive_lint"],
            "restores_audited": len(fid),
            "all_restores_ok": all(r["ok"] for r in fid),
        }
        # Every artifact the run promises must exist and parse.
        for name in ("attempts.jsonl", "selection.jsonl", "summary.json",
                     "luck.json", "provenance.json", "restore_fidelity.jsonl"):
            p = cfg.run_dir / name
            assert p.is_file() and p.stat().st_size > 0, f"{label}: missing {name}"

        # THE DIRECTIVE ACTUALLY ARRIVED, checked from the record rather than
        # inferred from the loop having run. This is the pilot's failure, and
        # this dry run used to pass straight through it: with a stub that
        # answered json on every round the directive JSON was never parsed out
        # of a TEXT round, which is what every round after the first is.
        if cfg.selector == "llm":
            attempts = [json.loads(l) for l in
                        cfg.attempts_path.read_text().splitlines() if l.strip()]
            treatments = [a for a in attempts if a["pair_role"] != "control"]
            assert treatments, f"{label}: no treatment attempts"
            for a in treatments:
                assert a["selection_source"] == "llm", (
                    f"{label}: attempt {a['attempt']} fell back to the scripted "
                    f"selector -- the LM decided nothing "
                    f"({(a.get('orchestrator_decision') or {}).get('fallback_reason')})")
                assert "orchestrator produced no directive" not in a["directive"], (
                    f"{label}: attempt {a['attempt']} was served the "
                    f"no-directive placeholder; this arm is its own control")
                assert a["directive_kind"] != "none", \
                    f"{label}: attempt {a['attempt']} carried no scoreable directive"
            print(f"  directive served    : {treatments[0]['directive'][:60]!r} "
                  f"(kind {treatments[0]['directive_kind']})")
            # ...and the opening plan was checked for degeneration, either way.
            degen = json.loads(
                (cfg.orchestrator_dir / "opening_degeneration.json").read_text())
            assert not degen[-1]["degeneration"]["degenerate"]
            print(f"  opening plan        : "
                  f"{degen[-1]['degeneration']['chars']} chars, "
                  f"unique-line ratio "
                  f"{degen[-1]['degeneration']['unique_line_ratio']} "
                  f"(degenerate: "
                  f"{degen[-1]['degeneration']['degenerate']})")
            report_extra = {
                "all_selections_llm": True,
                "opening_degeneration": degen[-1]["degeneration"],
            }
        else:
            report_extra = {}
        report[label].update(report_extra)

    # THE COMPARISON THE NULL ARM EXISTS FOR.
    a, c = report["A_go_explore"], report["C_matched_restart"]
    print("\n" + "=" * 78)
    print("MATCHED-N COMPARISON (the null the method must beat)")
    print(f"  go_explore     : N={a['attempts']}, "
          f"${a['budget']['total_usd']:.2f}")
    print(f"  matched_restart: N={c['attempts']}, "
          f"${c['budget']['total_usd']:.2f}")
    print(f"  same N         : {a['attempts'] == c['attempts']}")
    print("  Both arms emit the same attempt-record shape and draw on the same")
    print("  budget accounting, which is the only reason they are comparable.")
    print("=" * 78)
    (out / "dry_run_report.json").write_text(json.dumps(report, indent=2,
                                                        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
