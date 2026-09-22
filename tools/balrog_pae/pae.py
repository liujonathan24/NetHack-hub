"""PAE (checkpointed exploration) layered on BALROG's naive agent.

One run = N attempts on ONE episode (one game, one task, one seed).
Attempt 1 is a plain BALROG episode. Every later attempt resumes a saved
checkpoint: the env state is restored, the player's conversation history is
restored to exactly what it was at that checkpoint, and (unless ablated away)
one directive message from the orchestrator is inserted before the current
observation.

Ablation switches (the three used on NetHack):
  --select fixed|orchestrator   fixed = latest checkpoint of the previous attempt
  --directive on|off            off = no extra message at all
  --blind                       the extra message is the neutral "Continue playing."
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from .adapters import make_adapter, obs_digest
from .orchestrator import Orchestrator
from .player import PAEAgent, dump_prompt_state, load_prompt_state, messages_to_json, simulate_next_prompt
from .prime_client import MODEL_ID, Accountant, prime_client_factory

NEUTRAL_DIRECTIVE = "Continue playing."


@dataclass
class Config:
    game: str = "minihack"
    task: str = "MiniHack-Quest-Easy-v0"
    seed: int = 0
    attempts: int = 10
    checkpoint_every: int = 10
    plateau: int = 4
    select: str = "orchestrator"      # fixed | orchestrator
    directive: str = "on"             # on | off
    blind: bool = False
    max_steps: int | None = None      # None -> env default (BALROG protocol)
    model_id: str = MODEL_ID
    temperature: float = 1.0
    max_tokens: int = 8192
    run_dir: str = "runs/smoke"
    base_only: bool = False           # 1 unchanged BALROG episode, no PAE
    verify_restore: bool = True


def _jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


class Run:
    def __init__(self, cfg: Config, balrog_cfg):
        self.cfg = cfg
        self.bcfg = balrog_cfg
        self.dir = Path(cfg.run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.acct = Accountant(cfg.model_id)
        self.adapter = make_adapter(cfg.game, cfg.task, balrog_cfg)
        self.adapter.make()
        self.archive: list[dict] = []          # ledger rows (no blobs)
        self.blobs: dict[str, dict] = {}       # id -> full checkpoint
        self.attempts: list[dict] = []
        self.orch = None
        if cfg.select == "orchestrator" or (cfg.directive == "on" and not cfg.blind):
            self.orch = Orchestrator(cfg.game, cfg.task, self.acct, cfg.model_id, cfg.temperature, cfg.max_tokens)
        self.t0 = time.time()
        self.best = {"progression": -1.0, "aux": -1.0}
        try:
            self.step_cap = cfg.max_steps or self.adapter.max_steps
        except Exception:  # noqa: BLE001
            self.step_cap = cfg.max_steps or 100

    def resumable(self, ids=None):
        """Checkpoints that still have steps left under the episode cap.

        A checkpoint taken at the cap cannot be resumed: BALROG's horizon counts
        committed-trajectory steps, so resuming it would play zero steps.
        """
        pool = self.archive if ids is None else [c for c in self.archive if c["id"] in set(ids)]
        return [c for c in pool if c["step"] < self.step_cap]

    # -- agent ------------------------------------------------------------
    def _new_agent(self):
        from balrog.prompt_builder import create_prompt_builder

        client_cfg = self.bcfg.client
        agent = PAEAgent(prime_client_factory(client_cfg), create_prompt_builder(self.bcfg.agent))
        return agent

    # -- checkpoints ------------------------------------------------------
    def _save_checkpoint(self, attempt, step, agent, prev_action, obs, reason, parent):
        """Append a checkpoint.

        ``parent`` is the checkpoint immediately upstream of this one on the
        same trajectory: the previous checkpoint of this attempt, or - for the
        first checkpoint of an attempt - the checkpoint the attempt resumed
        from. It is None only for the root (attempt 1, step 0). The archive is
        therefore a tree reconstructible from ``archive/*/meta.json`` alone.
        """
        cid = f"c{len(self.archive) + 1}"
        prompt_state = dump_prompt_state(agent.prompt_builder)
        ck = {
            "id": cid,
            "parent": parent,
            "attempt": attempt,
            "step": step,
            "reason": reason,
            "progression": self.adapter.progression(),
            "aux_progress": self.adapter.aux_progress(),
            "summary": self.adapter.summary(),
            "prev_action": prev_action,
            "obs_text": dict(obs["text"]),
            "obs_digest": obs_digest(obs),
            "state_digest": self.adapter.state_digest(),
            "env": self.adapter.env_snapshot(),
            "prompt_state": prompt_state,
            "next_prompt": simulate_next_prompt(prompt_state, prev_action, obs),
        }
        self.blobs[cid] = ck
        self.archive.append({k: ck[k] for k in
                             ("id", "parent", "attempt", "step", "reason", "progression", "aux_progress", "summary")})
        d = self.dir / "archive" / cid
        d.mkdir(parents=True, exist_ok=True)
        with (d / "checkpoint.pkl").open("wb") as f:
            pickle.dump(ck, f)
        (d / "meta.json").write_text(json.dumps(
            {k: ck[k] for k in ("id", "parent", "attempt", "step", "reason", "progression", "aux_progress",
                                "summary", "prev_action", "obs_digest", "state_digest")},
            indent=2, default=str))
        return cid

    # -- one attempt ------------------------------------------------------
    def play(self, attempt: int, ckpt_id: str | None, directive: str | None):
        cfg = self.cfg
        agent = self._new_agent()
        adir = self.dir / "attempts" / f"a{attempt:03d}"
        adir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        fidelity = None
        info: dict = {}
        new_ckpts: list[str] = []

        if ckpt_id is None:
            obs, info = self.adapter.reset(cfg.seed)
            agent.prompt_builder.update_instruction_prompt(self.adapter.instruction_prompt(obs))
            prev_action, step0 = None, 0
            new_ckpts.append(self._save_checkpoint(attempt, 0, agent, None, obs, "episode start", None))
        else:
            ck = self.blobs[ckpt_id]
            robs, _, digests = self.adapter.env_restore(ck["env"])
            load_prompt_state(agent.prompt_builder, ck["prompt_state"])
            obs = {"text": dict(ck["obs_text"]), "image": None}
            if robs is None and ck.get("state_digest") is not None:
                got = self.adapter.state_digest()
                fidelity = {
                    "attempt": attempt, "checkpoint": ckpt_id, "method": ck["env"]["kind"],
                    "replayed_digest": got, "saved_digest": ck["state_digest"],
                    "identical": got == ck["state_digest"], "text_identical": True,
                    "n_replayed_steps": 0,
                }
                _jsonl(self.dir / "restore_fidelity.jsonl", fidelity)
                if cfg.verify_restore and not fidelity["identical"]:
                    raise RuntimeError(f"state digest mismatch on restore of {ckpt_id}: {fidelity}")
            if robs is not None:
                obs = robs
                fidelity = {
                    "attempt": attempt, "checkpoint": ckpt_id, "method": ck["env"]["kind"],
                    "replayed_digest": digests[-1], "saved_digest": ck["obs_digest"],
                    "identical": digests[-1] == ck["obs_digest"],
                    "text_identical": dict(robs["text"]) == dict(ck["obs_text"]),
                    "n_replayed_steps": len(digests) - 1,
                }
                _jsonl(self.dir / "restore_fidelity.jsonl", fidelity)
                if cfg.verify_restore and not fidelity["text_identical"]:
                    raise RuntimeError(f"restore parity failed for {ckpt_id}: {fidelity}")
            prev_action, step0 = ck["prev_action"], ck["step"]
            if directive:
                agent.pending_directive = directive

        max_steps = cfg.max_steps or self.adapter.max_steps
        outcome, step = "step_cap", step0
        last_prog, last_aux = self.adapter.progression(), self.adapter.aux_progress()
        in_tok = out_tok = calls = 0
        first_call_checked = ckpt_id is not None

        for step in range(step0, max_steps):
            response = agent.act(obs, prev_action=prev_action)
            calls += 1
            in_tok += response.input_tokens
            out_tok += response.output_tokens
            self.acct.add("player", response.input_tokens, response.output_tokens)

            if first_call_checked:
                base = messages_to_json(agent.last_base_messages)
                sent = messages_to_json(agent.last_messages)
                expected = self.blobs[ckpt_id]["next_prompt"]
                check = {
                    "attempt": attempt, "checkpoint": ckpt_id,
                    "history_matches_checkpoint": base == expected,
                    "n_messages_base": len(base), "n_messages_sent": len(sent),
                    "extra_messages": [m for m in sent if m not in base],
                }
                _jsonl(self.dir / "resume_prompt_check.jsonl", check)
                if cfg.verify_restore and not check["history_matches_checkpoint"]:
                    raise RuntimeError(f"resumed history != checkpoint history for {ckpt_id}")
                first_call_checked = False

            action = self.adapter.env.check_action_validity(response.completion)
            if hasattr(self.adapter, "record"):
                self.adapter.record(action, response.completion)
            obs, reward, terminated, truncated, info = self.adapter.step(action)
            done = terminated or truncated
            if (action != response.completion) and self.bcfg.eval.feedback_on_invalid_action:
                obs["text"]["long_term_context"] = (
                    f"\n\nYour previous output did not contain a valid action. Defaulted to action: {action}"
                    f"\n\nObservation:\n" + obs["text"]["long_term_context"]
                )
            prev_action = response.completion

            prog, aux = self.adapter.progression(), self.adapter.aux_progress()
            _jsonl(adir / "trace.jsonl", {
                "step": step, "completion": response.completion, "action": action,
                "valid": action == response.completion, "reward": float(reward), "done": bool(done),
                "progression": prog, "aux": aux,
                "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                "obs": obs["text"]["long_term_context"][:4000],
            })

            if not done:
                advanced = prog > last_prog or aux > last_aux
                if advanced or ((step + 1) % self.cfg.checkpoint_every == 0):
                    new_ckpts.append(self._save_checkpoint(
                        attempt, step + 1, agent, prev_action, obs,
                        "progress" if advanced else "periodic",
                        new_ckpts[-1] if new_ckpts else ckpt_id))
            last_prog, last_aux = prog, aux

            if done:
                status = str(info.get("end_status", "")) if isinstance(info, dict) else ""
                if prog >= 1.0 or status in ("2", "StepStatus.TASK_SUCCESSFUL"):
                    outcome = "solved"
                elif status in ("1", "StepStatus.DEATH"):
                    outcome = "died"
                else:
                    outcome = "ended"
                break
        else:
            step = max_steps - 1

        row = {
            "attempt": attempt,
            "from_checkpoint": ckpt_id,
            "start_step": step0,
            "end_step": step + 1,
            "steps_played": step + 1 - step0,
            "committed_steps": step + 1,
            "total_env_steps": self.adapter.total_env_steps,
            "directive": directive or "",
            "directive_kind": ("none" if not directive else ("neutral" if directive == NEUTRAL_DIRECTIVE else "orchestrator")),
            "selection_source": "fresh" if ckpt_id is None else self._sel_source,
            "progression": self.adapter.progression(),
            "aux_progress": self.adapter.aux_progress(),
            "outcome": outcome,
            "end_status": str(info.get("end_status", "")) if isinstance(info, dict) else "",
            "calls": calls,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": round(in_tok / 1e6 * 1.54 + out_tok / 1e6 * 4.84, 6),
            "wall_s": round(time.time() - t_start, 1),
            "new_checkpoints": new_ckpts,
            "restore_fidelity": fidelity,
        }
        self.attempts.append(row)
        _jsonl(self.dir / "attempts.jsonl", row)
        return row

    # -- selection --------------------------------------------------------
    def _select(self):
        cfg = self.cfg
        prev = self.attempts[-1]
        want_directive = cfg.directive == "on" and not cfg.blind
        fixed_pool = self.resumable(prev["new_checkpoints"]) or self.resumable()
        fixed_pick = fixed_pool[-1]["id"] if fixed_pool else None
        if fixed_pick is None:
            raise RuntimeError("no resumable checkpoint left (every checkpoint sits at the step cap)")
        if cfg.select == "fixed" or self.orch is None:
            self._sel_source = "fixed_rule"
            cid = fixed_pick
            directive = None
            if want_directive:
                d = self.orch.decide(self.resumable(), self.attempts, True, fixed_checkpoint=cid) if self.orch else {}
                directive = d.get("directive") or None
        else:
            d = self.orch.decide(self.resumable(), self.attempts, want_directive)
            cid = d["checkpoint"]
            self._sel_source = "llm"
            if cid is None:
                cid = fixed_pick
                self._sel_source = "fallback_fixed_rule"
            directive = (d.get("directive") or None) if want_directive else None
        if cfg.blind:
            directive = NEUTRAL_DIRECTIVE
        if cfg.directive == "off" and not cfg.blind:
            directive = None
        _jsonl(self.dir / "selection.jsonl", {
            "attempt": len(self.attempts) + 1, "checkpoint": cid,
            "source": self._sel_source, "directive": directive or ""})
        return cid, directive

    # -- driver -----------------------------------------------------------
    def go(self):
        cfg = self.cfg
        (self.dir / "config.json").write_text(json.dumps(
            {**cfg.__dict__, "balrog_agent": OmegaConf.to_container(self.bcfg.agent, resolve=True),
             "balrog_client": OmegaConf.to_container(self.bcfg.client, resolve=True)}, indent=2, default=str))
        self._sel_source = "fresh"
        stale = 0
        n = 1 if cfg.base_only else cfg.attempts
        stop = "attempts_exhausted"
        for attempt in range(1, n + 1):
            ckpt, directive = (None, None) if attempt == 1 else self._select()
            row = self.play(attempt, ckpt, directive)
            key = (row["progression"], row["aux_progress"])
            if key > (self.best["progression"], self.best["aux"]):
                self.best = {"progression": key[0], "aux": key[1], "attempt": attempt}
                stale = 0
            else:
                stale += 1
            if row["progression"] >= 1.0:
                stop = "solved"
                break
            if stale >= cfg.plateau:
                stop = f"plateau_{cfg.plateau}"
                break
        self.finish(stop)

    def finish(self, stop_reason):
        if self.orch is not None:
            for r in self.orch.rounds:
                _jsonl(self.dir / "orchestrator_rounds.jsonl", r)
        summary = {
            "game": self.cfg.game, "task": self.cfg.task, "seed": self.cfg.seed,
            "arm": "base" if self.cfg.base_only else
                   f"pae(select={self.cfg.select},directive={'neutral' if self.cfg.blind else self.cfg.directive})",
            "attempts": len(self.attempts),
            "checkpoints": len(self.archive),
            "max_progression": max((a["progression"] for a in self.attempts), default=0.0),
            "final_progression": self.attempts[-1]["progression"] if self.attempts else 0.0,
            "max_aux_progress": max((a["aux_progress"] for a in self.attempts), default=0.0),
            "committed_steps_max": max((a["committed_steps"] for a in self.attempts), default=0),
            "total_env_steps": self.adapter.total_env_steps,
            "llm_steps": sum(a["calls"] for a in self.attempts),
            "env_patches": list(self.adapter.env_patches),
            "stop_reason": stop_reason,
            "best": self.best,
            "tokens": self.acct.snapshot(),
            "wall_s": round(time.time() - self.t0, 1),
            "run_dir": str(self.dir),
        }
        st = summary["tokens"]["total"]
        summary["tokens_per_llm_step"] = {
            "input": round(st["input_tokens"] / max(1, summary["llm_steps"])),
            "output": round(st["output_tokens"] / max(1, summary["llm_steps"])),
        }
        summary["cost_per_llm_step_usd"] = round(st["cost_usd"] / max(1, summary["llm_steps"]), 6)
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        self.adapter.close()
        return summary
