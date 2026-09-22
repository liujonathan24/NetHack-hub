#!/usr/bin/env python
"""CLI for the BALROG+PAE loop.

  python -m tools.balrog_pae.run --game minihack --task MiniHack-Quest-Easy-v0 \
      --seed 0 --attempts 3 --select fixed --directive on --run-dir runs/smoke

  python -m tools.balrog_pae.run ... --base-only     # unchanged BALROG episode
"""
from __future__ import annotations

import argparse
import importlib.resources
import json
import sys
import warnings

warnings.filterwarnings("ignore")


def balrog_config(model_id, temperature, max_tokens, timeout):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(importlib.resources.files("balrog") / "config" / "config.yaml"))
    cfg.client.client_name = "openai"
    cfg.client.model_id = model_id
    cfg.client.base_url = "https://api.pinference.ai/api/v1"
    cfg.client.generate_kwargs.temperature = temperature
    cfg.client.generate_kwargs.max_tokens = max_tokens
    cfg.client.timeout = timeout
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="minihack", choices=["minihack", "crafter", "textworld"])
    p.add_argument("--task", default="MiniHack-Quest-Easy-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attempts", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--plateau", type=int, default=4)
    p.add_argument("--select", default="orchestrator", choices=["fixed", "orchestrator"])
    p.add_argument("--directive", default="on", choices=["on", "off"])
    p.add_argument("--blind", action="store_true", help="resumed player gets only 'Continue playing.'")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--model", default="z-ai/glm-5.2")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=8192)  # BALROG default
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--no-verify-restore", action="store_true")
    a = p.parse_args(argv)

    from .pae import Config, Run

    cfg = Config(
        game=a.game, task=a.task, seed=a.seed, attempts=a.attempts,
        checkpoint_every=a.checkpoint_every, plateau=a.plateau, select=a.select,
        directive=a.directive, blind=a.blind, max_steps=a.max_steps,
        model_id=a.model, temperature=a.temperature, max_tokens=a.max_tokens,
        run_dir=a.run_dir, base_only=a.base_only, verify_restore=not a.no_verify_restore,
    )
    run = Run(cfg, balrog_config(a.model, a.temperature, a.max_tokens, a.timeout))
    run.go()
    print(json.dumps(json.load(open(run.dir / "summary.json")), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
