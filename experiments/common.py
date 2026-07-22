"""Shared post-monolith plumbing for the experiments tab.

After the engine/Hub split every experiment runs the SAME way:

  * the NetHack engine is an EXTERNAL package (``nethack_core``) plus a prebuilt
    ``libnethack.so`` located via ``NLE_LIB_PATH`` — there is no
    ``third_party/NetHack`` in this repo anymore;
  * the model is called REMOTELY from Prime Inference. This machine is only the
    ORCHESTRATOR: it runs the env (CPU) and streams tokens from the model.
    Team billing needs the ``X-Prime-Team-ID`` header, which the verifiers Prime
    client emits from ``PRIME_TEAM_ID`` (or ``~/.prime/config.json``'s team_id).

Each experiment module DEFINES its config + matrix and calls into a runner; this
module centralizes the environment wiring so none of them re-implement it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

#: Cheapest served policy — the default for smokes so cost stays ~$0.
CHEAP_MODEL = "Qwen/Qwen3.5-0.8B"
CHEAP_TEACHER = "Qwen/Qwen3.5-2B"          # must differ from the policy
DEFAULT_VLM = "qwen/qwen3-vl-8b-instruct"  # for pixel encodings


def find_libnethack() -> Optional[str]:
    """Locate the prebuilt engine lib. NLE_LIB_PATH wins; else common build dirs."""
    env = os.environ.get("NLE_LIB_PATH")
    if env and Path(env).exists():
        return env
    # nethack_core's own locator will also find a bundled copy or the standard
    # build dir; we only pre-seed NLE_LIB_PATH when we can see a local build.
    for c in (
        Path.home() / "NetHackHarness/third_party/NetHack/src/build/libnethack.so",
        Path("/scratch/gpfs/ZHUANGL/jl0796/NetHackHarness/third_party/NetHack/src/build/libnethack.so"),
    ):
        if c.exists():
            return str(c)
    return None


def prime_team_id() -> Optional[str]:
    tid = os.environ.get("PRIME_TEAM_ID")
    if tid:
        return tid
    cfg = Path.home() / ".prime" / "config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text()).get("team_id")
        except Exception:
            return None
    return None


def post_monolith_env(extra: Optional[dict] = None) -> dict:
    """Return an environment dict wired for a post-monolith experiment run:
    engine lib via NLE_LIB_PATH, team billing via PRIME_TEAM_ID, uv on PATH, and
    UV_NO_SYNC so a shared prebuilt .venv is reused as-is."""
    env = dict(os.environ)
    so = find_libnethack()
    if so:
        env["NLE_LIB_PATH"] = so
    tid = prime_team_id()
    if tid:
        env["PRIME_TEAM_ID"] = tid
    env.setdefault("UV_NO_SYNC", "1")
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in env.get("PATH", ""):
        env["PATH"] = local_bin + os.pathsep + env.get("PATH", "")
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def apply_post_monolith_env() -> None:
    """Mutate os.environ in-place (for in-process experiments like go_explore)."""
    os.environ.update(post_monolith_env())
