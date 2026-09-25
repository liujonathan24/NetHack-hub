"""GLM-5.2 over Prime Inference for BALROG's naive agent.

BALROG's ``OpenAIWrapper`` builds its ``OpenAI`` client itself and only knows
about the openai / vllm / nvidia / xai flavours, none of which can send the
``X-Prime-Team-ID`` header that Prime Inference requires (without it
``z-ai/glm-5.2`` answers HTTP 402 ``insufficient_funds``).  The subclass below
overrides *only* ``_initialize_client``; ``convert_messages`` and ``generate``
- i.e. everything that shapes the prompt and parses the answer - are BALROG's,
unchanged, so the rows stay leaderboard-comparable.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from openai import OpenAI

from balrog.client import LLMClientWrapper, OpenAIWrapper  # noqa: F401

BASE_URL = "https://api.pinference.ai/api/v1"
MODEL_ID = "z-ai/glm-5.2"
# results/model_prices.json (zombie-fix), USD per 1M tokens
PRICES = {"z-ai/glm-5.2": {"input": 1.54, "output": 4.84}}


def prime_credentials() -> tuple[str, str]:
    """(api_key, team_id) from the environment, else ~/.prime/config.json."""
    key = os.environ.get("PRIME_API_KEY")
    team = os.environ.get("PRIME_TEAM_ID")
    if not (key and team):
        cfg = json.loads(Path("~/.prime/config.json").expanduser().read_text())
        key = key or cfg["api_key"]
        team = team or cfg["team_id"]
    return key, team


# Prime Inference returns the true billed cost on every completion; we sum it so
# the run does not have to trust a price table (see MEMORY: ledger spend_usd can
# overstate wallet cost).
BILLED = {"usd": 0.0, "calls": 0}


def _instrument(client: OpenAI) -> OpenAI:
    """Record the provider-reported cost of every completion. Prompts untouched."""
    original = client.chat.completions.create

    def create(*args, **kwargs):
        resp = original(*args, **kwargs)
        usage = getattr(resp, "usage", None)
        cost = getattr(usage, "cost", None) if usage is not None else None
        if cost is None:
            cost = getattr(resp, "cost", None)
        BILLED["usd"] += float(cost or 0.0)
        BILLED["calls"] += 1
        return resp

    client.chat.completions.create = create  # type: ignore[method-assign]
    return client


def make_openai_client(timeout: float = 600.0) -> OpenAI:
    key, team = prime_credentials()
    return _instrument(
        OpenAI(
            api_key=key,
            base_url=BASE_URL,
            timeout=timeout,
            default_headers={"X-Prime-Team-ID": team},
        )
    )


class PrimeOpenAIWrapper(OpenAIWrapper):
    """BALROG's OpenAI wrapper pointed at Prime Inference."""

    def _initialize_client(self):
        if not self._initialized:
            self.client = make_openai_client(timeout=self.timeout)
            self._initialized = True


def prime_client_factory(client_config):
    def factory():
        return PrimeOpenAIWrapper(client_config)

    return factory


class Accountant:
    """Token + cost ledger, split by role (player / orchestrator)."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._lock = threading.Lock()
        self.rows: dict[str, dict] = {}

    def add(self, role: str, input_tokens: int, output_tokens: int, calls: int = 1) -> None:
        with self._lock:
            r = self.rows.setdefault(role, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            r["calls"] += calls
            r["input_tokens"] += int(input_tokens)
            r["output_tokens"] += int(output_tokens)

    def cost(self, role: str | None = None) -> float:
        p = PRICES.get(self.model_id, {"input": 0.0, "output": 0.0})
        rows = self.rows.values() if role is None else [self.rows.get(role, {})]
        total = 0.0
        for r in rows:
            total += r.get("input_tokens", 0) / 1e6 * p["input"]
            total += r.get("output_tokens", 0) / 1e6 * p["output"]
        return total

    def snapshot(self) -> dict:
        out = {k: dict(v, cost_usd=round(self.cost(k), 6)) for k, v in self.rows.items()}
        out["total"] = {
            "calls": sum(v["calls"] for v in self.rows.values()),
            "input_tokens": sum(v["input_tokens"] for v in self.rows.values()),
            "output_tokens": sum(v["output_tokens"] for v in self.rows.values()),
            "cost_usd": round(self.cost(), 6),
        }
        out["billed_by_provider_usd"] = round(BILLED["usd"], 6)
        out["billed_calls"] = BILLED["calls"]
        return out


def provenance() -> dict:
    """git commits of this tool and of the BALROG checkout it drives."""
    import subprocess
    from pathlib import Path

    here = Path(__file__).resolve().parent

    def git(repo: Path, *args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                  text=True, timeout=20, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    repo = here.parent.parent
    try:
        import balrog

        balrog_dir = Path(balrog.__file__).resolve().parent.parent
    except Exception:  # noqa: BLE001
        balrog_dir = None
    return {
        "tool_commit": git(repo, "rev-parse", "HEAD"),
        "tool_branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "tool_dirty": bool(git(repo, "status", "--porcelain", "--", str(here))),
        "tool_dir": str(here),
        "balrog_commit": git(balrog_dir, "rev-parse", "HEAD") if balrog_dir else None,
        "balrog_dir": str(balrog_dir) if balrog_dir else None,
    }
