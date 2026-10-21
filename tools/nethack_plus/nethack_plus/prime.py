"""Prime account access: the API key, the wallet, and model prices."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PRIME_CONFIG = Path.home() / ".prime" / "config.json"


def api_key() -> str:
    """`$PRIME_API_KEY`, else the key `prime login` saved; the run cannot start without one."""
    key = os.environ.get("PRIME_API_KEY")
    if key:
        return key
    if PRIME_CONFIG.is_file():
        key = json.loads(PRIME_CONFIG.read_text()).get("api_key")
        if key:
            return key
    raise SystemExit("no Prime credentials: run `prime login` or set PRIME_API_KEY")


def _prime_json(*args: str) -> dict | list:
    result = subprocess.run(["prime", *args, "--output", "json"], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def balance() -> float:
    return float(_prime_json("wallet", "--limit", "1")["balance_usd"])


def prices(model: str) -> dict[str, float]:
    """Dollars per million tokens: `input`, `cached` (cache reads) and `output`."""
    listing = _prime_json("inference", "models")
    models = listing if isinstance(listing, list) else listing.get("data", [])
    for entry in models:
        if entry.get("id") == model:
            pricing = entry.get("pricing") or {}
            inp = float(pricing.get("input_usd_per_mtok") or 0.0)
            cached = pricing.get("cache_read_usd_per_mtok")
            # Anthropic models get no cache reads on Prime; elsewhere an unlisted rate is taken
            # as 0.12x input, what the wallet showed for models with no listed rate.
            if model.startswith("anthropic/"):
                cached = inp
            elif cached is None:
                cached = 0.12 * inp
            return {"input": inp, "cached": float(cached), "output": float(pricing.get("output_usd_per_mtok") or 0.0)}
    raise SystemExit(f"model {model!r} is not served by Prime inference (see `prime inference models`)")


def cost(usage: dict, price: dict[str, float]) -> float:
    """One model call's cost from its token counts. `prompt_tokens` excludes the cached input."""
    return (
        (usage.get("prompt_tokens") or 0) * price["input"]
        + (usage.get("cached_input_tokens") or 0) * price["cached"]
        + (usage.get("completion_tokens") or 0) * price["output"]
    ) / 1e6
