"""Shared helpers for the BALROG checkpoint verification scripts.

All scripts run inside the BALROG venv (see REPORT.md for the exact commands).
"""
from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import random
import sys
import time
import warnings
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")


def load_balrog_config():
    """Load BALROG's default hydra config (the one the evaluator uses)."""
    from omegaconf import OmegaConf

    cfg_path = importlib.resources.files("balrog") / "config" / "config.yaml"
    return OmegaConf.load(str(cfg_path))


def fingerprint(x: Any) -> str:
    """Stable content hash of an observation-ish object (nested dicts/arrays/str/PIL)."""
    h = hashlib.sha256()

    def feed(v):
        if v is None:
            h.update(b"None")
        elif isinstance(v, (str, bytes)):
            h.update(v if isinstance(v, bytes) else v.encode())
        elif isinstance(v, (bool, int, float, np.integer, np.floating)):
            h.update(repr(v).encode())
        elif isinstance(v, np.ndarray):
            h.update(str(v.dtype).encode() + str(v.shape).encode() + np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode())
                feed(v[k])
        elif isinstance(v, (list, tuple, set, frozenset)):
            items = sorted(v, key=repr) if isinstance(v, (set, frozenset)) else v
            for item in items:
                feed(item)
        elif hasattr(v, "tobytes") and hasattr(v, "size"):  # PIL Image
            h.update(v.tobytes())
        else:
            h.update(repr(v).encode())

    feed(x)
    return h.hexdigest()[:16]


class Timer:
    def __init__(self):
        self.samples: list[float] = []

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.samples.append((time.perf_counter() - self._t) * 1e3)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")

    @property
    def max_ms(self) -> float:
        return float(np.max(self.samples)) if self.samples else float("nan")


def fixed_actions(actions: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [rng.choice(actions) for _ in range(n)]


def compare_traces(a: list, b: list, label: str) -> bool:
    """Compare two per-step trace lists of dicts; print first divergence."""
    if len(a) != len(b):
        print(f"[{label}] LENGTH MISMATCH {len(a)} vs {len(b)}")
        return False
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"[{label}] DIVERGENCE at step {i}:")
            for k in x:
                if x.get(k) != y.get(k):
                    print(f"    key={k!r}: {str(x.get(k))[:120]!r} != {str(y.get(k))[:120]!r}")
            return False
    print(f"[{label}] identical over {len(a)} steps")
    return True


def write_result(path: str, result: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"wrote {path}")


def banner(msg: str) -> None:
    print("=" * 78)
    print(msg)
    print("=" * 78)
    sys.stdout.flush()
