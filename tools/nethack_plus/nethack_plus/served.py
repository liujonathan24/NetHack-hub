"""What a run served the model: the prompt, the tool list and the first observation, per attempt.

`nethack-plus served RUN` prints, for each library of a run, what attempt 0 and the
first resumed attempt were served, and the differences between the libraries, so a
comparison can be checked to differ only in what it is meant to.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from . import report


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def served(trace: dict) -> dict[str, str]:
    """The system and user prompt, the tool names and the first tool result a trace's model saw."""
    out = {"prompt": "", "tools": "", "first_observation": ""}
    nodes = [n.get("message") or {} for n in trace.get("nodes") or ()]
    out["prompt"] = "\n---\n".join(_text(m.get("content")) for m in nodes if m.get("role") in ("system", "user"))[:6000]
    out["tools"] = "\n".join(sorted(t.get("name", "") for t in trace.get("tools") or ()))
    first = next((m for m in nodes if m.get("role") == "tool"), None)
    out["first_observation"] = _text(first.get("content"))[:4000] if first else ""
    return out


def sample(arm: Path) -> dict[int, dict[str, str]]:
    """What attempt 0 and attempt 1 of the arm's first run were served."""
    for episode in report.episodes(arm):
        attempts = {}
        for trace in episode["traces"]:
            attempt = (trace.get("info", {}).get("explore") or {}).get("attempt", 0)
            attempts.setdefault(attempt, served(trace))
        if attempts:
            return {k: v for k, v in attempts.items() if k <= 1}
    return {}


def compare(run: Path) -> str:
    arms = report.arms(run)
    samples = {arm.name: sample(arm) for arm in arms}
    lines = []
    for name, attempts in samples.items():
        for attempt, parts in sorted(attempts.items()):
            lines += [
                f"===== {name} attempt {attempt}: prompt",
                parts["prompt"],
                f"----- tools ({len(parts['tools'].split())})",
            ]
    if len(samples) == 2:
        (a, sa), (b, sb) = samples.items()
        for attempt in sorted(set(sa) & set(sb)):
            for part in ("prompt", "tools", "first_observation"):
                diff = list(
                    difflib.unified_diff(
                        sa[attempt][part].splitlines(),
                        sb[attempt][part].splitlines(),
                        f"{a}/{part}",
                        f"{b}/{part}",
                        lineterm="",
                        n=0,
                    )
                )
                lines.append(
                    f"===== attempt {attempt} {part}: {'identical' if not diff else f'{len(diff)} diff lines'}"
                )
                lines += diff[:80]
    return "\n".join(lines)
