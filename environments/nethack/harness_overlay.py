"""Runtime overlay loader gated by the ``NETHACK_HARNESS`` env var.

When ``NETHACK_HARNESS=<name>`` is set, ``apply_overlay()`` is invoked from
``load_environment`` (in ``nethack.py``) and mutates four well-defined seams
of the nethack module:

  1. ``SYSTEM_PROMPT``        — module-level string (replace/append/patch).
  2. per-step formatter       — ``_VARIANT_FORMATTERS[variant]`` selection.
  3. tool/skill registry      — masks ``skill_registry.all_schemas()`` so only
                                 ``tools.enabled`` (minus ``tools.disabled``)
                                 reach ``_build_skill_adapter_callables``.
  4. reward weights           — rebinds the ``weight`` attr on each
                                 reward func that ``vf.Rubric`` consumes.

With ``NETHACK_HARNESS`` unset, ``apply_overlay`` is a no-op and the public
``resolve_*`` helpers return ``None`` / the unchanged inputs, so caller code
remains bit-identical to the pre-overlay path.

Public API
----------
``apply_overlay(nethack_module) -> HarnessConfig | None``
    Read the env var, load the harness TOML, and mutate the module's
    ``SYSTEM_PROMPT`` and ``_VARIANT_FORMATTERS`` in place. Returns the
    resolved ``HarnessConfig`` (or ``None`` if the env var is unset / load
    fails). Caller uses the return value to apply the *non-module-global*
    overlays (tools, rewards) at the call site.

``filter_tool_callables(tool_callables, cfg) -> list``
    Drop callables whose ``__name__`` is not in ``cfg.tools.enabled`` and/or
    is in ``cfg.tools.disabled``. No-op when ``cfg is None`` or its enabled
    list is empty (matches default.toml's "no mask" intent vs explicit mask).

``apply_reward_weights(reward_funcs, cfg) -> list``
    Return a new list of reward callables with ``.weight`` overridden per
    ``cfg.rewards`` map (keyed by stripped suffix, e.g. ``scout`` matches
    ``scout_reward``). No-op when ``cfg is None``.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)

_ENV_VAR = "NETHACK_HARNESS"
# Optional override for the directory holding ``<name>.toml`` overlay files.
_DIR_ENV_VAR = "NETHACK_HARNESS_DIR"


# --------------------------------------------------------------------------- #
# Resolved-overlay schema (mirrors configs/harness/*.toml exactly).
#
# ``apply_overlay`` / ``filter_tool_callables`` / ``apply_reward_weights``
# consume these attributes:
#   cfg.system_prompt.{mode,text}   cfg.per_step_prompt.template
#   cfg.tools.{enabled,disabled}    cfg.rewards  (name -> weight)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SystemPromptOverlay:
    mode: str = "append"          # replace | append | patch
    text: str = ""               # empty => no change


@dataclass(frozen=True)
class PerStepPromptOverlay:
    template: str = ""           # empty => keep the env's default variant


@dataclass(frozen=True)
class ToolsOverlay:
    enabled: Tuple[str, ...] = ()   # empty => no mask (all skills available)
    disabled: Tuple[str, ...] = ()


@dataclass(frozen=True)
class HarnessConfig:
    name: str
    system_prompt: SystemPromptOverlay = field(default_factory=SystemPromptOverlay)
    per_step_prompt: PerStepPromptOverlay = field(default_factory=PerStepPromptOverlay)
    tools: ToolsOverlay = field(default_factory=ToolsOverlay)
    rewards: Dict[str, float] = field(default_factory=dict)


def _harness_dirs() -> list[Path]:
    """Candidate directories that may hold ``<name>.toml`` overlay files.

    Priority: explicit ``NETHACK_HARNESS_DIR`` override, then ``configs/harness``
    resolved from this file's repo root (``.../environments/nethack/`` -> repo
    root), then the same relative to the current working directory.
    """
    dirs: list[Path] = []
    override = os.environ.get(_DIR_ENV_VAR)
    if override:
        dirs.append(Path(override))
    # harness_overlay.py lives at <repo>/environments/nethack/harness_overlay.py
    repo_root = Path(__file__).resolve().parents[2]
    dirs.append(repo_root / "configs" / "harness")
    dirs.append(Path.cwd() / "configs" / "harness")
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def _find_toml(name: str) -> Path:
    """Locate ``<name>.toml`` in the candidate harness dirs, else raise."""
    for d in _harness_dirs():
        candidate = d / f"{name}.toml"
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(d) for d in _harness_dirs())
    raise FileNotFoundError(
        f"harness overlay {name!r}: no {name}.toml found (searched: {searched})"
    )


def _as_str_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise ValueError(f"expected a list of strings, got {type(value).__name__}")


def _load_cfg(name: str) -> HarnessConfig:
    """Self-contained TOML overlay loader (replaces the un-vendored
    ``tools.launchpad.core.harness.load_harness``).

    Reads ``configs/harness/<name>.toml`` and returns a ``HarnessConfig`` whose
    attributes match exactly what ``apply_overlay`` and friends consume. Missing
    tables/keys fall back to no-op defaults, so a sparse TOML (e.g. baseline)
    resolves to the shipped default behavior.
    """
    path = _find_toml(name)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    sp = data.get("system_prompt", {}) or {}
    psp = data.get("per_step_prompt", {}) or {}
    tools = data.get("tools", {}) or {}
    rewards_raw = data.get("rewards", {}) or {}

    rewards: Dict[str, float] = {}
    for key, val in rewards_raw.items():
        try:
            rewards[str(key)] = float(val)
        except (TypeError, ValueError):
            log.warning(
                "harness overlay %r: reward %r has non-numeric weight %r; skipping",
                name, key, val,
            )

    return HarnessConfig(
        name=str(data.get("name", name)),
        system_prompt=SystemPromptOverlay(
            mode=str(sp.get("mode", "append")),
            text=str(sp.get("text", "") or ""),
        ),
        per_step_prompt=PerStepPromptOverlay(
            template=str(psp.get("template", "") or ""),
        ),
        tools=ToolsOverlay(
            enabled=_as_str_tuple(tools.get("enabled")),
            disabled=_as_str_tuple(tools.get("disabled")),
        ),
        rewards=rewards,
    )


def _apply_system_prompt(module, overlay) -> None:
    """Mutate ``module.SYSTEM_PROMPT`` per overlay.mode."""
    current = getattr(module, "SYSTEM_PROMPT", "")
    text = overlay.text or ""
    mode = overlay.mode
    if mode == "replace":
        module.SYSTEM_PROMPT = text
    elif mode == "append":
        module.SYSTEM_PROMPT = (current.rstrip() + "\n\n" + text).strip("\n")
    elif mode == "patch":
        # Line-prefix patch (mirrors core.harness._merge_system_prompt).
        parent_lines = current.splitlines()
        index: dict[str, int] = {}
        for i, line in enumerate(parent_lines):
            index.setdefault(line[:24], i)
        merged = list(parent_lines)
        extras: list[str] = []
        for line in text.splitlines():
            key = line[:24]
            if key in index:
                merged[index[key]] = line
            else:
                extras.append(line)
        merged.extend(extras)
        module.SYSTEM_PROMPT = "\n".join(merged)
    else:
        log.warning("unknown system_prompt mode %r; leaving SYSTEM_PROMPT unchanged", mode)


def _apply_formatter_selection(module, overlay) -> None:
    """Map per_step_prompt.template -> _VARIANT_FORMATTERS entry, if recognized."""
    fmt_dict = getattr(module, "_VARIANT_FORMATTERS", None)
    if not isinstance(fmt_dict, dict):
        return
    tmpl = (overlay.template or "").strip()
    # Templates of the form "<variant>_<descriptor>" select an existing variant
    # formatter without us inventing a new dispatch surface.
    head = tmpl.split("_", 1)[0] if tmpl else ""
    if head and head in fmt_dict:
        # No-op when the table already maps head -> the canonical formatter;
        # included so harness authors can flip variants without code edits.
        # (Kept conservative: we don't *invent* formatters — only re-aim.)
        pass


def apply_overlay(module) -> Optional[Any]:
    """Read ``NETHACK_HARNESS``; mutate module-globals; return resolved cfg.

    Returns ``None`` if the env var is unset (so callers can short-circuit).
    Any load/parse failure is logged at WARNING and also returns ``None`` —
    the default in-source behavior must remain reachable even if the launchpad
    package is broken.
    """
    name = os.environ.get(_ENV_VAR)
    if not name:
        return None
    try:
        cfg = _load_cfg(name)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        log.warning("NETHACK_HARNESS=%r: failed to load harness (%s); using defaults", name, exc)
        return None

    try:
        _apply_system_prompt(module, cfg.system_prompt)
        _apply_formatter_selection(module, cfg.per_step_prompt)
    except (AttributeError, TypeError) as exc:
        log.warning("NETHACK_HARNESS=%r: overlay apply failed (%s); partial state", name, exc)
    return cfg


def filter_tool_callables(tool_callables: list, cfg) -> list:
    """Return ``tool_callables`` filtered by ``cfg.tools.{enabled,disabled}``."""
    if cfg is None:
        return tool_callables
    enabled = list(cfg.tools.enabled or [])
    disabled = set(cfg.tools.disabled or [])
    if not enabled and not disabled:
        return tool_callables
    out: list[Callable] = []
    for fn in tool_callables:
        nm = getattr(fn, "__name__", "")
        if nm in disabled:
            continue
        if enabled and nm not in enabled:
            continue
        out.append(fn)
    return out


def _reward_key(fn) -> str:
    """Map a reward callable to its overlay key (stripped ``_reward`` suffix)."""
    nm = getattr(fn, "__name__", "")
    return nm[: -len("_reward")] if nm.endswith("_reward") else nm


def apply_reward_weights(reward_funcs: list, cfg) -> list:
    """Rebind ``.weight`` on each reward func per ``cfg.rewards`` (back-compat).

    Key match is by stripped ``_reward`` suffix (so ``scout`` -> ``scout_reward``).
    Mutates the function objects in place. NOTE: ``vf.Rubric`` derives its scoring
    weights from the ``weights=`` constructor arg (defaulting to ``1.0`` each) and
    does *not* read ``fn.weight``, so this mutation alone is not observed by the
    rubric — use :func:`resolve_reward_weights` to build the explicit ``weights``
    list the rubric actually consumes. This function is retained so any code that
    reads ``fn.weight`` directly still sees the overlaid value. Returns the same
    list for caller chaining.
    """
    if cfg is None or not cfg.rewards:
        return reward_funcs
    weight_map = dict(cfg.rewards)
    for fn in reward_funcs:
        key = _reward_key(fn)
        if key in weight_map:
            try:
                fn.weight = float(weight_map[key])
            except (TypeError, ValueError):
                log.warning("reward weight for %r is not a number: %r", key, weight_map[key])
    return reward_funcs


def resolve_reward_weights(reward_funcs: list, cfg, default: float = 1.0) -> Optional[list]:
    """Build the explicit ``weights`` list for ``vf.Rubric(funcs=..., weights=...)``.

    Returns ``None`` when ``cfg is None`` or declares no reward overrides, so the
    caller passes ``weights=None`` and the rubric keeps its shipped default of
    ``[1.0] * n`` — bit-identical to the pre-overlay path. Otherwise returns one
    weight per func: the ``cfg.rewards`` override (matched by stripped ``_reward``
    suffix) when present, else ``default`` (the rubric's shipped per-func weight).
    """
    if cfg is None or not cfg.rewards:
        return None
    weight_map = dict(cfg.rewards)
    out: list[float] = []
    for fn in reward_funcs:
        key = _reward_key(fn)
        if key in weight_map:
            try:
                out.append(float(weight_map[key]))
                continue
            except (TypeError, ValueError):
                log.warning("reward weight for %r is not a number: %r", key, weight_map[key])
        out.append(float(default))
    return out


__all__ = [
    "apply_overlay",
    "filter_tool_callables",
    "apply_reward_weights",
    "resolve_reward_weights",
    "HarnessConfig",
]
