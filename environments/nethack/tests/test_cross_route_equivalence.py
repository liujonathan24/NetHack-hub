"""Cross-route dispatch equivalence: control route vs native 0.2.1 MCP route.

Replaces the design doc's original "dispatch equivalence" criterion, which
compared `self_dispatch=True` against `self_dispatch=False`. That flag no longer
has two live values (see the `nethack_v1` module docstring): under verifiers
0.2.x a v1 toolset is an MCP server, so the harness-driven arm moved to the v0
legacy bridge. The property that still matters — and that this replaces it with —
is unchanged in substance:

    **the same skill sequence, from the same seed, must produce the same engine
    state whether it is dispatched by the control route or the CLI-agent route.**

That is the check that catches arm-0-vs-CLI-arm drift. If it fails, the two arms
are no longer playing the same game and no comparison between them means
anything.

* control route  = v0 `env_response` (what the legacy bridge runs, `legacy.py:517`)
* native route   = `NetHackToolset.tool_functions()[name](**args)` — the exact
  callable `_register` publishes over MCP

Both are expected to funnel into the single `_apply_tool_call` execution path;
this test is what makes that expectation falsifiable rather than assumed.

SCOPE: this covers engine state and the rendered observation, in process. The
**trace-field** half of the criterion (identical `Trace` fields end-to-end)
needs a booted MCP tool server and a launched harness program, so it is
specified in the design doc as a Task 9 acceptance criterion, not implemented
here.
"""

import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nethack_v1 as m  # noqa: E402

# Deterministic, engine-stepping, and available under the netplay skill set.
SEQUENCE = [
    ("search", {"times": 1}),
    ("search", {"times": 2}),
    ("pray", {}),
    ("search", {"times": 1}),
]

SEED = 0
ENV_ARGS = {"skill_set": "netplay"}
CHARACTER = "Val-hum-neu-fem"

def _scalars(state):
    """The values every reward and stop condition ultimately reads.

    Read through the same coercion the v0 rewards apply (`helpers.py:718`:
    `1.0 if state.get("succeeded") else 0.0`), because the v0 state only
    *creates* a flag once it fires — an unfired flag is absent, not `False`.
    Applied identically to both routes, so a real divergence still shows.
    """
    return {
        "scout_reward_total": float(state.get("scout_reward_total", 0.0) or 0.0),
        "descent_count": float(state.get("descent_count", 0.0) or 0.0),
        "max_dlvl_reached": int(state.get("max_dlvl_reached", 1) or 1),
        "succeeded": bool(state.get("succeeded")),
        "ascended": bool(state.get("ascended")),
        "died": bool(state.get("died")),
        "terminated": bool(state.get("terminated")),
    }


def _obs_fingerprint(state):
    """A comparable snapshot of the engine's observable state."""
    obs = state["structured_obs"]
    return {"status": dict(obs.status), "scalars": _scalars(state)}


def _text(content):
    from nethack_harness.prompt.content import content_to_text

    return content_to_text(content) if not isinstance(content, str) else content


def _control_route():
    """The v0 loop the legacy bridge runs: env_response over a tool call."""
    from nethack import load_environment
    import verifiers as vf0

    env = load_environment(
        task_spec="full_nle",
        n_examples=1,
        max_turns=len(SEQUENCE) + 1,
        explicit_seeds=[SEED],
        character=CHARACTER,
        **ENV_ARGS,
    )
    state = vf0.State({"task": {}, "info": dict(env.dataset[0]["info"])})
    asyncio.run(env.setup_state(state))

    observations, fingerprints = [], []
    for name, args in SEQUENCE:
        message = {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": name, "arguments": json.dumps(args)}}
            ],
        }
        response = asyncio.run(env.env_response([message], state))
        content = (
            response[0]["content"]
            if isinstance(response[0], dict)
            else response[0].content
        )
        observations.append(_text(content))
        fingerprints.append(_obs_fingerprint(state))
    env.close() if hasattr(env, "close") else None
    return observations, fingerprints


def _native_route():
    """The MCP route: the callables `NetHackToolset._register` publishes."""
    cfg = m.NetHackTasksetConfig(
        task_spec="full_nle",
        n_examples=1,
        explicit_seeds=[SEED],
        character=CHARACTER,
        max_turns=len(SEQUENCE) + 1,
        env_args=dict(ENV_ARGS),
    )
    task = m.load_taskset(cfg).select(1)[0]
    toolset = task.tool_servers()[0]
    asyncio.run(toolset.setup_task(task.data))

    tools = toolset.tool_functions()
    observations, fingerprints = [], []
    for name, args in SEQUENCE:
        observations.append(_text(asyncio.run(tools[name](**args))))
        fingerprints.append(_obs_fingerprint(toolset.v0_state))
    # The typed state the rewards actually read must agree with the v0 state.
    published = {
        "scout_reward_total": toolset.state.scout_reward_total,
        "descent_count": toolset.state.descent_count,
        "max_dlvl_reached": toolset.state.max_dlvl_reached,
        "succeeded": toolset.state.succeeded,
        "ascended": toolset.state.ascended,
        "died": toolset.state.died,
        "terminated": toolset.state.terminated,
    }
    asyncio.run(toolset.close())
    return observations, fingerprints, published


def test_cross_route_dispatch_equivalence():
    control_obs, control_fp = _control_route()
    native_obs, native_fp, published = _native_route()

    # The sequence actually ran (not silently short-circuited on both sides).
    assert len(control_obs) == len(SEQUENCE)
    assert all("=== STATUS ===" in text for text in control_obs)

    # Identical engine state after every step.
    for i, (a, b) in enumerate(zip(control_fp, native_fp)):
        assert a == b, f"engine state diverged at step {i} ({SEQUENCE[i][0]})"

    # Identical rendered observation from every call.
    for i, (a, b) in enumerate(zip(control_obs, native_obs)):
        assert a == b, f"observation diverged at step {i} ({SEQUENCE[i][0]})"

    # And the typed state the v1 rewards read agrees with the v0 state the
    # control arm's rewards read — otherwise the two arms would be scored off
    # different numbers even with identical play.
    assert published == native_fp[-1]["scalars"]


def test_cross_route_engine_state_is_actually_sensitive():
    # Guard against the equivalence test passing vacuously: a different seed
    # must produce a different fingerprint, so the comparison has teeth.
    global SEED
    control_obs, control_fp = _control_route()
    original = SEED
    try:
        SEED = 12345
        other_obs, other_fp = _control_route()
    finally:
        SEED = original
    assert control_fp != other_fp or control_obs != other_obs
