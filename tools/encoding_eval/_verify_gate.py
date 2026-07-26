"""End-to-end check that env_response rejects a tool NOT in the exposed set.

Builds a real netplay env, sets up one full_nle rollout, then feeds env_response
a synthetic `move(direction=E)` tool call (move is withheld by netplay) and
asserts the response is a not-available rejection with NO env step taken.
Run: PYTHONPATH=.:environments/nethack .venv/bin/python tools/encoding_eval/_verify_gate.py
"""
import asyncio, json
from nethack import load_environment


def _make_toolcall(name, args):
    return {"function": {"name": name, "arguments": json.dumps(args)}}


async def main():
    env = load_environment(
        task_spec="full_nle", variant="B0", skill_set="netplay",
        compact_obs=False, max_turns=10, character="Val-hum-neu-fem",
        explicit_seeds=[0],
    )
    # One task -> a live state via setup_state.
    ex = env.dataset[0]
    # The row carries no `task` column: `task` is a RESERVED rollout-input field
    # in verifiers (>=0.1.14) — `flatten_task_input` replaces the whole input
    # with it — so the per-rollout seed rides in `info`, which `setup_state`
    # reads as its fallback (`task.get("seed", info.get("seed", ...))`).
    state = {"task": {}, "info": ex["info"]}
    state = await env.setup_state(state) if asyncio.iscoroutinefunction(env.setup_state) else env.setup_state(state)
    dlvl0 = state.get("max_dlvl_reached")
    turn0 = state["structured_obs"]

    # 1) A withheld tool (move) must be rejected.
    msg = {"role": "assistant", "tool_calls": [_make_toolcall("move", {"direction": "E"})]}
    resp = await env.env_response([msg], state)
    text = resp[0]["content"] if isinstance(resp[0], dict) else getattr(resp[0], "content", "")
    text = text if isinstance(text, str) else json.dumps(text)
    assert "not available" in text and "move" in text, f"move NOT rejected: {text[:200]!r}"
    print("PASS: withheld `move` rejected ->", [ln for ln in text.splitlines() if "not available" in ln][:1])
    # The docstring's "NO env step taken" claim, actually asserted: a rejected
    # call must not reach the engine, so the observation object and the depth
    # bookkeeping are untouched. `dlvl0`/`turn0` were captured above and had
    # been going unused, which left `move executed = 0` unproven.
    assert state["structured_obs"] is turn0, "rejected `move` STEPPED the engine"
    assert state.get("max_dlvl_reached") == dlvl0, "rejected `move` changed depth bookkeeping"
    print("PASS: move executed = 0 (engine not stepped, depth unchanged)")

    # 2) An exposed tool (search) must NOT be rejected (dispatches normally).
    msg2 = {"role": "assistant", "tool_calls": [_make_toolcall("search", {})]}
    resp2 = await env.env_response([msg2], state)
    text2 = resp2[0]["content"] if isinstance(resp2[0], dict) else getattr(resp2[0], "content", "")
    text2 = text2 if isinstance(text2, str) else json.dumps(text2)
    assert "not available" not in text2, f"exposed `search` wrongly rejected: {text2[:200]!r}"
    print("PASS: exposed `search` dispatched (not rejected)")
    print("GATE OK")


if __name__ == "__main__":
    asyncio.run(main())
