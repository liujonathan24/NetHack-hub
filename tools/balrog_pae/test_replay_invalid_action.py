#!/usr/bin/env python
"""Replay parity across an INVALID action. No LLM calls, no money.

BALROG's evaluator rewrites the observation text whenever the model's raw
completion is not a legal action ("Your previous output did not contain a valid
action. Defaulted to action: ..."), and the player's history therefore contains
that rewritten text. A replay-based restore that steps the env with the
*validated* action but forgets the rewrite would silently hand the resumed
player a different prompt from the one the checkpoint recorded.

This exercises exactly that path: a prefix containing a deliberately invalid
completion, then env_snapshot / env_restore, asserting
  1. every replayed observation digest equals the original, step for step,
  2. the restored observation text is character-identical, and
  3. the invalid-action feedback banner is present in both.

  python -m tools.balrog_pae.test_replay_invalid_action
"""
from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

FEEDBACK = "Your previous output did not contain a valid action."


def apply_feedback(obs, action, completion, enabled):
    if enabled and action != completion:
        obs["text"]["long_term_context"] = (
            f"\n\nYour previous output did not contain a valid action. Defaulted to action: {action}"
            f"\n\nObservation:\n" + obs["text"]["long_term_context"]
        )
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="MiniHack-Quest-Easy-v0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from .adapters import make_adapter, obs_digest
    from .run import balrog_config

    cfg = balrog_config("test", 1.0, 64, 60)
    feedback_on = cfg.eval.feedback_on_invalid_action
    ad = make_adapter("minihack", a.task, cfg)
    ad.make()
    obs, _ = ad.reset(a.seed)

    # a prefix with three invalid completions in different shapes
    script = ["south", "I will now move north please", "east", "north", "<<<garbage>>>", "search", "42", "west"]
    digests, texts, invalid_steps = [obs_digest(obs)], [dict(obs["text"])], []
    for i, completion in enumerate(script):
        action = ad.env.check_action_validity(completion)
        if action != completion:
            invalid_steps.append(i)
        ad.record(action, completion)
        obs, _r, term, trunc, _info = ad.step(action)
        obs = apply_feedback(obs, action, completion, feedback_on)
        digests.append(obs_digest(obs))
        texts.append(dict(obs["text"]))
        if term or trunc:
            print(f"episode ended at prefix step {i}; truncating the test prefix there")
            break

    assert invalid_steps, "no invalid action was produced - the test would be vacuous"
    assert feedback_on, "BALROG's feedback_on_invalid_action is off; this test needs it on"
    banner_seen = sum(1 for t in texts if FEEDBACK in t["long_term_context"])
    assert banner_seen == len(invalid_steps), f"{banner_seen} banners for {len(invalid_steps)} invalid actions"

    snap = ad.env_snapshot()
    # mutate the live env so a silent no-op "restore" cannot pass
    for _ in range(5):
        ad.step("north")

    fails = []
    for trial in range(3):
        robs, _info, rdigests = ad.env_restore(snap)
        if rdigests != digests:
            first = next(i for i, (x, y) in enumerate(zip(rdigests, digests)) if x != y)
            fails.append(f"trial {trial}: digest divergence at replayed step {first}")
        if dict(robs["text"]) != texts[-1]:
            fails.append(f"trial {trial}: restored text differs from the recorded text")
        if FEEDBACK not in "".join(t["long_term_context"] for t in texts):
            fails.append(f"trial {trial}: feedback banner missing from the recorded prefix")
        for _ in range(3):
            ad.step("south")

    print(f"task={a.task} seed={a.seed} prefix={len(digests) - 1} steps, "
          f"invalid completions at steps {invalid_steps} (defaulted to '{ad.env.env.default_action}')")
    print(f"feedback banners in the recorded prefix: {banner_seen}")
    print(f"replayed digests identical to the original: {len(digests)} per trial x 3 trials")
    ad.close()
    if fails:
        for f in fails:
            print("FAIL:", f)
        return 1
    print("PASS: replay across invalid actions is byte-identical, feedback text included")
    return 0


if __name__ == "__main__":
    sys.exit(main())
