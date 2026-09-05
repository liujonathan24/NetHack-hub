"""E9b awareness probe: a per-turn reflection prompt.

E8 found the model's reasoning is accurate but INERT -- it narrates the game
well and then does not change the policy it executes. E9b tests whether simply
*asking the model to reflect* every turn closes that gap, without removing
agency (that is E9a's job) and without any machine analysis of its behavior.

It is exactly what it looks like: three questions appended to every turn's
observation. Prompt-only -- no published tool schema changes, so the action
surface stays byte-identical to the NPCORE_v3 control. Enabled by the `reflect`
env knob. docs/EXPERIMENT_E9.md.
"""
from __future__ import annotations

REFLECT_BLOCK = (
    "[Reflect on the previous steps before acting:\n"
    " 1. What have you been working on in the past steps?\n"
    " 2. What is going wrong, if anything?\n"
    " 3. What is the updated plan?]"
)
