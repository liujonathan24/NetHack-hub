"""The episode runner: BALROG's naive agent, plus a resumable prompt state.

``PAEAgent.act`` is BALROG's ``NaiveAgent.act`` with exactly one addition - an
optional single extra user message (the orchestrator's directive) inserted
immediately *before* the current-observation message.  With no directive the
two are byte-identical, which is what keeps the base arm on the leaderboard
protocol.
"""
from __future__ import annotations

import copy

from balrog.agents.naive import NaiveAgent
from balrog.prompt_builder.history import Message

NAIVE_INSTRUCTION = (
    "You always have to output one of the above actions at a time and no other text. "
    "You always have to output an action until the episode terminates."
)


class PAEAgent(NaiveAgent):
    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.pending_directive: str | None = None
        self.last_messages: list[Message] | None = None       # what was actually sent
        self.last_base_messages: list[Message] | None = None  # the same minus the directive

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + NAIVE_INSTRUCTION
        self.last_base_messages = list(messages)
        if self.pending_directive:
            messages = list(messages)
            messages.insert(len(messages) - 1, Message(role="user", content=self.pending_directive))
            self.pending_directive = None
        self.last_messages = messages
        response = self.client.generate(messages)
        return self._extract_final_answer(response)


# --- prompt-builder state (this IS the player's conversation history) --------
def dump_prompt_state(pb) -> dict:
    return {
        "system_prompt": pb.system_prompt,
        "events": copy.deepcopy(list(pb._events)),
        "last_short_term_obs": pb._last_short_term_obs,
        "previous_reasoning": pb.previous_reasoning,
        "max_text_history": pb.max_text_history,
        "max_image_history": pb.max_image_history,
        "max_cot_history": pb.max_cot_history,
    }


def load_prompt_state(pb, state: dict) -> None:
    pb.system_prompt = state["system_prompt"]
    pb._events.clear()
    for ev in copy.deepcopy(state["events"]):
        pb._events.append(ev)
    pb._last_short_term_obs = state["last_short_term_obs"]
    pb.previous_reasoning = state["previous_reasoning"]


def messages_to_json(messages) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def simulate_next_prompt(state: dict, prev_action, obs) -> list[dict]:
    """The exact message list the player would send next from ``state``.

    Used to prove that a resumed player's history equals the checkpoint's
    history (plus the directive turn).
    """
    from balrog.prompt_builder.history import HistoryPromptBuilder

    pb = HistoryPromptBuilder(
        max_text_history=state["max_text_history"],
        max_image_history=state["max_image_history"],
        max_cot_history=state["max_cot_history"],
    )
    load_prompt_state(pb, state)
    if prev_action:
        pb.update_action(prev_action)
    pb.update_observation(obs)
    messages = pb.get_prompt()
    if messages and messages[-1].role == "user":
        messages[-1].content += "\n\n" + NAIVE_INSTRUCTION
    return messages_to_json(messages)
