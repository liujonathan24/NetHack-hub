# Fixtures from the E16 GE-wiki pilot

Real bytes from `/root/nld/e16_runs/gewiki_pilot/`, vendored so the regression
tests do not depend on a run directory that may be archived or deleted. Neither
file is edited; both are extracted verbatim from the orchestrator's own
prime-agent session file,
`orchestrator/agent/sessions/01a04847-35f7-70b8-b336-5496c599c569.jsonl`.

## `pilot_round1_print_stdout.txt` (1,105 bytes)

The assistant text of the pilot's round-1 turn (session record 49): a rationale
paragraph, a blank line, and then the decision object the round prompt demanded
"EXACTLY ... on its own line". Plain `prime-agent --print` writes assistant text
and nothing else, so this **is** the stdout the driver read.

The driver recorded 678 of these 1,104 characters and launched attempt 1 with
"(orchestrator produced no directive for this attempt; play as you judge best)".
`test_THE_PILOT_DIRECTIVE_survives_a_text_mode_round` reproduces that exact
678-char truncation through the old code path before asserting the fix.

## `pilot_opening_assistant_text.json` (2 strings)

The two assistant text blocks of the pilot's opening round (session records 42
and 44): the 60-char "Now I have a complete picture..." line and the
11,478-char plan itself. The run recorded this round as 2,656,190 characters —
533 cumulative prefixes of the same plan — because the json-mode parser
concatenated every streaming `message_update`. The model did not loop.

Used by `test_json_mode_stream_is_not_multiplied_by_its_own_deltas` (which
reconstructs the stream framing from `prime-agent/docs/json.md`, but keeps the
pilot's real text) and by
`test_degeneration_detector_flags_the_pilots_recorded_opening_plan` (0.033
unique-line ratio for the recorded artifact, 0.959 for the real plan).
