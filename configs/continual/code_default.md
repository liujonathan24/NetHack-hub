Reflect broadly on the code. Nothing beyond the contract above is prescribed:
decide for yourself which policies are worth changing and how.

This is the control variant for the skills-as-code experiment -- the one every
other code-reflection prompt is compared against. Keep it unopinionated so a
difference in results is attributable to the variant's added guidance rather
than to this file.

Two reminders that are easy to forget under time pressure, not new rules:
- The differential is the point. "explore.py changed and seed 7 got deeper" is
  only evidence if seed 7 is a seed explore.py actually affects, and if the lift
  survives across the seeds that exercise it -- one game is noise.
- Reverting is a real, valued action, not a failure. A change that looked good
  and measured worse should go back; the git lineage exists so that it can.
