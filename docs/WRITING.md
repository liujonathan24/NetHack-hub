# Writing standards

The rules every paper draft in this project is checked against. Two sources:
the Zhuang lab writing guide, and the project's own conventions. Mechanical
checks for most of these live in the paper repo's edit-review tooling.

## Focus

- Write for the reader. At most one new term per paper.
- State the contributions explicitly in 2-3 sentences, in both the abstract
  and the introduction. Repeat at other places.
- The majority of experiments must support the core claims, not side analysis.
- Do not overclaim. Lean rigorous; add conditions to conclusions when unsure.
- Core-contribution sections get length proportionate to their importance.

## Structure

- Paragraphs are 5-8 lines (9 max, 3 min), one point each, summarizable in a
  sentence.
- Abstract at most 15-17 lines in single-column format.
- Section titles 1-4 words. No subsubsections. Summarizing sentences go in the
  text, not the title.
- Do not squeeze for the page limit until the last 24 hours.

## Citations

- Every citation is relevant and the *most* relevant. `\citet` for names in
  the sentence, `\citep` for parentheticals. Multi-cites in time order.
- Related work uses `\paragraph{}` topic blocks, 2-3 of them, 4 max.
- Even citation density across sections - not dense-then-none.
- Bib format: 4-letter venue abbreviations (NeurIPS, ICLR, ICML); delete
  pages/volume/number/publisher; `@inproceedings` for conferences, `@article`
  for arXiv; always cite the published version when one exists.
- Soft citation colour (the lab blue, `#0071BC`). Whitespace before cites.
- 60+ references. No duplicate entries. Verify the paper cited is the paper
  meant.

## Language

- Cut every word that can go without losing meaning.
- Cut anything that hints at LLM generation: delve, leverage, robust,
  comprehensive, seamless, and the rest of the list. Also the structural
  tells: contrastive-binary epigrams ("X, not Y" as a closer), "notably",
  "interestingly", punchy fragments, em-dash asides.
- Mix long and short sentences. Hunt "which"; do not overuse "that" or
  "-ing" clause endings.
- Simple present tense within each section.
- No objectively-good words about your own work (novel, critical, great).
- No unnecessary bold or capitalisation. No contractions. At most a couple
  of dashes in the whole paper.
- Active voice. Frame limitations as opportunities.
- No colons or em-dashes in prose. Cut adverbs, intensifiers, judgement
  words, vague quantifiers, and empty cliches.

## Figures and tables

- Text inside figures renders at roughly the caption size (~9pt) after
  placement scaling. Author figures near their printed width.
- Crop whitespace. PDF with selectable text. Sans-serif (Helvetica/Arial).
- Every float referenced from the text. No vertical rules. Captions below.
- Plots go through the shared style module (`figs/style.py` in the paper
  repo) so ticks, fonts, and palette match.

## Numbers and terminology

- Right precision: 81.2%, not 81.23%.
- Numbers in math mode, consistently.
- Expand every abbreviation at first use. Correct `i.e.,` / `e.g.,` forms.
- One consistent name per model, hyphenated at the version: GLM-5.2,
  GPT-5.6 Sol, Gemini-3.7 Flash, Qwen-3.8 Max.
- Renameable names go through LaTeX macros, never literal prose:
  `\balmin`, `\balmax`, `\pae`/`\Pae`/`\PAE`, `\netcore`, `\baseharness`.
- `\balmax` is reported only in contrast with `\balmin`; prefer citing
  dungeon level and experience level directly where the state is the point.

## Submission checklist

- Zero typos, including captions, re-checked after every caption edit.
- No paragraph ending with a 1-3 word line.
- Floats placed near the text that discusses them.
- No appendix section that is a plot or table with no text.
- Exactly the page limit at submission, not a line under.
- Strip author comments and unused files before arXiv; the source is public.
- Appendix opens with `\section*{\Large Appendix}`.
