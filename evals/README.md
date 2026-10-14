# Evals

Quality evaluation for cohr (PLAN §18). Synthetic "planted-facts" meetings: every decision,
action item (owner + due date) and number is known with the second it's spoken, so the main
metrics are exact checks with no human labels and no LLM judge. A local judge
(Qwen3-4B on MLX) adds secondary metrics that stay out of the release gate.

## Run

```sh
uv sync --all-groups              # eval group: jiwer, pyyaml
uv run cohr eval --set quick      # ~10 min on an M2 Max; models must be downloaded
```

- Items are generated on first use into `evals/data/gen/<item>/` (gitignored): `say` speech,
  numpy noise/music/silence, PIL slides, then an ffmpeg screen recording or a PDF deck, plus
  `truth.json`. They are regenerated when the item spec or the generator code changes.
- Each item keeps one run in `evals/data/runs/<item>/`. Later evals reuse it, so the
  pipeline's skip logic re-runs only the stages a config or prompt change affects. `--fresh`
  starts over.
- `-c key=value` overrides the config, e.g. `-c fuse.window_s=120` (D12) or
  `-c reduce.citation_min_score=0.2` (D19).
- Reports go to `evals/reports/<YYYYmmdd-HHMMSS>-<set>.json` + `.md` (gitignored).
- `--baseline PATH` compares with an earlier report (exit 1 on a regression, §18 thresholds).
  Each set has its own baseline: `evals/baseline.json` (quick), `evals/baseline-<set>.json`
  otherwise; `--update-baseline` rewrites the set's. Only `quick` and `full` exit 1; `long` is
  report-only.
- `-c reduce.max_input_tokens=4000` forces two-pass Reduce and Ask ranking on the 5-minute quick
  items (2 groups each): the fast loop for long-meeting work (6c).

- After every item has run, the judge (`src/cohr/judge.py`) answers its questions; answers are
  cached in `evals/data/judge/<item>.json`, so re-scoring loads no model. `--no-judge` skips it.
- `uv run cohr eval --calibrate-judge` runs only the judge on the 30 hand-labeled pairs in
  `evals/gen/judge_calibration.json` and reports its agreement with the labels (D15). Don't
  tune the judge prompts on these pairs.

Generate one item by hand: `uv run python -m evals.gen clean-en evals/data/gen/clean-en`.

## Sets

| Set | Items |
|---|---|
| `quick` | `clean-en` (screen recording), `noisy-en` (background noise, music and noise gaps), `pdf-slides-en` (audio + PDF deck, text alignment), `slide-conflict-en` (a number misspoken, correct on the slide) |
| `long` | `long-en`: ~2.5 h, audio + PDF deck, 5 regions × 8 topics, 88 facts; Ask asks a 24-question sample (6c). ~1 h with generation |
| `full` | The `quick` and `lang` items, plus FLEURS: 20 test clips per language in Transcribe's 14, scored by WER (CER for ja/ko/zh) on Transcribe alone, report-only. Streamed from the Hub on first use (~15 MB kept per language in `evals/data/fleurs/`). `--item fleurs` runs only FLEURS, `--item fleurs-ja` one language. AMI follows (6e-1) |

## Metrics

Fact recall (right section), citation accuracy (refined jump time within ±15 s of the fact,
or its slide), uncited claims, made-up decisions/action items (a real task with someone
else's owner or due date counts), open-question recall and made-up open questions (planted
open questions vs. the Open questions section), VAD speech F1 and words transcribed in non-speech gaps,
WER, slide titles/cells/numbers, slide alignment (`pdf-slides-en`), output language, and
seconds per audio hour and peak memory per stage. Two-pass runs also count their groups and how
each pass-2 citation resolved (`pass2_marker`, `pass2_claim`, `pass2_group`, `pass2_uncited`). Matching rules: `src/cohr/evals.py`.

Judge (secondary): **support rate**, the share of cited claims whose cited transcript and
slides state them ("yes"; partial, no and unjudged are counted apart), and **judge recall** for
cross-language text facts. Its cost is reported as its own row.

## Data

`quick`, `lang` and `long` use synthetic data only. `full` adds FLEURS (`google/fleurs`,
streamed from the Hub) and 5 AMI meetings (audio from the official mirror, summaries from the
official annotations zip), kept in `evals/data/` (gitignored, never committed). Every report
that includes them ends with this line:

Evaluation data: AMI Meeting Corpus (University of Edinburgh et al., CC BY 4.0); FLEURS (Google, CC BY 4.0). Data was not modified or redistributed.

`full` gates only on its `quick` items' metrics and stage speeds (D31, `gate:` in
`evals/sets/full.yaml`), against `evals/baseline-full.json`; `lang`, FLEURS and AMI are
report-only.
