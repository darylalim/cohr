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
  `evals/baseline.json` is committed from Phase 6 on; `--update-baseline` rewrites it.

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
| `full` | Phase 6: adds cross-language items, `long-en`, FLEURS and AMI |

## Metrics

Fact recall (right section), citation accuracy (refined jump time within ±15 s of the fact,
or its slide), uncited claims, made-up decisions/action items (a real task with someone
else's owner or due date counts), open-question recall and made-up open questions (planted
open questions vs. the Open questions section), VAD speech F1 and words transcribed in non-speech gaps,
WER, slide titles/cells/numbers, slide alignment (`pdf-slides-en`), output language, and
seconds per audio hour and peak memory per stage. Matching rules: `src/cohr/evals.py`.

Judge (secondary): **support rate**, the share of cited claims whose cited transcript and
slides state them ("yes"; partial, no and unjudged are counted apart), and **judge recall** for
cross-language text facts. Its cost is reported as its own row.

## Data

`quick` uses synthetic data only. When `full` adds external data (Phase 6), its reports end
with: "Evaluation data: AMI Meeting Corpus (University of Edinburgh et al., CC BY 4.0);
FLEURS (Google, CC BY 4.0). Data was not modified or redistributed."
