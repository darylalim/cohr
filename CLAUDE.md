# cohr: cited meeting notes on MLX

Local Streamlit app: recording + slides → multilingual notes where every claim cites
[mm:ss] or [slide N]. Design: PLAN.md (full). Progress and findings: NOTES.md.
Current phase: see the last entry in NOTES.md.

## Commands (always through uv)
uv sync --all-groups · uv run ruff format . · uv run ruff check --fix . · uv run ty check
uv run pytest (no models) · uv run streamlit run app.py · uv run cohr run … · uv run cohr eval --set quick
Eval flags: -c seed=N · --no-ask --no-judge (fast loop) · --rerun STAGE (code-only changes, e.g.
the number guard, don't move a stage's config hash; re-runs that stage onward; --fresh redoes
every stage) · --update-baseline (instant right after a run)
uv run cohr eval --set long --baseline evals/baseline-long.json (~40 min; ask first)
uv run cohr eval --set lang -c seed=1 --no-ask (de, ar, beta sw; report-only, ~10 min)
uv run python scripts/trace_facts.py evals/reports/<ts>.json: each planted fact's path transcript → Map → notes
uv run python -m cohr.schemas --write-snapshots: after a FORMAT_VERSION bump

## Hard rules
- Python 3.13, uv only. Never pip. Add deps with `uv add` (dev: `--dev`, eval: `--group eval`).
- All model inference on MLX (mlx-audio / mlx-speech, mlx-vlm, mlx-lm). No torch (CI fails on it).
- Import mlx, mlx_lm, mlx_vlm, mlx_audio, mlx_speech only inside functions in the model
  wrappers (vad, asr, vision, mapper, reduce, judge, models). Unit tests run in CI without a GPU.
- Run files are read and written only through src/cohr/schemas.py. Changing a model means
  bumping its FORMAT_VERSION and regenerating snapshots (PLAN §19).
- Every output file is written atomically (temp file + os.replace).
- Prompts live in src/cohr/prompts/; don't inline prompt text in code.
- Citations must survive every transformation (two-pass reduce, translation). Test it.
- Reduce output uses fixed English section markers (## SUMMARY …); never parse localized
  headings. Uncited claims are kept and flagged, never given invented citations (PLAN §5.6).
- Streamlit: theme only via .streamlit/config.toml, no custom CSS; pages in app_pages/;
  never use_container_width; no telemetry (gatherUsageStats=false).
- Never commit recordings, frames, model weights or runs/. Never send user recordings anywhere.
- New dependency → check its license (PLAN §11 rule: no in-process GPL/AGPL).
- Don't bump the version, push, merge, or change GitHub settings without asking.
- Don't run `pytest -m models` or `cohr eval --set full` without asking (slow, ~15 GB).

## Eval and fix lessons
- Small local models: prompts set the output's shape; fix values (digits, days, slide conflicts)
  in pure-Python checks against the transcript/slides. Prompt edits to Map reshuffle which facts
  it keeps (temp 0.3): judge them on the mean of seeds 1–3, never one seed.
- quick results don't carry over to long-en (denser segments, two-pass Reduce) or to lang
  (non-English notes); check the sets a change can reach.
- When a model ignores a format rule, start its reply instead of rewording the prompt: R7B
  translated the section markers in de/ar until Reduce prefilled `## SUMMARY` (6d-1).
- Tiny Aya translates example lines in a translation prompt along with the real ones: keep
  translate.md example-free. It also changes digits when translating: check values in Python.
- Example lines in prompts must not overlap evals/gen/script.py's task or topic wording.
- A fallback needs a test where it must not fire (a too-strict filter hid behind one in 6b-2).
- Read a meeting's numbers in its own language (`numbers.source_numbers_in`), notes in the
  notes language: Transcribe writes es "11.800", ja "35万4000", and the English reading (11.8;
  35 and 4000) let the guard keep wrong values (6d-2). Run an es/ja meeting before guessing.
- A model override that scores exactly like the default on every seed didn't apply: check the
  run log's "loading <model> from" line (eval overrides loaded the defaults until 6d-2).
- Before building on an external dataset, check its real layout and file sizes (Hub tree API,
  HEAD requests), not just its name and card: `edinburghcstr/ami` is per-utterance clips, not
  meetings, and FLEURS needs 5.9 GB of test tars for 280 clips (6e-1, §18). The AMI mirror
  cut a scripted download, then answered 403: ask the user to download in a browser into
  evals/data/ami/ (`ami.ensure` skips files already there).
- Notes missing a section ("missing section markers" in the run log) or ending mid-citation:
  count the tokens of `raw` in notes.json against `reduce.max_tokens` before blaming the
  prompt. R7B's citation markup is token-heavy; 4 of 5 AMI notes hit 2048 silently (6e-1).
- A set always run with a flag (lang with `--no-ask`) never tests what the flag skips: run a
  new item once with Ask and the judge on (6e-1: the first full run crashed on xling-es Ask).
- Changing any Reduce prompt moves REDUCE_HASH in tests/test_ask.py; update it only on purpose.
- Any evals/gen/*.py edit regenerates each item's media on its set's next run; long runs only at
  the release gate, so its baseline can be phases older (6e-2: new long-en audio vs a 6b-2
  baseline). To split code from input, replay the stage with main's code (git worktree) on a
  copy of the run and score both.
- Pass 2 reads pass 1's parsed notes, so Python on pass-1 claims (6e-2's decision → action
  move) changes what pass 2 writes: check long-en, not only quick.
- A PLAN section isn't the code: the §16 release job was planned but never in ci.yml, so 1.0.0
  merged without a release (6e-2). Check the workflow before saying a merge releases.
- The guard hook blocks Bash commands containing the full-eval or models-marker text (heredocs
  too): write such text with the Edit/Write tools. It also blocks `rm -rf`: copy scratch
  output to new folder names instead of deleting.

## Where things are (PLAN.md)
Architecture §3 · Memory §4 · Stages §5 · Layout & tooling §6 · UI, layout, theme §7 ·
Tests §8 · Phases §9 · Storage §12 · Hooks §14 · CI §15 · Releases §16 · Evals §18 ·
Schemas §19 · Open decisions §20

## Each session
Start: CLAUDE.md → PLAN sections for the phase → last NOTES.md entry.
End: tick PLAN checklist, fill §20 results, add a NOTES.md entry, gate green, open PR.
