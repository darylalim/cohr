# cohr
Independent project. Not affiliated with or endorsed by Cohere.

Local, offline meeting notes for Apple Silicon. A recording plus slides (or a screen-recording
video) becomes notes in which every claim cites its timestamp `[mm:ss]` or `[slide N]`. All
models run on MLX; nothing leaves your machine.

Status: v1.7.0. The pipeline, the app and the evals meet the v1 targets (Phase 6e); 1.1 adds
cross-meeting search (Phase 7); 1.2 places citations at their moment when the notes and the
meeting are in different languages (Phase 8a); 1.3 drops action-item due dates the meeting
didn't say and moves vague owner-less tasks to Key points (Phase 8b); 1.4 places each
citation by its best-matching span and checks Map's dates and dues against the meeting, so
Ask answers more questions on long meetings (Phase 8c); 1.5 finds action items the notes
missed: tasks filed as decisions or without their owner, and tasks Reduce dropped (Phase 8d);
1.6 adds the tasks Map never wrote: a sentence that gives a named person a task is checked by
one small model call and added with that person as owner (Phase 8e); 1.7 adds the decisions
the notes lost: a sentence like "We agreed to …" that no decision states becomes one, cited at
that sentence (Phase 8f). Recordings in 14 languages (Transcribe's), notes in Command R7B's
23 languages plus "Other (beta)" languages translated by Tiny Aya. See `PLAN.md` and
`NOTES.md`.

## Requirements
- Apple Silicon Mac, Python 3.13, [uv](https://docs.astral.sh/uv/), `ffmpeg` on `PATH`
  (`brew install ffmpeg`).
- Memory: the default `resident` policy keeps each model loaded once it's used (peak ~17 GB:
  32 GB Macs). `sequential` loads one model at a time (peak ~6 GB, same notes): choose it
  as the Memory policy on the New run page on a 16 GB Mac.
- Disk: ~14.4 GB for the models (below), plus each run's media unless you prune it.
- Speed (M2 Max, 32 GB): about 6 minutes per hour of audio, audio only; slides add Vision.

## Setup
```
uv sync
uv run cohr models download     # the five pipeline models, pinned revisions (~14.4 GB)
uv run cohr models              # what's downloaded, with sizes
```
`cohr models download` checks free space first and resumes an interrupted download when run
again. Models go to the Hugging Face cache (`~/.cache/huggingface/hub`), or to
`$COHR_HF_HOME/hub` when that's set. Only `cohr models download` uses the network; runs never do.

## App
```
uv run streamlit run app.py
```
Pages: **New run** (upload or link a recording and slides, pick the languages), **Notes** (cited
notes, with a Sources panel of the audio and slides beside or below them), **Ask** (questions about a run,
answered with citations), **Search** (words, names and numbers across every run's transcript,
slides and notes, and every meeting's action items by owner), **Runs** (status, size, prune,
delete).

## Command line
```
uv run cohr run meeting.m4a --slides deck.pdf --target de   # creates runs/<id>/ and processes it
uv run cohr run --video screen.mp4 --lang es                 # slides from the video's frames
uv run cohr runs                                             # list runs with size and status
uv run cohr resume <run>                                     # re-run from the first incomplete stage
uv run cohr prune <run>                                      # drop media, keep notes and JSON
uv run cohr rm <run>                                         # delete a run
uv run cohr search "budget" --kind note                      # search every run
uv run cohr index rebuild                                    # rebuild runs/index.db
```
The search index (`runs/index.db`) is rebuilt from the run files whenever it's missing or
out of date, so deleting it is always safe.
Settings: `COHR_RUNS_DIR` (default `./runs`), `COHR_HF_HOME` (model cache location).

## Deleting models
Remove one model with `hf cache rm` (it ships with huggingface_hub; add
`--cache-dir "$COHR_HF_HOME/hub"` if you set `COHR_HF_HOME`). Run it with `--dry-run` first:
```
uv run hf cache rm model/mlx-community/c4ai-command-r7b-12-2024-4bit --dry-run
uv run hf cache rm model/mlx-community/c4ai-command-r7b-12-2024-4bit
```
`uv run cohr models` lists the five repos. **Don't use `hf cache prune`**: it removes revisions
that no branch points to, and cohr pins exactly such revisions, so it deletes cohr's models
(its dry run lists R7B, Tiny Aya, North Micro Vision and the judge). If you use it for other
caches, run `hf cache prune --dry-run` first. `cohr models download` fetches anything removed.

## Evals
Quality is measured on meetings with known facts (`evals/README.md`, PLAN §18):
- `quick`: four 5-minute English meetings (clean, noisy, PDF deck, misspoken number); ~10 min.
- `lang`: notes in German, Arabic, Hindi, Swahili and Bengali; Spanish and Japanese meetings.
- `long`: one 2.5-hour meeting (two-pass notes and Ask at scale); ~1 h.
- `full`: `quick` + `lang` + FLEURS clips in 14 languages + 5 real AMI meetings; the
  release check, ~1 h.

## License
Two layers with separate licenses:
- **This project's code:** Apache-2.0 (`LICENSE`, `NOTICE`).
- **Model weights:** downloaded at runtime, never redistributed, governed by their authors'
  licenses. Cohere Transcribe and North Micro Vision are Apache-2.0, Silero VAD is MIT, and
  Tiny Aya Global and Command R7B are **CC-BY-NC-4.0**. With the default models the app is for
  personal and research use only.
- **Eval data** (`full` only, fetched at eval time, never committed): AMI Meeting Corpus
  (University of Edinburgh et al.) and FLEURS (Google), both CC BY 4.0.
