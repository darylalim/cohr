# cohr
Independent project. Not affiliated with or endorsed by Cohere.

Local, offline meeting notes for Apple Silicon. A recording plus slides (or a screen-recording
video) becomes notes in which every claim cites its timestamp `[mm:ss]` or `[slide N]`. All
models run on MLX; nothing leaves your machine.

Status: in development (Phase 1: scaffold + ingest). See `PLAN.md` and `NOTES.md`.

## Requirements
- Apple Silicon Mac, Python 3.13, [uv](https://docs.astral.sh/uv/), `ffmpeg` on `PATH`.

## Usage
```
uv sync
uv run cohr run meeting.m4a --slides deck.pdf     # creates runs/<id>/ (ingest only for now)
uv run cohr runs                                  # list runs with size and status
uv run cohr resume <run>                          # re-run from the first incomplete stage
uv run cohr prune <run>                           # drop media, keep notes and JSON
uv run cohr rm <run>                              # delete a run
```
Settings: `COHR_RUNS_DIR` (default `./runs`), `COHR_HF_HOME` (model cache location).

## License
Two layers with separate licenses:
- **This project's code:** Apache-2.0 (`LICENSE`, `NOTICE`).
- **Model weights:** downloaded at runtime, never redistributed, governed by their authors'
  licenses. Cohere Transcribe and North Micro Vision are Apache-2.0, Silero VAD is MIT, and
  Tiny Aya Global and Command R7B are **CC-BY-NC-4.0**. With the default models the app is for
  personal and research use only.
