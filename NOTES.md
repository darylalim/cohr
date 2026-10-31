# cohr: implementation notes

Findings, measurements and decisions per phase. Results for the PLAN §20 decisions table are
recorded here in detail and copied into the table's **Result** column.

Entry template:

```markdown
## YYYY-MM-DD · Phase N (<name>)

### Findings
- …

### Decisions (PLAN §20)
| # | Result | Evidence |
|---|---|---|

### Plan changes
- §4: GPU limit on this M2 Max is 26.8 GB (measured), not ~21.5 GB; memory plan still holds.
- §2 / §9 go/no-go: the 4-bit Transcribe fallback is unusable (mlx-audio-swift format).
- …

### Next
- …
```

---

## 2026-10-10 · Phase 0 (checks before building)

### Findings
- `.gitignore` created from PLAN §6; `git check-ignore` check passes (runs/, *.mp4,
  *.safetensors, .env ignored; uv.lock, .claude/settings.json, NOTES.md not ignored).
- Hooks (§14) created: guard-bash, protect, ruff-file, gate. The 7 §14 pipe tests plus 16 extra
  cases (sudo, rm -rf runs, `pytest -m models` vs `-m "not models"`, `gh api -X PUT`,
  `gh pr create` allowed, absolute paths for runs/.env/HF cache) all pass.
  Small additions beyond the §14 sketches:
  - `ruff-file.sh` also skips when `.venv/bin/ruff` doesn't exist yet (between `uv init` and
    `uv add --dev ruff`), not only when `pyproject.toml` is missing.
  - `gate.sh` treats pytest exit 5 ("no tests collected") as a pass, for early Phase 0 before
    `tests/` exists.
  - `guard-bash.sh` also catches `-m 'models'` (single quotes) and `python -m pip install`
    with any spacing.
- Hooks confirmed live after restart: all four in `/hooks`; an unquoted `pip install` in a
  Bash call was blocked. (A quoted `"pip install"` passes, as intended: the regex needs a
  separator before `pip`.)
- uv project: `uv init --package --name cohr --python 3.13` (uv 0.12.22). Runtime deps, `dev`
  (ruff 0.17.0, ty 0.0.86, pytest 9.1.1) and `eval` (jiwer 4.0.0, pyyaml 6.0.3) groups added.
  `uv sync --all-groups` resolves 91 packages on **Python 3.13.12**; every runtime package
  imports: mlx 0.32.3, mlx-lm 0.32.0, mlx-vlm 0.7.7 (git), mlx-audio 0.5.8, mlx-speech 0.5.3,
  streamlit 1.65.0, pypdfium2 5.14.0, imagehash 4.3.2, lingua 2.2.0, soundfile 0.14.0.
- `mlx-vlm` pinned to main commit `952d4f6bc65bd5095e74abe71c78038764ee08aa` (2026-10-10) in
  `[tool.uv.sources]`. Re-pin if the Micro Vision check fails on it (§9 go/no-go).
- **ruff 0.17 formats Python blocks inside Markdown** by default, so the gate failed on a
  PLAN.md snippet. Fix: `extend-exclude = ["*.md"]` in `[tool.ruff]`. The rest of the §6 ruff
  config (line length, target, rules) was added at the same time; ty/pytest config stays in
  Phase 1.
- System: ffmpeg 7.1.1, jq 1.7.1. Disk: 43 GB free (91% used), enough for ~15 GB of models
  + 2 GB margin.
- **License audit (runtime + dev + eval, from package metadata):** no GPL/AGPL. Permissive:
  MIT, MIT-0, MIT-CMU (pillow), BSD-2/3, 0BSD, ISC, Apache-2.0, PSF-2.0, Zlib, CC0 (inside
  numpy), CNRI-Python (regex). Metadata empty but classifier/repo permissive: lingua (Apache-2.0),
  jinja2/itsdangerous (BSD), safetensors/tokenizers (Apache-2.0), ty (MIT). opencv-python is
  Apache-2.0 (its wheel bundles LGPL ffmpeg libs; LGPL is allowed). **Outside the §11 list:
  MPL-2.0** in `certifi` and `tqdm` (MPL-2.0 AND MIT). Both are transitive, used unmodified.
  Raised with the user.
- **MPL-2.0 accepted by the user (2026-10-10)** for transitive dependencies used unmodified
  (`certifi`, `tqdm`). PLAN §11 dependency rule to be updated in the Phase 0 PR.
- Model repos checked on the Hub 2026-10-10: all ungated; card licenses match §11
  (Transcribe, Micro Vision, Qwen3 judge, 4-bit Transcribe: apache-2.0; Tiny Aya, R7B:
  cc-by-nc-4.0; Silero VAD: mit). Download total 19.2 GB incl. the 4-bit Transcribe (D4).
- API reading: **neither Transcribe runtime auto-detects the language.** mlx-audio requires
  `language=` and validates it; mlx-speech puts a language token in the decoder prompt
  (default `"en"`). mlx-audio's Cohere config has `max_audio_clip_s = 35.0`.
- Fixtures (`scripts/fixtures.py`, §8): 92 s two-voice `say` meeting (Samantha, Daniel) with a
  20 s noise+tones gap at 39.5–59.5 s, 3 PIL slides with known numbers, 30 s music/noise/silence.
  The default PIL font has no `•` glyph; slides use `-`.
- **Transcribe runtime comparison (D1)**, 8-bit repo, same weights and fixture, 4 chunks ≤30 s:
  | | mlx-speech 0.5.3 | mlx-audio 0.5.8 |
  |---|---|---|
  | Loads §2 repo | yes (`mlx-int8/`) | "loads" `mlx-int8/` with `strict=False` but weights don't map |
  | WER (fixture script) | **0.110** | 3.235 (gibberish, identical for any input) |
  | s per audio min | **1.96** | 3.27 |
  | load / peak memory | 0.7 s / **5.48 GB** | 1.8 s / 6.51 GB |
  mlx-speech's remaining WER is mostly number formatting ("$14,500", "3rd" vs spelled-out
  script). mlx-audio is kept only for Silero VAD.
- **Max input (D2):** mlx-speech chunks long input internally; words/s flat (2.85–2.95) at
  20/30/35/40/60 s vs 3.07 reference. No truncation.
- **Made-up text check (no VAD), mlx-speech:** silence and music → "The world is a very important
  part of the world." ×7; room noise → "the other one is" looped to the token limit. Strong
  evidence for VAD on by default (§5.1a).
- **VAD (§5.1a):** `mlx-community/silero-vad-v6` loads via `mlx_audio.vad.load(repo, revision=)`
  (0.8 s), card license MIT. `get_speech_timestamps(audio, sample_rate=16000, return_seconds=True)`
  → `[{start, end}]`. On the fixture: 0.6 s for 92 s audio, 20 MB peak.
  | | Silero (defaults) | `silence` fallback (floor+6 dB = −33.2 dB) |
  |---|---|---|
  | speech-frame F1 vs truth | **0.979** | 0.854 |
  | speech found inside the 20 s noise+tones gap | **0.0 s** | 20.0 s (whole gap) |
  | packed chunks (merge <0.5 s, ≤30 s) | 4, max 22.7 s | 4, max 30.0 s |
  Silero finds 0 s of speech in the 30 s music, noise and silence clips. The fallback can't tell
  noise/music from speech, so it would feed the gap to Transcribe (made-up text above).
- **Micro Vision (§5.3)** via mlx-vlm `952d4f6`: `mlx_vlm.load(repo, revision=)` 1.0 s, 3.15 GB;
  3/3 fixture slides valid JSON on the first try, titles exact, 8/8 table cells; 3–4.7 s per
  slide, 4.45 GB peak. Quirks for `vision.py` (Phase 2): `tables_md` came back as a list of row
  objects, not Markdown strings; empty `chart_notes` sometimes `"None"`. Normalize both.
  Typing: `generate()` expects `ProcessorLike`, `load()` returns `ProcessorMixin` (ty ignore).
- **Tiny Aya Map (§5.5)**: `mlx_lm.load(repo, revision=)` 1.1 s, 3.56 GB; whole fixture as one
  segment, temperature 0.3, 4 seeds × en/de/ja: **12/12 valid JSON**, ~5.6 s per call, 4.31 GB
  peak. Content complete (budget, split, dates, owners, 38%/31%, 42 tickets, open question).
  **Output language:** with the language instruction only at the top, de 2/4 and ja 0/4 came
  back in English (the English transcript at the end wins). Repeating it as the prompt's last
  line fixed it: **12/12 in the target language** (lingua). Kept in `prompts/map.md`.
  Made-up detail seen once (ja): "end of next week" became the due date "10月27日". Candidate
  Map rule for Phase 3: copy due dates as spoken, never compute them; the eval's made-up-items
  metric should catch it.
- **R7B `rag` citations (§5.6, D7)**: `mlx_lm.load(repo, revision=)` 2.0 s, 4.52 GB. Tokenizer
  templates: `default`, `tool_use`, `rag`; `rag` variables: `documents`, `enable_citations`,
  `preamble`. Working call on the underlying HF tokenizer (`tokenizer._tokenizer`):
  `apply_chat_template(conv, documents=[{"title", "snippet"}…], chat_template="rag",
  enable_citations=True, tokenize=False, add_generation_prompt=True)`. Documents render as a
  tool result keyed "0".."n" in list order. **First try, temperature 0:** 22 citations, format
  exactly `<co>span</co: 0:[i,j]>`, all indices valid (0–3), all 5 section markers in order;
  correct sources (amounts → slide 1, retention → slide 3, spoken decisions → segment, mixed
  claims → `[0,1]`). 13.5 s, 5.73 GB peak, 2,253-token prompt.
  For Phase 4: output is wrapped in `<|START_RESPONSE|>…<|END_RESPONSE|>` (strip); action items
  came back as `Priya: screenshots by 20 October`, not `[owner: …] [due: …]` (the §5.6 map.json
  fallback or a prompt example is needed).
  Card languages (23): en fr de es it pt ja ko zh ar el fa pl id cs he hi nl ro ru tr uk vi.
  All 14 Transcribe languages are included.
- **4-bit Transcribe (D4):** `beshkenadze/cohere-transcribe-03-2026-mlx-4bit` was converted for
  `mlx-audio-swift` (merged QKV, single `model.safetensors` + `key_map.json`). mlx-speech
  refuses it (strict: 795 checkpoint-only, 660 model-only, 145 shape mismatches); mlx-audio
  "loads" it non-strict and outputs the same gibberish for any input (WER 3.26). Not usable.
  The Micro Vision / Tiny Aya 4-bit repos were not downloaded: with 8.6 GB of headroom at a
  32K prompt (below) no 4-bit fallback is needed.
- **Qwen3 judge:** loads in 1.1 s, 2.52 GB peak; one claim/segment pair → valid JSON
  `{"supported": "yes", …}` in 0.5 s.
- **Spike (`scripts/spike.py --mem32k`) PASSES.** All five models loaded together (resident),
  chained on the fixtures, writing only to a temp dir:
  | Stage | s | peak GB |
  |---|---|---|
  | load all 5 | 7.6 | 15.35 (active) |
  | VAD | 0.5 | 15.37 |
  | Transcribe (4 chunks, 72 s speech) | 2.9 | 16.66 |
  | Vision (3 slides) | 13.5 | 16.66 |
  | Map (1 segment) | 5.8 | 16.08 |
  | Reduce (2.4K-token prompt) | 15.8 | 16.57 |
  | Reduce, 34,393-token prompt | 108.6 | **18.16** |
  23 citations, 0 dropped, all resolve to the segment. Map JSON valid. GPU limit
  (`max_recommended_working_set_size`) is **26.8 GB**, not the ~21.5 GB the plan assumed, so the
  resident policy with a 32K prompt has 8.6 GB headroom.
  Spike-only observations for Phase 4: (1) `## SUMMARY` follows `<|START_RESPONSE|>` on the same
  line, so marker parsing must strip the wrapper first; (2) with map notes in the segment
  document, R7B cited only the segment (slides were cited when they were separate documents
  without map notes, D7 run); (3) action items were split into separate `Owner:` / `Due:` /
  task lines, each cited. The action item format needs an example in `reduce.md`.
- **Speed (D10), projected for a 60 min meeting** (resident, from the measured rates):
  VAD ~20 s; Transcribe ~2 min (1.96 s per audio min); Vision ~3.5 min (45 slides × ~4.5 s);
  Map ~70 s (12 segments × 5.8 s); Reduce ~1.5 min (~18K-token prompt: ~315 tok/s prefill +
  ~50 tok/s generation); ingest (ffmpeg) ~0.5 min. **Total ≈ 9–10 min per audio hour**, under
  the 20 min target. The 32K prefill (≈110 s) is the largest single step; it only occurs for
  meetings near 2 h.

### Decisions (PLAN §20)
| # | Result | Evidence |
|---|---|---|
| D1 | **mlx-speech** (Auto, go/no-go row 1). mlx-audio can't load the int8 checkpoint correctly | Comparison table above |
| D2 | `max_chunk_s` = 30 s kept; no truncation up to 60 s (runtime chunks internally) | Words/s probe |
| D3 | No auto-detection in either runtime: the user picks the source language | Runtime APIs |
| D5 | Defaults kept: threshold 0.5, 250/100/30 ms, merge < 0.5 s | Speech F1 0.979, 0 s in gap; re-check on eval |
| D7 | `rag` template + `enable_citations=True` works as planned; format `<co>…</co: 0:[i]>`; 23 languages recorded | First try, 22 valid citations |
| D4 | 4-bit Transcribe repo **not usable** (mlx-audio-swift format, neither runtime). Not needed: 8.6 GB headroom | Load + WER |
| D8 | Voices for all 14 languages. ≥2 real voices: en, de, es, fr, it, ja, ko, nl, pt, zh. Only 1: ar (Majed), el (Melina), pl (Zosia), vi (Linh) | `say -v '?'` (counts include Eloquence voices like Eddy/Flo) |
| D6 | Pinned revisions below (downloaded at these SHAs); mlx-vlm `952d4f6bc65bd5095e74abe71c78038764ee08aa` | Hub `model_info` 2026-10-10 |
| D10 | **Realistic:** ≈ 9–10 min per audio hour projected (target ≤ 20) | Spike stage timings |
| D9 | No torch/torchvision/torchaudio in `uv.lock`; `torch` not in `sys.modules` after importing every runtime package | `grep '^name = "torch' uv.lock`; import check |

#### Pinned model revisions (D6)
| | Repo | Revision | Size |
|---|---|---|---|
| | `mlx-community/cohere-transcribe-03-2026-mlx-8bit` | `a0acb7f93cd32d82c4fbf801b6d8fb39d20c509f` | 4.13 GB |
| | `mlx-community/North-Micro-Vision-Instruct-8bit` | `26e6b3aef507899fbf86057593d99fc12418e26f` | 3.16 GB |
| | `mlx-community/tiny-aya-global-8bit-mlx` | `5d5dc68b477abd7d583563a515bed98604d45d52` | 3.58 GB |
| | `mlx-community/c4ai-command-r7b-12-2024-4bit` | `2ae601b3d780410caacd2148c6c96e620144f22d` | 4.54 GB |
| | `mlx-community/silero-vad-v6` | `2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06` | 0.0 GB |
| | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` | 2.28 GB |
| | `beshkenadze/cohere-transcribe-03-2026-mlx-4bit` | `104bc4391b5b1a12b040859793d7148525e1a08c` | 1.51 GB |

### Plan changes
- §5.5: the Map prompt repeats the target-language instruction at the end (Phase 0 finding).
- §5.2/§2: Transcribe runtime is mlx-speech; mlx-audio only for VAD (§1, §6 wording).
- §11 dependency rule: add MPL-2.0 (transitive, unmodified), user-approved.
- §6 ruff config gains `extend-exclude = ["*.md"]` (to update in PLAN §6 in the Phase 0 PR).

### Next
- Phase 0 complete. Open the Phase 0 PR (ask before pushing). Then Phase 1 (scaffold + ingest).

## 2026-10-11 · Phase 1 (scaffold + ingest)

### Findings
- **Modules:** `config.py`, `schemas.py`, `atomic.py`, `storage.py`, `pipeline.py`, `ingest.py`,
  `cli.py`. `LICENSE` (Apache-2.0 text from apache.org), `NOTICE` (§11 text), README License
  section, `license`/`license-files` in `pyproject.toml`. Entry point is now `cohr.cli:main`.
- **Config:** plain pydantic (frozen, `extra="forbid"`), not `pydantic-settings`: only two env vars
  (`COHR_RUNS_DIR`, `COHR_HF_HOME`), read by `config.runs_dir()` / `hf_home()`, so no new
  dependency. Sections per stage (`ingest`, `vad`, `fuse`, `reduce`) so each stage hashes only its
  own section. Pinned revisions from D6.
- **Hashes:** two per stage. `config_hash` = stage config keys + prompt file hash + FORMAT_VERSION;
  `input_hash` = chained from the inputs' sha256 through every earlier stage's hashes. Added
  `input_hash` to `StageState` (§19 table updated). Bumping a FORMAT_VERSION, editing a prompt or
  changing a section re-runs that stage and every later one; `pipeline.plan()` reports why
  (`format`, `config`, `input`, `missing output`, `not done`, `new`).
- **Manifest `config`** is stored as a loose JSON object, not the `Config` model, so adding a
  config field doesn't change the manifest snapshot. Stages re-validate it with `Config`.
- **Schemas (§19):** 8 run-file models with `FORMAT_VERSION`/`FILENAME` class vars, all
  version 1; snapshots in `tests/schemas/`. Additions to the §19 table: `InputFile.kind` and
  `.stored` (name inside `source/`, `NN-<name>` so two inputs can't collide), `Frame.page`.
- **Atomic writes:** `mkstemp`/`mkdtemp` create 0600/0700, so `atomic.py` applies the umask
  afterwards (outputs get normal modes). `frames/` is built in a temp dir and swapped in whole.
  `log.txt` is the one non-atomic file (an append-only log).
- **Ingest:** ffmpeg 7 wants `-fps_mode vfr` (`-vsync` is deprecated). The scene filter is
  `select='eq(n\,0)+gt(scene\,T)',scale=…,showinfo` with long-edge scaling for portrait sources;
  pts come from `showinfo` lines, and the frame count must match the timestamp count or ingest
  fails. `ffprobe` cover art (`attached_pic`) in audio files is not treated as video.
  Solid-colour test frames all share one phash, so test media draws patterns.
- **Run ids** use local time (UTC made `20261011-160526` for a 06:05 run).
- **Stages for Phase 1:** `ingest` only. `pipeline.default_stages()` grows as stages land.
  `HF_HUB_DISABLE_TELEMETRY=1` is set in `cohr/__init__.py`; `HF_HUB_OFFLINE=1` once cached is
  left to the ModelManager (Phase 2), which knows what's cached.
- **Tests:** 76 pass in ~4 s, no models (schemas 21, pipeline 17, storage 13, ingest 14,
  config 8, cli 3). ffmpeg tests skip if ffmpeg is missing; CI installs it.
  `tests/__init__.py` exists so `tests.conftest` imports resolve under ty.
- **CI (§15):** `.github/workflows/ci.yml` as planned. Actions are pinned to commit SHAs:
  `actions/checkout` v7.0.1 and `astral-sh/setup-uv` v10.3.0. setup-uv publishes only full
  version tags (no moving `v10`), so `@v10` failed the first run with "unable to find version".
- **§17 topics:** the §17 list has 17 topics, not 18; all 17 applied.
- **GitHub (§17), done 2026-10-11 with the user's OK:** topics (step 2), squash-only merges +
  delete branch on merge + wiki/projects off (step 3), Actions token read-only (step 4, already
  the default), branch protection on `main` requiring `check`, strict, no force pushes (step 5,
  after CI's first green run). Verified (step 6): CI ran 76 tests with none skipped (ffmpeg
  present on the runner), PR #3 shows `CLEAN` with `check` passing.
- **Not in Phase 1 (by plan):** release job (§16), ModelManager and download disk check (Phase
  2), upload streaming (Phase 5).

### Next
- Merge PR #3 (squash). Then Phase 2 (perception: `vad.py`, `asr.py`, `vision.py`,
  ModelManager with pinned revisions, offline mode once cached, free-disk check).

## 2026-10-11 · Phase 2 (perception)

### Findings
- **Modules:** `models.py` (ModelManager), `vad.py`, `asr.py`, `vision.py`; stages
  `ingest → vad → transcribe → vision`. Each wrapper's `load(path)` returns a small adapter
  (`SileroVad.detect`, `CohereAsr.transcribe`, `MicroVision.describe`); MLX imports live only
  inside those, so unit tests use fake adapters through the real ModelManager.
- **ModelManager:** one `RLock`; `use(name)` loads on first use and releases afterwards under
  `sequential` (VAD is always released). Load failures raise `ModelLoadError`; models that
  aren't downloaded raise `ModelsMissing` listing all of them. `pipeline.run(models=…)` checks
  every model needed by the stages that will run **before** any stage starts, so a fresh
  machine fails fast with "Run `uv run cohr models download`". Peak memory is reset per stage
  and read only if MLX is already imported (ingest-only runs and CI never import MLX).
- **Offline:** `huggingface_hub` reads `HF_HUB_OFFLINE` once at import, so setting it after
  checking the cache (§12 as planned) would do nothing. Runtime resolution uses
  `snapshot_download(local_files_only=True)` and loaders get local paths (all three loaders
  accept them). The smoke test passes with `HF_HUB_OFFLINE=1`, so no hidden Hub call remains.
- **Downloads:** `cohr models [list|download] [names]`. Download sizes come from
  `model_info(files_metadata=True)`; free space must cover missing weights + 2 GB before
  anything is fetched. `COHR_HF_HOME` → `cache_dir=<it>/hub`. Tests point it at an empty
  tmp dir (conftest), so they never see the real cache.
- **VAD:** `generate()` returns timestamps and one probability per 512 samples; the
  probabilities drive the long-run split (least-speechy frame in the last 5 s before
  `max_chunk_s`). New `max_inner_gap_s = 2.0`: the Phase 0 packer would have packed two runs
  around a 20 s noise gap into one chunk whenever both fit in 30 s, sending the noise to
  Transcribe. Gaps = complement of the chunks, so "gaps are never transcribed" holds exactly.
- **Silence fallback** in numpy (32 ms frames, floor + 6 dB, dips < 0.3 s filled) instead of
  `ffmpeg silencedetect`: same rule, no stderr parsing, unit-testable; its frame levels drive the
  long-run split the same way.
- **Vision:** `vision_retry.md` holds the retry instruction (prompt text stays in prompts/);
  `Stage.prompt` became `Stage.prompts` (tuple) so both prompt files feed the hash. Phase 0
  quirks handled: `tables_md` as a list of row objects → one Markdown table; row lists; `"None"`
  / `"null"` / `"N/A"` → `""`. Seed passed to `mx.random.seed` before every slide.
  `VisionConfig` (max_tokens 1024, temperature 0) so those settings are in the stage hash.
- **Transcribe:** chunks read with `soundfile.read(start=, stop=)`; text stripped; the 14
  languages are checked in the stage and by `cohr run --lang` choices.
- **Smoke test** (`tests/test_pipeline_smoke.py`, `models` marker, run by the user with
  `HF_HUB_OFFLINE=1`): passes in 26.8 s. VAD 1.4 s → 4 chunks, 69.5 of 92.3 s sent to
  Transcribe, none inside the noise+music gap; Transcribe 3.9 s; Vision 12.5 s, 3/3 slides valid
  JSON first try, "14,500" found; `cohr resume` skips all four stages.
- **Tests:** 112 pass without models in ~4 s (new: vad 11, models 10, vision 6, asr 4, cli +3,
  pipeline +2). The guard hook blocks any command containing the models-marker flag, even
  in a heredoc, so test files that mention it are written with the file tools.
- **Schemas:** unchanged (Phase 1 models already matched); no FORMAT_VERSION bump.

### Decisions (PLAN §20)
None due this phase (D11 is Phase 3).

### Plan changes
- §5.1a: `max_inner_gap_s` (2 s) and `split_search_s` (5 s) in `VadConfig`; silence fallback in
  numpy instead of ffmpeg.
- §12: offline by local-only resolution, not by setting `HF_HUB_OFFLINE` after a cache check.
- Known trade-off: `input_hash` chains linearly, so a VAD or language change also re-runs
  Vision (it doesn't depend on them). ~13 s per 3 slides; revisit if it matters (a stage could
  declare its real inputs). `vad` is preflight-required even with `chunker = "silence"`.

### Next
- Open the Phase 2 PR (ask before pushing). Then Phase 3 (fuse + map: slide alignment, D11,
  `segments.json`, `map.json`, token budget).

## 2026-10-11 · Phase 3 (fuse + map)

### Findings
- **Fuse (`fuse.py`, no models):** slides without `pts` are text-aligned to transcript chunks
  with a monotonic DP (`best[i][j]` = best total for the first i slides on chunks < j; skips
  allowed, two slides may share a chunk, ties go to the earliest chunk). Score = shared tokens
  / max(min(|slide|, |chunk|), 8): plain slide coverage punished long slides (a 50-token slide
  rarely reaches 0.15 against one 30 s chunk), plain overlap coefficient let a two-word chunk
  score 0.5. English stopwords plus a document-frequency filter (≥ 4 chunks: drop tokens in
  more than half of them) so function words in the other 13 languages and the project name
  don't count. CJK runs → character bigrams.
- Fuse **rewrites `slides.json`** (`t`, `t_source="aligned"`, `align_score`) as §5.4 says, and
  recomputes every non-`pts` slide on each run, so a threshold change gives the same result as
  a first run (tested). Vision's skip check only looks at existence, so this is safe.
- Slides per segment: timed slides in (previous segment end, this end], first/last open-ended,
  plus the slide on screen at the start. A slide flipped during a silence lands in the next
  segment instead of nowhere.
- **Token budget** is estimated (1 per CJK char, 1 per 3 other chars), because Fuse loads no
  model; a 5 min window of speech is ~1.5K estimated tokens, so splits only happen with heavy
  slides. Map counts real tokens with Tiny Aya's tokenizer and warns when over.
- **Map (`mapper.py`):** `fill()` substitutes the four placeholders in one regex pass (the
  prompt has literal JSON braces; a transcript containing `{slides}` is never re-substituted).
  Retry = prompt + `map_retry.md`, which repeats the target language so the last line still
  names it. Coercion: list `summary` joined, string `terms` → list, string action items →
  `task`, `"null"`/`"?"`/`"unknown"` owners and dues → null, empty tasks dropped.
- `mlx_lm.load` is typed as returning 2 or 3 values, so the loader indexes `[0]`, `[1]`.
- **D11 probe (no models, fixture script text as chunks):** 1 line per chunk: correct chunks
  score 0.33–0.56, best wrong 0.11–0.31 (the recap line, ruled out by monotonicity); 2 lines
  per chunk: right 0.23–0.38. An undiscussed "Hiring" slide: 0.125. Thresholds 0.10–0.20 all
  place 3/3; 0.25 drops one. **0.15 kept, provisional** until `pdf-slides-en` (Phase 4).
- **Tests:** 137 pass without models in ~4 s (new: fuse 17, mapper 9; pipeline order updated).
  Smoke test extended to fuse + map (slides must align in order onto the chunks that discuss
  them; Map JSON valid with decisions, action items and 14,500): **passes in 34.3 s** offline
  (Phase 2: 26.8 s, so fuse + map add ~7.5 s), run by the user.

### Decisions (PLAN §20)
- D11: 0.15 provisional (above). Re-check on `pdf-slides-en` in Phase 4.

### Plan changes
- §19: `SegmentNotes.raw` is `bool` (was `str`), `MapFile` FORMAT_VERSION 2, snapshot regenerated.
- §5.4: score formula, DF filter, CJK bigrams, `t_source` values `pts|aligned|none` (the schema's,
  not the plan's old `video|text|null`), slide-per-segment rule, estimated token budget
  (`fuse.max_segment_tokens = 5000`).
- §5.5: `map_retry.md`; due dates copied as spoken (Phase 0 candidate rule adopted).
- `config.py`: `MapConfig` (temperature 0.3, max_tokens 1024) and `TARGET_LANGUAGES` (R7B's
  23, §5.6 said Phase 0 would record them there; it hadn't).

### Next
- Open the Phase 3 PR. Then Phase 4
  (reduce + render, eval generator, D11 re-check, D12, D15, D19).

## 2026-10-11 · Phase 4a (reduce + render)

### Findings
- **Documents:** one per segment (title `0:00–1:32 · slides 1,2,3`, snippet = map notes as
  labelled lines + transcript) and **one per slide, aligned or not** (`slide 2: <title>`,
  slide text + chart notes). Phase 0 showed R7B cites only the segment when slide text sits
  inside it, which would leave `[slide N]` citations (and the Phase 5 slide anchors) unused.
- **Reply parsing (`reduce.py`, pure):** strip `<|START_RESPONSE|>…<|END_RESPONSE|>` first
  (the first marker shares the wrapper's line); split on the fixed markers (case, spaces and
  `**`/`:` around a marker normalized, still English-only); unknown headings go to Summary
  and are logged. `<co>` spans are atomic pieces, so a citation always stays with its claim.
  Summary splits into sentences (`.!?` + space, `。！？`), other sections into list items.
- **Citations:** `0:[i,j]` → [i, j] (the 0 is the tool call); out-of-range indices dropped and
  counted; first occurrence of each target kept. Refinement = §5.4 overlap of the cited span
  against the segment's chunks, ≥ `citation_min_score` (0.1, D19) or the segment's `t0`.
- **Action items:** `[owner: …] [due: …] task` (either bracket optional, `?` → null); else the
  map.json fallback (overlap = shared / smaller token set ≥ 0.6, cited segments only). The
  prompt now shows two example lines (Phase 0 got `Priya: screenshots by 20 October`).
- **Over the 32K cap:** transcripts trimmed 25% at a time, stage fails below 300 chars per
  segment with a "two-pass reduce" message. Map notes and slides are never trimmed.
- **Render:** a separate no-model `render` stage (`notes.md`, FORMAT_VERSION 1), so label
  changes don't re-run R7B. `LABELS[lang]` (23 languages × 9 words) replaces the plan's
  `SECTION_TITLES`. Portable: `[12:47]` / `[slide 4]`, `[uncited]`, Sources appendix; app mode
  (for Phase 5): `[12:47](#seg-3)` / `[slide 4](#slide-3)` (anchors use ids, labels `id + 1`).
- `NotesDoc` v2 adds `raw` (the reply as generated) instead of a separate `notes_raw.txt`, so
  every run file still goes through `schemas.py`. `ReduceConfig` gains `max_tokens` 2048 and
  `temperature` 0.0.
- **Tests:** 161 pass without models in ~4 s (new: reduce 18, render 6). Smoke test extended
  to reduce + render (≥ 1 resolving citation, refined `t` inside the segment, notes.md with
  Sources and no anchors): **passes in 55.4 s** offline, run by the user (Phase 3: 34.3 s;
  reduce 21.5 s on a 2,698-token prompt, 4 documents, render 0.0 s).
- **Smoke-test notes:** 18 claims, **0 uncited, 0 dropped**, all 5 markers in order; slide
  citations now appear (slides 1–3 cited across key points and decisions); refined times 0:00,
  0:17, 1:00 instead of all 0:00; action items came back in the exact `[owner] [due]` format.
  R7B copies the map notes' action lines, so their owner/due are only as good as Map's.
  **Error found (upstream, for 4b):** Tiny Aya gave Priya's screenshots item Tom's due date
  ("end of next week"; the task text itself says "by October 20"), and R7B kept it. The 4b
  made-up-items metric should count this; candidate fix is in `map.md`, not in reduce.
  Summary was 1 sentence (the prompt asks for 2–4).

### Decisions (PLAN §20)
- None final. D19 (0.1) used as-is; measured in 4b.

### Plan changes
- §3, §5.6: every slide its own document; `raw` in notes.json; over-cap trimming until Phase 6;
  claim splitting rules; `LABELS`; render as a stage; uncited flag in both modes.
- §9: Phase 4 split into 4a (this) and 4b (evals, D11/D12/D15/D19).
- §19: `NotesDoc` v2 (`raw`), snapshot regenerated.

### Next
- Merge the Phase 4a PR. Then 4b (planted-facts generator, `cohr eval
  --set quick`, D11 re-check, D12, D15, D19).

---

## 2026-10-11 · Phase 4b planning (no code)

### Findings
- Phase 4b is split in two PRs, like 4a, because D15 needs 30 hand-labeled claim/segment pairs:
  Claude can draft the labels from real pipeline output, but the calibration only means something
  once the user has reviewed them. Keeping the judge separate means that review doesn't block the
  generator, metrics and the other decisions.
- `quick` is the four English variants only. The cross-language scoring code is built and
  unit-tested in 4b-1; the `xling-*`, `en → de/ar` and `long-en` variants, FLEURS and AMI stay in
  `full` (Phase 6).

### Decisions (PLAN §20)
- None. D11, D12, D19 move to 4b-1; D15 to 4b-2.

### Plan changes
- §9: 4b split into **4b-1** (generator, exact metrics, cross-language exact-field scoring,
  `cohr eval --set quick`, reports, `test_evals.py`, first numbers, D11/D12/D19, fix for the 4a
  Map due-date error) and **4b-2** (Qwen3 judge, support rate, judge recall, calibration set, D15).
- §20: D11, D12, D19 → Phase 4b-1; D15 → Phase 4b-2.

### Next
- Phase 4b-1 in a new session.

---

## 2026-10-11 · Phase 4b-1 (evals without the judge)

### Built
- `evals/gen/`: seeded planted-facts meetings. `script.py` (pure) picks 8 topics, 4 decisions,
  5 action items (owner + spoken due date) and one number per topic, plus fact-free discussion
  and filler; `build.py` speaks it with `say` (3 voices, byte-identical per text), adds numpy
  pink noise/chords/silence, draws PIL slides (coloured header per slide), and makes a 5 fps
  screen recording (ffmpeg concat) or a PDF deck, plus `truth.json`. Items are ~340–360 s
  (two 300 s Fuse windows), 17 facts, 8 slides. Every number in a meeting is distinct and
  > 31, so a value means one fact and never a date.
- `src/cohr/evals.py`: truth/set models, matching, metrics, reports, baseline gate, runner.
  `cohr eval --set quick [-c key=value] [--item] [--seed] [--fresh] [--baseline]`. Each item
  keeps one run under `evals/data/runs/<item>/`; the pipeline's skip logic re-runs only what an
  override or prompt change touches, so D12/D19 experiments and re-scoring are cheap (a
  re-score with nothing stale takes < 1 s and loads no model).
- `tests/test_evals.py` (37 tests, no models): number normalization, owner/due/text matching,
  cross-language exact fields, citation tolerance, made-up items, VAD F1, gap text, slide
  matching, baseline thresholds, overrides, generator determinism (fake `say`), and
  `score_item` on a hand-built run.
- Matching rules beyond §18: stemmed content tokens (fact tokens found ≥ 0.6); due dates
  compared as normalized token sets ("the 22nd of November" = "Nov 22"); an action item
  with the right task and someone else's owner/due counts as made up; with no owner stated it
  is "partial" (not found, not made up: R7B repeats action items under Decisions).

### First numbers (`quick`, 4 items, 68 facts, 23 min of audio)
| Run | Change | Recall | Cit. acc. | Made up | Slide titles / cells |
|---|---|---|---|---|---|
| 2 | before (old ingest, old Map, old Reduce) | 82.3% | 60.7% | 2 | 28% / 34% |
| 3 | ingest fix | 67.7% | 63.0% | 5 | 88% / 100% |
| 4a | + Reduce marker fix + D19 sentence offset | 79.4% | 98.2% | 5 | 88% / 100% |
| 4b | + Map prompt fix (final) | 82.3% | 98.2% | 3 | 88% / 100% |
| 5 | 4b with Map temperature 0 (rejected) | 82.3% | 100% | 4 | |
| 6 | 4b with Fuse window 120 s (D12, rejected) | 79.4% | 100% | 6 | |
| 7 | 4b again, `--baseline` run 4b | 82.3% | 98.2% | 3 | pass (reproducible) |

Final (run 7): uncited 0, VAD gap text 0 words, VAD speech F1 93.1%, WER 0.9%, slide numbers
100%, alignment 100% (`pdf-slides-en`), output language 100%. Cost 890 s per audio hour total
(~15 min per audio hour, under the 20 min target); Vision is now the largest stage (347 s/h)
because video items yield all 8 slides; peak 16.8 GB (resident).
Against the v1 targets: recall 82.3% (< 85%), citation accuracy 98.2% (≥ 90%), made-up 3 (> 0).

Run 1 (before the generator fixes below) is not comparable; it found the generator issues.

### Findings
- **Ingest dropped all but the first slide of a deck recording** (fixed, user approved in
  session): ffmpeg scene scores between white text slides are 0.04–0.07 (threshold was 0.3), and
  8×8 phashes 4–8 bits apart (limit 6). Now `scene_threshold` 0.02, `phash_size` 16,
  `dedupe_max_distance` 24 (16×16: 38–60 apart). All 8 frames found; slide titles 28% → 88%,
  cells and numbers 34% → 100%. Regression test fails under the old defaults (1 frame of 3).
- **Reduce marker parsing:** R7B sometimes cites the marker itself (`## <co>SUMMARY</co: 0:[0]>`);
  the whole reply then landed in Summary (`slide-conflict-en` recall 23.5%). Tags are now
  stripped from heading lines (test added).
- **Citation jump time (D19):** the chunk's `t0` was a median 12.5 s early (chunks up to 30 s).
  Placing `t` at the cited sentence's character position inside the chunk: 61% → 98% within
  ±15 s, offline on stored replies and confirmed by a real run. `NotesDoc` v3 (snapshot
  regenerated) so older runs re-run Reduce.
- **Map due dates (the 4a error):** Tiny Aya wrote ISO dates with an invented year
  (`2024-11-24`, `2024-10-31` for "end of next week") despite the rule. `map.md` now says to take
  owner/due only from the task's sentence, copy the words said (no numeric dates, no year), and
  not to turn past events into action items. ISO dates: gone. Made-up items 5 → 3, recall
  79.4% → 82.3% (runs 4a → 4b, same seeds; examples in the prompt avoid generator values).
- **Open: Map changes digits.** Tiny Aya turns "$42,500" into "$420,500", 336 into "3,360",
  "22nd" into "2nd"; the transcript is right each time. The prompt rule didn't help and
  temperature 0 doesn't either (run 5), so it's systematic. Most remaining number misses and
  one made-up item come from it. Candidate (not done): a deterministic guard in `mapper.py`
  that checks every number in Map's output against the segment's transcript and slide text.
- **Open: slide conflict.** With the slide visible, Map used the misspoken value ($80,000)
  over the slide's $18,000 despite the §5.5 rule.
- **Remaining made-up items:** "Eleanor" for Elena (ASR: the transcript says Eleanor), "Nov 2nd"
  for 22nd (Map digits), one "update the dashboard" from a past-tense line (noisy-en).
- **Generator fixes found by run 1:** "Sofia" → "Sophia" and "Priya" → "Kreia" in ASR, so owners
  are now names with one common spelling (Elena still came back as Eleanor once: candidate
  swap in Phase 6); a discussion line ("keep an eye on X over the next few weeks") read as an
  action item and was removed.
- Fact recall per item is noisy with 17 facts (one fact = 5.9 points); D16 (3 repeated runs)
  should come before trusting differences of a few points.

### Decisions (PLAN §20)
- D11 = 0.15 (final), D12 = 300 s, D19 = 0.1 / ±15 s plus the sentence offset.

### Plan changes
- §5.1 scene threshold 0.02, 16×16 phash ≤ 24; §5.5 Map rules; §5.6 sentence offset;
  §18 made-up rule details; §19 `NotesDoc` v3; §9 4b-1 done; §20 D11/D12/D19.

### Next
- Phase 4b-2 (judge) in a new session. Before or alongside it, decide the Map digit guard
  (recall is below the 85% target mainly because of it) and the slide-conflict miss.

---

## 2026-10-11 · Phase 4b-1b planning (no code)

### Findings
- The 4b-1 digit errors all change one digit or a factor of 10 ($42,500 → $420,500, 57,000 →
  570,000, 336 → 3,360, 22nd → 2nd, $59,500 → $595,000, $11,800 → $1,180), and the transcript
  was right each time. A prompt rule and temperature 0 didn't help, so the fix is a
  deterministic check of Map's numbers against the segment's own transcript and slides.
- Doing it before 4b-2 means the judge's 30 calibration pairs are drawn from output without
  this known error.

### Decisions (PLAN §20)
- None. D20 (repair rule) is decided in 4b-1b from `quick` before/after.

### Plan changes
- §5.5 Number guard (repair when exactly one source number is one digit edit or a factor of
  10/100/1000 away, else keep and flag); §6 `numbers.py`; §9 Phase 4b-1b; §19 `MapFile` v3
  (`unverified_numbers`); §20 D20; §21 kickoff prompt.

### Next
- Phase 4b-1b in a new session, then 4b-2.

---

## 2026-10-11 · Phase 4b-1b (Map number guard)

### Built
- `src/cohr/numbers.py`: `digits`/`numbers_in` moved from `evals.py` unchanged (only
  `test_evals.py`'s import line moved), plus the guard's reading of numbers *as written*:
  `WRITTEN` (skips digits glued to letters: "Q3", "R7B", "3D"), `readings` (every value the
  separators allow: "42,500" = 42500 or 42.5), `near` (one digit edit or ×/÷ 10, 100, 1000),
  `write_like` (repair in the output's separators and decimal places).
- `mapper.check_numbers` / `guard_numbers` (pure), applied per parsed segment in `run_stage`
  when `map.number_guard` (default on). Keep if any reading is in the segment's transcript +
  attached slide text; repair if exactly one source number is near; else keep and flag.
  `terms` and raw (JSON fallback) segments untouched. Suffixed ("42.5k") and plain-space
  grouped ("42 500") numbers are only kept or flagged. Repairs logged (info), flags warned.
- `MapFile` v3: `number_repairs[{original, repaired}]` (not in the 4b-1b plan; needed so the
  eval can count and check repairs from `map.json`) and `unverified_numbers[]`. Snapshot
  regenerated.
- Eval: per item and summary `map_numbers_repaired`, `map_numbers_unverified`,
  `unsupported_numbers` (numbers in `notes.json` claims in no transcript or slide text), with
  the values listed per item in the report; summary now also has `recall_decision/action/number`.
- Tests: `tests/test_numbers.py` (new), guard and stage tests in `test_mapper.py` (all §9 cases
  plus the two 4b-1b eval cases below), `test_evals.py` end-to-end item with a `map.json`.

### D20: `quick`, same seeds (guard off = `-c map.number_guard=false`)
| Run | Recall | Number recall | Cit. acc. | Made up | Repaired | Unverified | In no source |
|---|---|---|---|---|---|---|---|
| guard off (091204) | 82.3% | 81.2% | 98.2% | 3 | 0 | 0 | 6 |
| guard on (091454) | 85.3% | 87.5% | 98.3% | 3 | 8 | 2 | 1 |
| guard on, padding fix, `--fresh` (092133) | 85.3% | 87.5% | 100% | 3 | 8 | 2 | 1 |

Guard off reproduces 4b-1 run 7 exactly (Map is deterministic at the fixed seed). Every repair
checked against the transcript and `truth.json`: all 8 land on the transcript's value
($420,500→$42,500, $595,000→$59,500, $2,480,000→$248,000, $1,610,000→$161,000,
$570,000→$57,000, $540,000→$54,000, 4,246→426, 4,6500→4650). No wrong value. Flagged (correct):
`3,360` (336 and 330 both near), `2nd` (22 and small numbers near). Fact recall reaches the 85%
v1 target. **D20: default rule kept.**

### Findings
- **Padding bug (fixed in session):** Map wrote a malformed "4,6500"; it only reads as 4.65
  (decimal comma), ×1000 = 4650 is right, but keeping 4 decimal places wrote "4650,0000".
  Places are now padded only when the repaired value has a fraction. Regression tests added.
- **Reduce drops repaired facts:** Map now has "4650" signups (clean-en) and "$57,000" budget
  (pdf-slides-en) right, but R7B leaves both lines out of `notes.json`. Reduce selection, not
  the guard; candidate for Phase 6 (or the 4b-2 judge's view of omissions).
- **Short numbers are weak evidence:** any one-digit number is one edit from every other, so a
  segment whose only small source number happens to be near a Map value gets "repaired" (a
  unit fixture turned "20 October" into "0 October" when the transcript was "transcript 0").
  Real transcripts have many small numbers ("one" → 1), so these mostly get flagged. Not seen
  on `quick`; watch on `full`.
- **Non-English separators:** source normalization is English-only, so German "42.500" in a
  transcript is 42.5. Keeping decimal places makes the repair "42.500", which reads right
  either way. Number words with hyphens ("forty-two") aren't normalized by `digits()` (ASR
  writes digits, so `quick` isn't affected). Cross-language items (Phase 6) will show more.
- Slide conflict unchanged (both values are in the sources, as planned).

### Decisions (PLAN §20)
- D20 = default rule (exactly one candidate, one digit edit or factor 10/100/1000).

### Plan changes
- §5.5 guard details (readings, names, suffix/space-grouped only flagged, decimal places,
  `number_repairs`); §9 4b-1b done; §19 `MapFile` v3 field list; §20 D20.

### Next
- Phase 4b-2 (judge) in a new session.

---

## 2026-10-11 · Phase 4b-2 kickoff prompt (no code)

### Plan changes
- §21 kickoff prompt for 4b-2 (judge). Choices it fixes, as recommendations for the session:
  a new `judge.py` model wrapper; calibration labels drafted and reviewed **before** the judge
  runs on them (so the labels aren't anchored on its answers), with planted negatives because
  real outputs are mostly supported; D15 measured on the 3-way label (binary reported too);
  judge answers cached per item so re-scoring stays model-free; judge metrics kept out of the
  gate until D16 (Phase 6); no `--judge claude`.
- §9 4b-2 bullet: labels come from 4b-1b outputs.

### Next
- Phase 4b-2 in a new session with the §21 prompt.

## 2026-10-11 · Phase 4b-2 (judge)

### Built
- `src/cohr/judge.py` (eval only): Qwen3-4B-Instruct-2507-4bit on mlx-lm, temperature 0,
  256 max tokens. Two questions: support (claim + cited text → yes/partial/no + reason) and
  coverage (fact + one section's claims → covered). JSON parsed from the outermost `{…}`; one
  retry with `judge_retry.md` appended (at temperature 0 a plain resend repeats the reply),
  then "unjudged" / `covered: None`. `CachedJudge`: one JSON file per item in
  `evals/data/judge/`, key = question kind + texts + hash of all three judge prompts + judge
  revision; the model is requested only on the first miss.
- Prompts: `judge.md`, `judge_coverage.md`, `judge_retry.md`. The claim is prefixed with its
  section ("Open question: …") and action items carry owner and due, since "an open question
  was raised" can't be judged from the words alone.
- `evals.py`: `cited_text` (seg citation = transcript only, not the Map notes Reduce saw;
  slide = slide text), `judge_item` (support per cited claim, coverage per judge_only fact),
  `judge_items` (releases the pipeline models, loads the judge lazily, releases it after;
  cost row `Report.judge`, outside `stages` so neither the stage total nor `compare()` sees
  it), summary `support_rate`/`support_partial`/`support_no`/`support_unjudged`/
  `judge_recall`, report lines for every non-"yes" claim. Calibration: `Calibration` file
  model, `calibrate`, `agreement` (3-way and yes vs not-yes; unjudged disagrees with every
  label), confusion matrix, `*-judge-calibration.{json,md}` report.
- CLI: `cohr eval --no-judge`, `cohr eval --calibrate-judge`. `ModelManager` has a `judge`
  loader (not in `PIPELINE_MODELS`).
- `mapper.fill` substitutes the placeholder names it's given (was a fixed list), so the judge
  prompts reuse it. No prompt text changed, so no stage hash changed.
- Tests: `tests/test_judge.py` (parsing, retry, prompts, cache key, cache across instances,
  corrupt entry), `tests/test_eval_judge.py` (fake judge: support rate, unjudged in the
  denominator, judge recall, re-scoring makes no call and loads nothing, pipeline models
  released before the judge's first answer, summary and markdown, `compare()` ignores judge
  metrics, agreement, calibrate end to end, the committed calibration file's consistency).
  272 tests.

### Calibration set (`evals/gen/judge_calibration.json`)
30 pairs from the 4b-1b `quick` runs: 12 yes / 8 partial / 10 no. 14 are real claims with their
real citations; 7 move the citation (wrong segment, another meeting's segment, the conflicting
slide); 9 edit the claim (an added detail, a merged second decision, a swapped owner).
Drafted by Claude and reviewed by the user before the judge ran on them. One change from the
review: c11 was a second "Seasonality in measurement" real output (ambiguous, since the
budget's measurement *is* mentioned); it became a merged-decisions partial (Thursday move from
the cited segment + contract designer from the uncited one). Labeling choices, agreed in review:
support = "the cited text says it", not "it's true" (c17, the misspoken $80,000 cited to the
transcript, is yes; c01, the same claim cited to the $18,000 slide, is no); the claim's
section is part of it.

### D15 (`20261011-095636-judge-calibration`)
| | yes | partial | no |
|---|---|---|---|
| label yes (12) | 12 | 0 | 0 |
| label partial (8) | 1 | 3 | 4 |
| label no (10) | 0 | 1 | 9 |

3-way 80.0% (pass, exactly at the threshold), yes vs not-yes 96.7%. 30 calls, 61 s, peak
3.28 GB. The misses: four partials with an *added* detail judged "no" (c07, c11, c15, c28;
each reason says the main point is stated and the detail isn't, i.e. the partial definition);
c10 judged "yes" with a reason that's wrong (it says retention's seasonality is mentioned;
only other topics' is); c27 (the made-up "update dashboards" decision) judged "partial". The
prompt was **not** tuned on these pairs. **D15 = pass, no margin.** Since support rate counts
only "yes", the partial/no confusion doesn't move it; the binary agreement is what it rests on.

### `quick` with the judge (`20261011-095926-quick`, pipeline stages reused from 4b-1b)
| Item | Support rate | partial | no |
|---|---|---|---|
| clean-en | 100% (17/17) | 0 | 0 |
| noisy-en | 85.0% (17/20) | 2 | 1 |
| pdf-slides-en | 82.6% (19/23) | 0 | 4 |
| slide-conflict-en | 94.7% (18/19) | 0 | 1 |
| **all** | **90.0% (72/80)** | 2 | 6 |

Exact metrics unchanged (recall 85.3%, citation accuracy 100%, made up 3). Judge recall: no
judge_only facts in `quick` (cross-language items run in `full`, Phase 6; unit tests only).
Judge cost: 80 calls, 156.5 s = 403 s per audio hour, peak 3.37 GB (pipeline models released
first). A second scoring: 0 calls, 80 cached, no model loaded.

### Findings
- **Open questions are mostly made up.** 5 of the 6 open questions in `quick` are judged not
  supported ("How to improve site speed and cloud costs?", "Support queue seasonality", …):
  Reduce turns the filler line "there's some seasonality in X" into questions nobody asked.
  The exact metrics can't see this (made-up items count only decisions and action items).
  Candidate for Phase 6: a Reduce prompt rule, and counting open questions in made-up items.
- The judge confirms the known errors without being told: 3,360 vs 336 and "November 2nd" vs
  the 22nd are both "no"; the made-up dashboard decision is "partial".
- All the "no" answers on `quick` were checked by hand against the transcripts: all correct.
- Judge cost is ~45% of the pipeline's 904 s per audio hour, eval only; caching makes
  repeated scoring free.
- The calibration pairs are harder than real output (real claims are mostly "yes"), so
  agreement on real eval output is likely higher than 80%.

### Decisions (PLAN §20)
- D15 = pass (3-way 80.0%, binary 96.7%). Judge metrics still stay out of `compare()` until
  their thresholds come with D16 in Phase 6.

### Plan changes
- §6 layout (`judge.py`, judge prompts), §8 and CLAUDE.md wrapper lists (+ `judge`), §9 4b-2
  done, §18 Judge details (section prefix, transcript-only citations, support ≠ truth, retry,
  cache, cost row, don't tune on the calibration pairs), §20 D15.

### Next
- Phase 5 (Streamlit UI) in a new session.

## 2026-10-11 · Phase 5 split and 5a kickoff prompt (no code)

### Plan changes
- §9 Phase 5 split in two PRs, approved by the user: **5a** = theme, job layer
  (`src/cohr/jobs.py`, no Streamlit), `pipeline.run(force=…)`, app shell, New run, Notes and
  Runs pages, `test_theme.py`/`test_app.py`, the width and light/dark checks, D13 and D14;
  **5b** = Ask (`ask.py`, `prompts/ask.md`, the Ask page, "Open in Ask"). Ask is new model work
  (prompt, R7B chat history, answer language, citations in answers), the same reason Phase 4
  was split.
- §21 kickoff prompt for 5a. Choices it fixes: theme and its test first; job logic outside
  Streamlit so it's unit tested without AppTest; progress polled with `st.fragment(run_every=…)`;
  Claude in Chrome only after asking.

### Checked (Streamlit 1.65.0, installed)
Every API §7 uses exists: `st.dialog(position=…)`, `st.pills`, `st.audio(start_time,
end_time)`, `st.container(horizontal, width, height)`, `st.columns(wrap)`, `st.badge`,
`st.navigation`, `st.image(alt)`, `st.fragment(run_every)`, `AppTest`. `st.context` still has
no viewport width (the §7 "never measure the screen" rule holds). `pipeline.run` has no way to
force a stage, so 5a adds `force=`.

### Next
- Phase 5a in a new session with the §21 prompt.

## 2026-10-11 · Phase 5a (UI without Ask)

### Built
- `.streamlit/config.toml` exactly as §7, plus `server.fileWatcherType = "none"` (below).
  `tests/test_theme.py`: light/dark key parity, WCAG contrast (text ≥ 7, link and semantic
  colors ≥ 4.5, white on primary ≥ 4.5, primary vs background ≥ 3), sidebar darker than main
  in both modes, no Google Fonts, no telemetry, localhost only.
- `pipeline.run(force=[…])` / `plan(force=…)`: a forced stage gets reason `forced` and every
  later stage follows (the existing "once one is stale, the rest are too" rule). No hash
  changes, so the next plain run skips everything again. Unknown names raise before anything runs.
- `src/cohr/jobs.py` (no Streamlit): `JobRunner.start(run_dir, force=…)` runs `pipeline.run` in
  a daemon thread; progress events and a final `_Finished` go through one queue, and
  `Job.snapshot()` drains it into an immutable `JobState` (one `StageView` per stage).
  `start` raises `JobBusy` while a job runs; model use is still serialized by the ModelManager
  lock. Every exception ends the job (`failed` + message) so the UI never hangs.
  `recover_all(runs_dir)` runs `recover_interrupted` on every run and skips broken manifests.
- `ModelManager.set_config(config)`: the UI shares one manager, so each job switches it to the
  run's config. It releases models whose pinned spec changed, and everything when the policy
  becomes sequential. `MlxMemory.active()` / `limit()` read `mx` only when it's already imported.
- `storage.stage_upload` / `discard_upload` and `Source(upload=True)`: uploads are streamed in
  1 MB chunks to `runs/.uploads/<random>/`, which is on the same disk as the run, then
  `create_run` moves them into `source/` (manifest `mode="upload"`, `path=""`). The filename
  is reduced to its basename.
- `render` app mode is now Streamlit markdown: `$`, `[`, `]` and `\` are escaped (`$` would
  start LaTeX: "$14,500 … $80,000" became math), uncited claims are `:gray[…] :gray-badge[uncited]`,
  and Decisions / Action items / Open questions get a count badge in green / orange / violet.
  The portable `notes.md` is unchanged (no FORMAT_VERSION bump).
- `src/cohr/ui.py`: cached `ModelManager` and `JobRunner` (`make_models` / `make_runner` are
  swapped by tests), `startup()` (recovery once per process), sidebar (run picker, status
  line, footer), `job_panel` (one `st.status` per started stage with the progress bar,
  "Next: …" for pending ones, Cancel), and pure helpers (run rows, status, memory rows,
  cited times, `audio_clip`).
- `app.py` + `app_pages/new_run.py`, `notes.py`, `runs.py` per §7. The New run form submits
  through an `on_click` callback, because the callback runs before the script and so can set the sidebar's current run
  (`st.session_state[RUN_KEY]`) without the "widget already created" error.
- Tests: `test_theme.py` (7), `test_jobs.py` (11), `test_ui.py` (6), `test_app.py` (15
  AppTest: every page with and without runs, one Sources block per segment, pruned run, Re-reduce
  forces `reduce`, Runs lists the run, New run refuses a missing path and an empty form, a
  local-path run starts a job and becomes the current run, recovery on app start), plus force,
  set_config, upload and app-mode render tests. 321 tests.

### Manual checks (Claude in Chrome, real models, `evals/data/gen/clean-en`)
- New run with a local path + 8 slide paths → run `20261011-103055-meeting`. Stages streamed
  live (Transcribe "chunk 2/12", Describe slides "slide 2/8", Map "segment 2/2").
- Cancel landed in Describe slides (Transcribe had just finished): Ingest, Speech detection and
  Transcribe stayed `done`, Describe slides `cancelled`, the rest never started.
- Runs → Resume: the first three stages `reused`, the rest ran; "Notes are ready". Per-stage
  times on this 6-min meeting: transcribe 11.6 s, vision 32.7 s, map 15.6 s, reduce 31.8 s;
  peaks 5.6 / 8.6 / 11.7 / 16.8 GB (decimal), under the 26.8 GB limit.
- Notes: `$` amounts render as text, badges show, 2 segment blocks with audio, 8 slide blocks.
- **D14 = anchors work**: clicking `[5:41](#seg-1)` scrolled the Sources container from 0 to
  2358 px and put the `4:51–5:51` block at the top.
- **D13 = shrinks** (measured in an iframe of each width, because `resize_window` didn't change
  the maximized window's viewport): 420 → notes 377 px (stacked, no horizontal scroll, sidebar
  hidden on a fresh visit); 800 → 209 / 128 px with the sidebar open, 422 / 271 closed;
  1280 → 420 / 269; 1512 → 559 / 362; 2560 → 880 (cap holds) / 781. No horizontal scroll at
  any width.
- Light and dark: Notes and Runs checked in both; the bar chart uses the theme's categorical
  colors.
- **Audio playback not verified in the browser**: the automation tab is a hidden window
  (`visibilityState = hidden`), and Chrome didn't load media there (readyState 0). The clip
  endpoint was checked from the shell: 200 and 206 range responses, valid 16 kHz mono WAV,
  60.0 s for a 60 s segment. Needs one click in a normal window.

### Findings
- **Streamlit's file watcher tries to import torch.** Its `sys.modules` walk calls
  `hasattr(m, "__path__")` on transformers' lazy modules (pulled in by mlx-vlm / mlx-audio),
  which import `transformers.models.*` and fail on `import torch`: 872 tracebacks in the log
  of one run. It would also reload modules on save and drop the cached JobRunner during a job.
  Fixed with `server.fileWatcherType = "none"` (the app doesn't need hot reload; the theme test
  asserts it).
- `st.audio(path)` reads and hashes the whole file on every call (memory media storage), so
  ~115 MB/h for each segment block on every rerun. Now one cached WAV clip per segment.
- **Medium windows with the sidebar open are cramped**: `st.columns` only stacks at
  ≤ 640 px viewport width, and the sidebar stays open at 800 px, so Sources is 128 px wide.
  Closing the sidebar fixes it. Options for later, all within the no-CSS rule: Notes uses the
  drawer for Sources below some size (needs the width, which Python can't see), or a
  "Sources below notes" toggle on the page. Left open.
- Streamlit persists the sidebar state per page in localStorage (`stSidebarCollapsed-`), which
  overrides `initial_sidebar_state="auto"` on later visits.
- Theme choice from the settings menu is lost on a full URL load (in-app navigation keeps it).
- Memory figures in the UI are decimal GB (`/1e9`) like the evals, so the limit reads 26.8 GB
  as in §4; `storage.GB` (GiB) stays for disk sizes.
- Small: the "Deploy" toolbar button shows in this local-only app (`client.toolbarMode =
  "minimal"` would hide it; not in §7, left as is). Slide 1's title came back empty from
  Vision on clean-en (the Sources heading reads "Slide 1"); not a UI issue.

### Decisions (PLAN §20)
- D13 = the 880 px container shrinks; keep `width=880`. D14 = anchors scroll the Sources panel;
  no drawer fallback.

### Plan changes
- Status line; §6 layout (+ `jobs.py`, `ui.py`); §7 config (`fileWatcherType`), Sources panel
  (anchors, per-segment clips), medium-window finding; §9 5a done; §20 D13, D14.

### Next
- Phase 5b (Ask) in a new session. Before that, decide the medium-window layout (above) and
  check audio playback once in a normal browser window.

## 2026-10-11 · Phase 5b kickoff prompt (no code)

### Plan changes
- §21 kickoff prompt for 5b (Ask); the §9 5b bullet points to it. Choices it fixes:
  - ask.py reuses reduce.py (documents, fit_prompt, citation parsing, segment_refiner) instead
    of copying it. CommandR7B gets a chat variant of prompt() and the old one stays as it is,
    so Reduce's config hash doesn't move (tested).
  - Ask history lives in st.session_state per run: no run file, no schema change.
  - While a job runs, the chat input is disabled with a reason, instead of blocking the
    script thread on the ModelManager lock (a reading of §7's "waits for the lock").
  - One source block in ui.py serves both the Ask drawer and Notes' Sources panel, which also
    prepares the Phase 6 fix for medium windows (5a finding).
  - The manual checks include the audio playback 5a couldn't verify.

### Next
- Phase 5b in a new session with the §21 prompt.

## 2026-10-11 · D21: Notes layout in medium windows (no code)

### Decisions (PLAN §20)
- D21 = a "Sources: Beside / Below" st.segmented_control in the Notes header, default Beside,
  kept in st.session_state. Below puts the source blocks under the notes without the
  fixed-height box, so anchors scroll the page (D14 still holds). The user picks because Python
  can't see the width. Rejected: Sources always in the drawer (undoes D14 and loses the
  side-by-side view on wide screens), a "close the sidebar" hint only, and collapsing the
  sidebar on Notes (localStorage overrides initial_sidebar_state, 5a finding).

### Plan changes
- §7 Notes medium-window bullet; §9 5b bullet; §20 D21; §21 5b prompt (item 4, a test, and a
  manual check at ~800 px).

### Next
- Phase 5b in a new session with the §21 prompt.

## 2026-10-11 · Phase 5b (Ask)

### Built
- `src/cohr/ask.py` (no Streamlit): `ask()` / `run()` answer one question from the run's own
  files with Reduce's pieces: `reduce.documents` and `fit_prompt` (transcripts trimmed when
  the documents alone don't fit), `claims` / `resolve` (one claim per sentence, out-of-range
  indices dropped and counted, uncited sentences kept with `citations: []`) and
  `segment_refiner` (D19 jump times). Returns a frozen `Answer` (claims, language, prompt
  tokens, turns dropped, transcript limit, dropped citations, raw reply). `run()` switches the
  shared ModelManager to the run's config under its lock, uses `reduce`, and restores the
  previous config, so a job that set it keeps it. **`reduce.py` is unchanged** (the planned chat
  variant of `prompt()` was dropped, D22), and the config hash test pins Reduce's hash.
- `prompts/ask.md`, `ask_history.md`, `ask_turn.md`: rules, then history, then the question,
  then "Answer in <language>." as the last line.
- History: the last 3 turns, answers as their claim texts (no `<co>` markup), dropped
  oldest-first until the prompt fits `reduce.max_input_tokens`; then Reduce's trimming.
- Answer language (D23): lingua over R7B's 23 languages; ≥ 2 words or CJK, confidence ≥ 0.2,
  lead ≥ 0.05, else the run's target language. The detector is built once (0.2 s).
- `render` gets a `chat` mode: app-mode escaping and the uncited badge, but plain `[12:47]`
  labels, because the Ask page has no anchors to link to. `render.answer_md`.
- `ui.py`: `segment_block` / `slide_block` / `RunMedia` (one source block for Notes' Sources
  panel and the Ask drawer), `citation_label`, `unique_citations`, cached `ask_sources`.
- `app_pages/ask.py` (centered): chat history per run in `st.session_state`, answers with the
  uncited badge, one `st.pills` row of cited moments per answer; a pill opens
  `st.dialog(position="right")` with that segment's block, its audio starting at the cited `t`
  (or the slide). The pill clears itself so the same source can be reopened. "Clear chat".
- **Ask while a job runs (§7 reading):** the chat input is disabled with a caption instead of
  blocking the script thread on the ModelManager lock; the sidebar status fragment reruns
  the page when the job ends, which re-enables it. The lock still serializes model use if a
  job starts mid-question.
- Notes: "Open in Ask" (switches page, same current run), and the D21 "Sources" segmented
  control (Beside / Below, `persist_state="session"`). Below = notes in the 880 px container,
  then the source blocks without the fixed-height box.
- Tests: `test_ask.py` (19: instruction shape, history oldest-first and last-3, transcript
  trimming after history, too-long error, no markup in history, language fallback cases and
  real lingua, citations parsed/refined, dropped indices, uncited kept, run config applied and
  restored, Reduce hash unchanged); `test_app.py` +6 (Ask renders, answer + pills, pill opens
  the drawer, follow-up history, input disabled during a job, Open in Ask, Notes Below).
  348 tests.

### Manual checks (real R7B, `runs/20261011-103055-meeting` = clean-en)
Six questions in one conversation (`ask.run`, history carried like the page), ~9–10 s each,
~4.3K prompt tokens, peak 5.9 GB (R7B alone):
| Question | Truth | Answer | Citations |
|---|---|---|---|
| How many open tickets are there? | 390, 0:12–0:15, slide 1 | "There are 390 open tickets." | 0:12, slide 1 ✓ |
| What needs to be sent to legal? | Laura, Friday, 0:46–0:50 | "Laura needs to send a contract to legal by Friday." | 0:46 ✓ |
| Who owns that, and by when? (follow-up) | Laura, Friday | "Laura owns the contract to be sent to legal by Friday." | 0:46 ✓ |
| What does Omar need to do? | hiring plan, end of next week, 1:39–1:42 | "Omar needs to draft the hiring plan by the end of next week." | 1:39 ✓ |
| Wie lange dauert das Laden der Seite? | 950 ms, 1:08–1:12, slide 2 | "Ich konnte keine Informationen … finden." (German, but wrong) | uncited |
| What is the marketing budget for next year? | not in the meeting | "The approved budget for next quarter is $42,500." | 1:48, slide 3 |

### Findings
- **R7B's multi-turn `rag` format doesn't work for follow-ups (D22).** The template puts the
  documents after the first user turn only; on later turns R7B wrote a new tool call
  (`{"tool_call_id": "1", "tool_name": "direct-injected-document"}`) instead of an answer
  (4 of 6). Prefilling `<|START_RESPONSE|>` made it answer, but follow-ups lost their
  citations. Folding history into one user turn (Reduce's single-turn shape) fixed both.
- With the language line first, a German question got an English answer; last line fixes it
  (the Map lesson, §5.5).
- **Cross-language after history is fragile.** The German question alone: "Das Laden der
  Seite dauert 950 ms." [1:07, slide 2]; after two English turns (legal, Omar): correct; after
  one turn (Omar): R7B repeated "Omar needs to draft the hiring plan…"; after legal / who owns /
  Omar: "no information". Three history layouts tried (labels, context first): none fixed all
  cases without breaking another, so the simplest stays. German questions without history
  were all right (tickets 390, Laura → legal by Friday, 950 ms).
- **Near-miss questions get a related fact.** "Marketing budget for next year" → next
  quarter's $42,500 (cited) under every rule wording tried. A clean miss ("Who was hired as
  the new head of sales?") is answered "the meeting doesn't mention…" with the rule now in
  `ask.md`; without it, R7B invented "Omar was hired as the new head of sales."
- Citations on German answers to English documents can't be refined (no token overlap), so
  they jump to the segment start (0:00 instead of 1:07 once); slide citations still land.
- lingua on single words is unreliable ("Priya?" → Indonesian 0.56, "Q3 ARR?" → Spanish
  0.24), hence the 2-word minimum (D23).
- `st.switch_page` inside an `on_click` callback raises "Could not find page": callbacks run
  before `st.navigation` registers the pages. Called from the script instead.
- AppTest renders `st.dialog` content into the main tree, so the drawer is testable.

### Decisions (PLAN §20)
- D22 = history folded into one user turn, language line last. D23 = lingua with ≥ 2 words
  (or CJK), confidence ≥ 0.2, lead ≥ 0.05.

### Plan changes
- Status line; §5.7 (prompt shape, language rule, known limits); §6 prompts; §9 5b; §20 D22, D23.

### Browser checks (user, normal Chrome window, real models)
- Audio plays in the Notes Sources panel, starting at the earliest cited moment (5a couldn't
  verify playback).
- Open in Ask switches to Ask with the same run; "How many open tickets are there?" → 390 with
  pills `0:12` and `slide 1`; the `0:12` pill opens the right-side drawer and its audio starts
  at ~0:12; the same pill reopens it after closing. The follow-up (legal → "Who owns that, and
  by when?") answers Laura, Friday, cited 0:46.
- At ~800 px with the sidebar open, Notes in Below mode: clicking a citation scrolls the page
  to its block; Below stays selected after going to Ask and back.

### Next
- Phase 6. Candidates from 5b: an Ask eval on the planted facts (answer accuracy, citation
  jump, "not stated" on near-miss questions, cross-language follow-ups), and FTS5 ranking.

## 2026-10-11 · Phase 6 split and 6a kickoff prompt (no code)

### Plan changes
- §9 Phase 6 split in five PRs: 6a baseline + Ask eval (measurement only), 6b quality fixes,
  6c long meetings (two-pass reduce, `long-en`, Ask FTS5 ranking), 6d languages ("Other
  (beta)", cross-language items, D18), 6e `full` set + v1 (FLEURS/AMI, error messages,
  sequential check, README cache note, release gate, `1.0.0` only when asked).
- Why measurement first: 5b tuned the Ask prompt by eye on one fixture and stopped with
  known limits; from 6b on, every change is compared with a committed baseline and D16
  thresholds.
- D16 method made precise (§20): same media, pipeline seeds 1–3 via `-c seed=N`. `--seed`
  changes the generator seed (new meetings), and with a fixed pipeline seed the outputs are
  deterministic, so repeating a run would show zero noise.
- §21 kickoff prompt for 6a. Prompts for 6b–6e are written after each sub-phase, as for 4b and 5b.

### Next
- Phase 6a in a new session with the §21 prompt.

## 2026-10-11 · Phase 6a (baseline + Ask eval)

### Ask eval rules (recorded before the first run)
- **Questions** (`evals.ask_questions`, templates from `truth.json`, no LLM), per item: one per
  number fact ("What is <label>?", label words from `NUMBER_LABELS`), one per action ("What
  does <owner> need to do?"), a follow-up after the first action ("Who owns that, and by
  when?"), two German number questions ("Wie hoch ist <label>?": the first number alone; the
  second after the English action + follow-up turns) and one near-miss (the last number's
  label with "two years ago", "for the marketing team" or "for next year", picked by the item
  id; the generator only ever says "last quarter/month/week/release"). History = the earlier
  answers as `Answer.turn()`, as the Ask page keeps it. 17 questions per quick item.
- **Correct** = the answer states the fact and a claim stating it cites the fact's segment
  within ±15 s (D19) or its slide. Numbers: the value in a claim (German "42.500 $" read as
  42500, eval-only). Actions: `fact_fit` as in fact recall (owner exact, task overlap ≥ 0.6).
  Follow-up: the owner and the planted due date both in the answer.
- **Near-miss rule:** planted values = every number in the script or slide cells above 31
  (1–31 are dates and "week two"). Going through the answer's claims in order: a claim with a
  "not given" marker (not, no, n't, never, none, unknown, cannot; German kein/nicht) passes the
  answer; a claim with a planted value before any such marker fails it; no planted value at all
  passes. Why not "no planted value anywhere": `ask.md` asks R7B to say the meeting doesn't give
  it, *then* what it does give, so a related fact after the negation is the wanted answer.
  Known blind spot: "It's $42,500 for next quarter, not next year." passes (marker and value in
  one claim).
- **Metrics:** Ask accuracy (correct / English number + action questions), Ask stated, Ask
  citation accuracy (cited right / stated), follow-up, German alone, German after history,
  near-miss (one each per item, so 4 per set: 25-point steps).
- Answers cached in `evals/data/ask/<item>.json`, keyed by question, history, the ask prompt
  hashes, the R7B revision, a hash of the run's documents and the seed / target language /
  Reduce config.

### Built
- `evals.py` Ask eval: `ask_questions` (templates above), `score_answer` / `ask_metrics`,
  `answer_numbers` (German thousands separators, eval-only; `numbers.py` unchanged),
  `due_in`, `meeting_numbers`, `near_miss_ok`; `CachedAsk` + `ask_session` (misses go through
  `ask.run` on the shared ModelManager while R7B is still resident from Reduce, before the judge
  releases the pipeline models); `ask_item` adds `ask` results and metrics to each item.
  Report: Ask metrics in the summary, an `ask_accuracy` column, every wrong answer listed per
  item, Ask cost line. `cohr eval --no-ask`.
- D16: `MAX_DROP` / `MAX_RISE` tables drive `compare()`; `evals.noise(reports)` (max − min per
  summary metric, slowest/fastest − 1 per stage).
- Tests: `test_eval_ask.py` (13: templates deterministic with every kind, labels cover the
  generator, right fact + citation, wrong owner, follow-up, German numbers, near-miss pass/fail,
  metrics, cache key incl. prompt hash, end to end through `ask.run` with a rule-based R7B,
  cache hit with no model, seed change = miss, report); `test_evals.py` +2 (noise, gate tables).
  363 tests.

### D16 runs (quick, same media, `-c seed=1|2|3`; ~15 min each incl. 68 Ask questions ~2.5 min/item and the judge)
| Metric | seed 1 | seed 2 | seed 3 | noise | gate |
|---|---|---|---|---|---|
| fact recall | 83.8% | 85.3% | 86.8% | 2.9 | −5.9 |
| citation accuracy | 100% | 100% | 100% | 0 | −3 |
| made-up items | 4 | 3 | 3 | 1 | +2 |
| uncited / gap words | 0 / 0 | 0 / 0 | 0 / 0 | 0 | no rise |
| recall decision / action | 81 / 70% | 88 / 70% | 81 / 80% | 6 / 10 | report |
| support rate (judge) | 79.3% | 86.5% | 88.4% | 9.1 | report |
| Ask accuracy | 98.1% | 98.1% | 98.1% | 0 | −5 |
| Ask citation accuracy | 100% | 100% | 100% | 0 | −3 |
| follow-up | 3/4 | 4/4 | 4/4 | 25 | report |
| German alone / after history | 4/4, 3/4 | 4/4, 4/4 | 4/4, 4/4 | 0 / 25 | report |
| near-miss | 0/4 | 0/4 | 0/4 | 0 | report |
| reduce s/audio h | 274 | 293 | 303 | 10.5% | +25% |
Ingest, VAD and Transcribe don't read `seed`, so they were reused (no noise measured: 25%
default kept). Seeds 2 and 3 pass `compare()` against the seed-1 baseline.

### Findings
- **Near-miss is 0 of 12** over the seeds: R7B answers the off-period/off-team question with a
  meeting number every time, despite the `ask.md` rule ("The approved budget two years ago was
  $54,000": that's last quarter's value; "MRR for the marketing team is $248,000"). The 5b
  limit, now measured. 6b.
- Ask on English facts is near perfect (51 of 52 per seed, all cited right). The one miss is
  `slide-conflict-en`'s budget: Ask gives the misspoken $80,000, not the slide's $18,000, against
  the `ask.md` slide rule.
- Cross-language after history: 1 of 12 wrong (seed 1), and it stated the right value but cited
  the segment start (German answer, no token overlap for refinement, as in 5b). German alone:
  12 of 12. The 5b fragility shows much less on the eval's history layout (action + follow-up).
- Follow-up seed 1: R7B answered a different task's owner/date ("book the venue … by 3
  October" instead of the contract by November 18).
- Open questions (all filler on quick, nothing planted): 13 / 11 / 4 claims over the seeds:
  too noisy to count as made up (kickoff item 3), left to 6b's fix.
- Ask answers are deterministic per seed (98.1% on all three) while Reduce output varies;
  most Reduce noise is in recall of decisions/actions, not numbers.

### Decisions (PLAN §20)
- D16 as in the table; report-only choices confirmed by the user.

### Plan changes
- Status line; §9 6a done; §18 Ask row and gate thresholds; §20 D16.

### Next
- 6b kickoff prompt: near-miss (0/4), slide-conflict in Ask, filler → open questions, Reduce
  dropping facts Map has right; each change compared with `evals/baseline.json`.

## 2026-10-11 · Phase 6b kickoff prompt (no code)

### Plan changes
- §21 kickoff prompt for 6b; §9 6b bullet points to it.
- Measurement first: quick plants no open questions, so a Reduce fix that drops every open
  question would score perfectly on "made-up open questions". 6b first adds planted open
  questions to the generator (new media → re-baseline before any fix) and asks all three
  near-miss templates per item (12 per set, not 4).
- Correction to the 6a entry: on slide-conflict-en the **notes** also use the misspoken
  $80,000 (baseline `conflict`: slide_value_used false, misspoken_value_used true), not only
  Ask. So the cause is likely upstream (Map or the documents), and 6b investigates it with
  the dropped-facts issue before changing a prompt.

### Next
- Phase 6b in a new session with the §21 prompt.

## 2026-10-11 · Phase 6b (quality fixes)

Every eval with `-c seed=1` against `evals/baseline.json`; each baseline update is its own
commit with the reason and before/after numbers.

### 1. Measurement first
- **Planted open questions** (`evals/gen`): 1–2 per item from `OPEN_QUESTIONS` ("One thing we
  haven't settled: should we …?" and two other templates), on their own RNG
  (`Random(f"{seed}-open-questions")`) so decisions, actions and numbers stay as they were
  (test). No numbers and no content word shared with a decision or task (test). Quick plants 5.
  Scored as `recall_open_question` (Open questions section, overlap ≥ 0.6) and
  `made_up_open_questions` (claims there matching no planted one); kept out of fact recall and
  citation accuracy (`RECALL_KINDS`), judge-only cross-language like decisions.
- **Near-miss:** every `NEAR_MISS` template per item, each on another number from the last (12
  per quick set, ids `near-miss-1..3`).
- Re-baselined ("generator: planted open questions"): open-question recall 5/5, made-up open
  questions 5 (all filler: "how should we handle seasonality …"), near-miss 0/12.
- Gate (user): both enter the gate after the fixes, with thresholds from 3 seeds on the final
  config (D24).

### 2. Ask: near-miss (ask.md)
- Kept: "each number belongs to the period and team the documents give for it; another period
  or team → the meeting doesn't have that number: say so, then the number with its own
  period", plus one off-domain worked example (Android installs in 2023). **Near-miss 0/12 →
  7/12**; Ask accuracy 98.1%, citation 100%, follow-up 75%, German alone / after history
  100% / 100%, all unchanged. German answers checked by eye: right values, in German (two keep
  English thousands separators, as before).
- Tried and dropped: a fixed opener ("start with 'The meeting doesn't give'") plus a check of
  period, team, person and budget. Near-miss went up, but R7B then refused "What does Elena need
  to do?" because the transcript says "Eleanor" (ASR), and kept refusing with a name rule, the
  rule as its own bullet and a contrasting name example (5 versions). Rules that name a refusal
  phrase prime refusals in R7B.
- Still failing: team questions ("candidates in the pipeline for the marketing team is 75"),
  and "two years ago" answered with last quarter's value *labelled as last quarter* (the
  scorer counts it: no "not given" marker before the number).

### 3. Filler → open questions (map.md, not reduce.md)
- The made-up questions are made by **Map** (`map.json` open_questions); Reduce copies Map's
  list word for word. Two reduce.md rules ("only questions someone asked and left open"; "the
  part notes' lists were written by another model, keep one only with a transcript sentence")
  changed nothing: same 5, same text.
- Moved to map.md (user): `open_questions` = "questions a participant asked and left
  unresolved in this part; a remark about a topic or a trend is not one". **Made-up open
  questions 5 → 0**, open-question recall 5/5 → 4/5 (missed "One thing we haven't settled
  should we run the user conference online?", ASR dropped the colon; "or said are not settled
  yet" didn't bring it back), fact recall 82.3 → 83.8%, made-up items 6 → 4, support rate
  88.2 → 93.2%. Reduce's config hash didn't move (test_ask pin unchanged).

### 4. Causes (no fix)
- **a. "Reduce dropping facts Map has right": not Reduce.** Fit: ~4.3K prompt tokens of 32K,
  nothing trimmed. On the 6b baseline every missed fact is upstream:
  - **Map drops facts on long segments.** Fuse gives a ~280 s first segment (7 topics) and a
    ~70 s second one; Map's notes for segment 0 list ~7 key points and drop the rest (noisy-en:
    MRR $248,000 and the venue action; slide-conflict-en: two actions and a decision;
    pdf-slides-en: the Maria action, ASR right). D12 already found 120 s windows worse; a
    real fix is Map coverage on dense segments (e.g. a per-topic list or a second pass), not
    a window tweak. For 6c (two-pass work) or later.
  - **Map misreads a due date:** "by the 22nd of November" → "November 2nd" (the number guard
    covers numbers, not dues).
  - **ASR: "Elena" → "Eleanor"** (say + ASR round trip: 5 of 9). Replaced by **Maria** (9 of 9)
    in `OWNERS` (user), re-baselined.
- **b. Slide conflict:** the slide document reaching R7B has the right **$18,000**; Map's
  segment-0 note says "$80,000 (80,000)" from the misspoken transcript although that slide was
  in its input. R7B then sees $80,000 twice (Map note + transcript) against $18,000 once. Not a
  clear fix: Map would need to link a slide row ("Approved budget") to the spoken value.
- Side finding: slide 1 on slide-conflict-en is the vision fallback (unparseable JSON kept as raw
  text, by design); its values are still right.

### Baseline after the Maria swap
Regenerated media (every line re-synthesized) is a new sample: against the previous baseline it
**failed citation accuracy 98.2 → 92.9%**. Cause: on the new clean-en, Reduce wrote run-on key
points bundling 3–4 facts per claim ("Support Queue: 390 open tickets … New logo … Laura to
send contract …"), each citing one point, so 3 number facts land 19–35 s off. Re-baselined
with the failure stated in the commit.

### D24 runs (quick, final config, `-c seed=1|2|3`, Ask, no judge for 2/3)
| Metric | seed 1 | seed 2 | seed 3 | noise |
|---|---|---|---|---|
| fact recall | 82.3% | 85.3% | 85.3% | 2.9 |
| citation accuracy | 92.9% | 96.6% | 100% | **7.1** |
| made-up items | 2 | 1 | 1 | 1 |
| made-up open questions | 0 | 1 | 0 | 1 |
| open-question recall | 4/5 | 5/5 | 4/5 | 20 |
| Ask accuracy | 98.1% | 96.2% | 98.1% | 1.9 |
| near-miss | 7/12 | 7/12 | 6/12 | 8.3 |
| reduce / vision s/audio h | | | | 21% / 21% |
Seeds 2 and 3 pass against the seed-1 baseline.

### Decisions (PLAN §20)
- D24: near-miss −16.7 points, made-up open questions +2. Citation accuracy noise (7.1) is now
  above its D16 gate (3); left at 3 for now (seed 1 is the low point, so later runs tend to
  pass), to revisit.

### Plan changes
- Status line; §9 6b done; §18 open-questions row, Ask row (3 near-misses), gate text; §20 D24;
  evals/README metrics.

### Next
- 6c kickoff prompt (long meetings). Carry over: Map coverage on long/dense segments (4a), the
  slide-conflict value in Map (4b), Reduce bundling several facts into one claim (citation
  noise), and whether citation accuracy's gate should follow its new noise.

## 2026-10-11 · Phase 6c kickoff prompt (no code)

### Plan changes
- §21 kickoff prompt for 6c; §9 6c bullet updated and points to it.
- Carry-overs from 6b, split (user): Reduce bundling several facts into one claim and the
  citation-accuracy gate go first in 6c (step 0, D25), because citation accuracy is how two-pass
  is judged and its noise (7.1 points) is above its gate (3). Map coverage on dense segments,
  the due-date misread and the slide-conflict value go to a new **6b-2: Map fixes** after 6c.
- New decisions D25 (citation gate after the bundling fix), D26 (two-pass grouping), D27 (Ask
  ranking under the cap); D17 (FTS5 in uv's Python) moves from Phase 7 to 6c, since Ask ranking
  needs it.
- Fast loop: a lowered `reduce.max_input_tokens` forces two-pass Reduce and Ask ranking on the
  5-minute quick items; long-en (~2.5 h, audio + PDF, own `long` set, ~40 min of pipeline per
  run) only confirms, at most three runs, report-only baseline (evals/baseline-long.json).
- The generator's word lists can't give unique facts over 2.5 h; long-en widens them while
  quick scripts stay identical (any evals/gen edit regenerates quick media).

### Next
- Phase 6c in a new session with the §21 prompt.

## 2026-10-12 · Phase 6c (long meetings)

quick evals with `-c seed=1` against `evals/baseline.json`; long-en against
`evals/baseline-long.json` (report-only). Each baseline update is its own commit.

### 0. Reduce bundling and the citation gate (D25)
- `reduce.md`: "Write one fact per item, in one short sentence. Put decisions and tasks only
  under DECISIONS and ACTION_ITEMS, not inside key points." **Recall 82.3 → 89.7%, citation
  92.9 → 96.7%**, support 94.0 → 95.5%, Ask unchanged. Re-baselined.
- The gain is decisions and actions as their own cited items. Key points stay bundled: they are
  **Map's** key points copied word for word ("Support Queue: 390 open tickets … Laura to send
  contract to legal by Friday."). Tried and dropped: an off-domain one-fact example plus "no
  remarks like 'trend as expected'" (recall 83.8%); "split a part-note line that joins several
  facts" (identical notes). → 6b-2.
- D25 (seeds 1–3): citation 96.7 / 96.6 / 100%, noise 3.4 → **gate −6.9** (user). Recall
  89.7 / 85.3 / 83.8%, noise 5.9 (was 2.9); gate kept at −5.9 (user: seed 1 is the high point).

### 1. Two-pass reduce (§4, D26)
- Over the cap: pass 1 per group of consecutive segments (+ aligned slides) → group summary
  with `[seg-N]` / `[slide-N]` after every claim; pass 2 over summaries + unaligned slides.
  A pass-2 citation of a group resolves to markers in the span, else the best-matching pass-1
  claim (overlap ≥ 0.5), else every segment of the group; a match to an uncited pass-1 claim
  stays uncited. `notes.json` v4 keeps the groups and the resolution counts; the eval reports
  them (`pass2_*`).
- R7B rarely copies markers into its spans (38 of 108 with the first pass-2 prompt, 0 with the
  final one); the claim match carries resolution. Group fallbacks: 0 on quick, 5 of 139 on
  long-en.
- Under the cap nothing changes (same notes and metrics as the baseline).

### 2. Ask ranking (§5.7, D17, D27)
- D17: SQLite 3.50.4 in uv's Python has FTS5 and `trigram` (no new dependency).
- Over the cap: FTS5 BM25 over segment + slide documents for the question plus the last turn,
  top 12 that fit with the history, sent in meeting order. Replaces transcript trimming in Ask.
- quick at cap 4000 vs no cap / vs the old trimming at 4000: accuracy 96.2 / 98.1 / 98.1%,
  citation 98.0 / 100 / 100%, follow-up 3/4 / 3/4 / 1/4, near-miss 4/12 / 7/12 / 5/12.
  Fitting the cap costs near-misses either way; ranking keeps follow-ups and scales.

### 3. long-en (§18)
- `long` set: 5 regions × 8 topics (region in the spoken metric, fact label and slide title),
  36 discussion lines per topic, 88 facts, 40 slides, audio + PDF; ~2.2 h. Realistic density
  (user): at quick's density 2.5 h would be ~460 facts. Extra word lists are used only above
  quick's sizes (quick scripts pinned by hash). Ask asks 24 questions (12 number, 6 action,
  follow-up, 2 German, 3 near-miss); action questions name the due date when owners repeat.
- Run 1: **VAD failed** ("[metal::malloc] Resource limit (499000) exceeded"): mlx-audio's
  Silero keeps one MLX array per 512-sample window. Now per 10-minute block.
- Run 2 (32K cap): two-pass citations all resolved, but **recall 11.3%**: pass 1 on 2 groups
  of ~23K tokens spent its 2048 output tokens on ~66 key points or a citation-index loop
  (`0:[1,16,20,…,533]`), so DECISIONS/ACTION_ITEMS were never written. Ask 61% at ~100 s per
  answer (ranking filled the cap with ~60 documents; same citation loops).
- Fixes, checked with Reduce-only reruns on run 2's media: `reduce_part.md` + `reduce_pass2.md`
  list decisions/actions/open questions before figures-only key points, 4096 output tokens
  (recall 37.5%); groups at 40% of the cap (53.8%, citation 97.7%). Then: my new filler lines
  ("I'd like us to keep an eye on …") had become 36 Map decisions (made-up 54) → fact-free
  filler; Ask top 12; `say` timeout + retry (one call hung for 40 min).
- Run 3 (final, baseline-long.json): **recall 67.5%** (numbers 90, decisions 50, actions 40,
  open questions 75%), **citation 98.1%**, uncited 0, made-up 9 / 2; Ask 83.3%, citation 93.8%,
  ~32 s per answer; 467 s per audio hour, peak 17.3 GB.
- Where long-en loses decisions: Map 19/20 → pass 1 15 → pass 2 10. Pass 2 finishes (not
  truncated) but drops or rewords about a third.

### Final gate (quick, seed 1, Ask + judge)
**Pass, identical to the baseline** after all of 6c (regenerated media, new prompt list):
recall 89.7%, citation 96.7%, uncited 0, made-up 2 / 0, support 95.5%, Ask 98.1%, citation
100%, follow-up 3/4, German 4/4 / 4/4, near-miss 7/12.

### Decisions (PLAN §20)
- D17 available; D25 citation −6.9; D26 groups at 40% of the cap with the long-meeting
  prompts and 4096 tokens; D27 FTS5 top 12.

### Plan changes
- Status line; §4, §5.6, §5.7 (two-pass, ranking); §9 6c done with carry-overs; §18 long-en
  row, `long` set, D25 gate, per-set baselines; §19 notes.json v4; §20 D17, D25–D27;
  evals/README.

### Next
- 6b-2 (Map fixes) kickoff prompt. Carry-overs: Map's bundled key points; two-pass pass 2
  losing decisions on long-en; near-miss under the cap; German after history on long-en.

## 2026-10-12 · Phase 6b-2 kickoff prompt (no code)

### Plan changes
- §21 kickoff prompt for 6b-2 (Map fixes); §9 6b-2 bullet updated and points to it.
- Measurement first (step 0): each planted fact's path transcript → Map → notes, Map key points
  per segment, and whether Map replies hit `map.max_tokens` (1024). It decides the order of
  the fixes.
- 6b-2 takes 6c's Map carry-over (bundled key points that Reduce copies, step 1) before the 6b
  items (dense segments, due-date misread, slide-conflict value). Value-changing guards need
  the user's approval first, like D20.
- long-en (step 5): Map changes change it. One full run checks whether two-pass pass 2 still
  drops about a third of pass 1's decisions; Reduce-only reruns fix it if so; at most two long
  runs; report-only re-baseline.
- Not in 6b-2: Ask near-miss under the cap (4/12 vs 7/12) and long-en German-after-history,
  left for 6e's long-meeting check.

### Next
- Phase 6b-2 in a new session with the §21 prompt.

## 2026-10-12 · Phase 6b-2 (Map fixes)

quick evals with `-c seed=1` against `evals/baseline.json`; each kept Map change gets its own
baseline commit.

### 0. Measurement (no fixes)
On the 6c final-gate runs (seed 1, report `20261012-113844-quick`), each planted fact traced
transcript → Map (segment containing `t0`, which list, how many facts the Map string holds) →
notes.json. Map replies re-generated with the same prompt and seed to count tokens (identical
output; key points differ only where the guard repaired a number).

**Map replies are not truncated.** Out tokens per segment: clean 595 / 280, noisy 395 / 374,
pdf 481 / 311, slide-conflict 413 / 273, all ending in `<|END_RESPONSE|>` (max_tokens 1024).
No retries, no raw segments. Map stops on its own on the dense segments: step 2 is a prompt
problem, not `map.max_tokens`.

| item | seg | s | facts in seg | Map key points (facts each) | dec | act | oq |
|---|---|---|---|---|---|---|---|
| clean-en | 0 | 291 | 13 | 7 (3, 1, 1, 2, 1, 2, 1) | 5 | 3 | 0 |
| clean-en | 1 | 66 | 5 | 5 (1, 1, 0, 1, 1) | 2 | 1 | 1 |
| noisy-en | 0 | 275 | 13 | 7 (1 each) | 3 | 1 | 1 |
| noisy-en | 1 | 84 | 5 | 7 (0, 0, 0, 1, 1, 1, 1) | 2 | 3 | 0 |
| pdf-slides-en | 0 | 271 | 14 | 6 (1 each) | 4 | 3 | 1 |
| pdf-slides-en | 1 | 85 | 4 | 5 (1, 1, 0, 0, 1) | 2 | 1 | 0 |
| slide-conflict-en | 0 | 285 | 15 | 7 (1 each) | 2 | 2 | 0 |
| slide-conflict-en | 1 | 64 | 4 | 5 (1, 1, 0, 1, 1) | 1 | 0 | 1 |

- **ASR is right for every planted fact** (the slide-conflict budget is misspoken by design).
- **Map has 58/68 recall facts in their home list; notes have 61/68** (89.7%, the baseline).
  Reduce recovers 3 from the transcript (slide-conflict: Nadia and Grace actions in seg 0,
  Omar in seg 1, which Map's action list left out). **Every notes miss is a Map miss**:
  | cause | facts |
  |---|---|
  | dense seg 0: dropped | clean f5 (Omar: hiring plan), noisy f7 (Victor: venue), slide-conflict f2 (decision: sync to Thursdays) |
  | wrong list | pdf f14 (Maria: security audit, under decisions as "Maria will review…") |
  | short seg 1: number dropped | noisy f16 (MRR $248,000: "Revenue has shown more stability…") |
  | due misread | pdf f16 ("22nd of November" → "November 2nd"; guard flags `2nd`; scored wrong + made up) |
  | slide conflict | slide-conflict f7 ($80,000 spoken, $18,000 on the slide) |
  Planted open question dropped by Map: slide-conflict f4 (ASR drops the colon, as in 6b).
- **Bundling is one segment:** clean-en seg 0's key points join a number with the decision or
  task said next to it ("Support Queue: 390 … New logo on website approved. Laura to send
  contract to legal by Friday.", 3 facts; cloud + Victor; retention + Maria). The same segment
  repeats its 3 actions under decisions. The other 7 segments write one fact per key point. On
  seed 1 the bundles cost no citation (f1, f7, f10 within ±15 s); 6c saw them cost 3.
- **Citation misses (2): one is the number guard.** slide-conflict f11 (page load 740 ms)
  matches "Quarterly budget approved at $80,000, compared to $740 last quarter.": Map wrote
  "$770,000" for the spoken $77,500; its decimal-comma reading 770.0 is one digit from 740
  (page load), the only candidate, so the guard "repaired" it to a wrong number. The other
  (clean f13, 75 candidates) cites seg 0 163 s early: a single-fact key point, Reduce's citation.

**Order:** 1 → 2 → 3 → 4 as planned. Step 1 (one fact per key point; decisions and tasks only
in their lists) also targets the wrong-list case (pdf f14) and the duplicated actions. Step 2:
prompt (list every decision and task said in this part), aimed at the 3 dense drops; the
short-segment number drop (noisy f16) says it's coverage, not only length. Step 3 adds the
guard's decimal-comma mis-repair: in "$770,000" a 3-digit group after a comma is a thousands
group, so the 770.0 reading shouldn't make a repair candidate (to confirm with the user: it
changes which numbers the guard repairs).

### 1. Bundled key points (map.md, kept)
- `map.md` gets a `key_points` rule (it had none): "one fact per item, in one short sentence: a
  figure with what it measures, or one other point that was made. Never join two facts in one
  item. Decisions and tasks go only in "decisions" and "action_items", never in a key point",
  and under decisions: "A task someone will do goes only in "action_items", not here."
- Seeds 1–3 (no Ask, no judge): recall 83.8 / 85.3 / 86.8% (was 89.7 / 85.3 / 83.8, mean
  86.3 → 85.3, inside the 5.9 noise); **citation 100 / 100 / 100%** (was 96.7 / 96.6 / 100).
  No key point holds more than one fact (seeds 1 and 3 traced).
- Seed 1 is a reshuffle, not a pure loss: Map now has Omar's hiring plan (clean) and Maria's
  audit under action items (pdf), but lost the owner of Maria's survey (clean), wrote "6,900"
  for 69 candidates (pdf, flagged), replaced 340 tickets with "Support queue has been more
  stable…" (pdf), dropped "keep the onboarding checklist" (noisy), and Reduce no longer
  recovers two slide-conflict actions. Coverage of seg 0 is step 2's job.
- Re-baselined (seed 1, Ask + judge, gate pass): recall 83.8%, citation 100%, made-up 1 / 0,
  uncited 0, support 95.5 → 91.5% (report-only; check in step 2), Ask 98.1 → 96.2%, Ask
  citation 100%, near-miss 7/12.

### Decision for step 3 (user)
- Number-guard mis-repair ("$770,000" → 740 via the decimal-comma reading): **option 3**. When
  looking for a repair candidate, read separators the way the target language writes numbers
  (en: "," groups thousands, "." decimal; decimal-comma languages the reverse; languages not in
  the table keep both readings). Keeping a number is unchanged (any reading in the sources).
  Unit test: "$770,000" with sources {77500, 740} → flagged. No MapFile change. Rejected:
  never repairing two-reading numbers (loses "$425,000" → $42,500, the D20 case); "comma + 3
  digits = thousands" (one-sided: German "770.000" keeps the bug).

### 2. Dense segments (2026-10-13, nothing kept)
Seeds 1–3, no Ask, no judge, each traced; reference is step 1 (recall 83.8 / 85.3 / 86.8%,
mean 85.3).

| try | recall s1 / s2 / s3 | mean | result |
|---|---|---|---|
| A. map.md: "Cover the whole part … list every figure, every decision and every task … A long part needs longer lists, not shorter items." | 85.3 / 86.8 / 82.3 | 84.8 | no gain; made-up open questions 1–2 on s2–s3 (was 0); reverted |
| B. map.md: JSON keys reordered, decisions / action_items / open_questions before key_points | 80.9 / 72.1 / (stopped) | — | figures drop instead (number recall 78 / 69%); reverted |
| C. `fuse.window_s=180` (D12 re-check against D16 noise) | 89.7 / 85.3 / 83.8 | 86.3 | +1.0, inside the 5.9 noise; kept 300 s |

- **Seg 0 isn't capped at ~7 key points: it holds 6–7 figures, and Map keeps all of them.**
  The loss is decisions and tasks: the same six drop on every seed under the step-1 prompt and
  try A (clean f2; noisy f3, f7; slide-conflict f2, f5, f6). ASR and Map's input are clean
  ("We decided to move the weekly sync to Thursdays."); most sit a few seconds from another
  decision, task or open question (clean f2@42 + f3@46, slide-conflict f4@94 + f5@98 + f6@106).
  A Tiny Aya coverage limit on a 285 s, 7-topic part, not a token or prompt-wording problem.
- B shows list order moves the loss, it doesn't remove it: whatever comes last gets short.
- C: with 150–175 s segments the persistent drops mostly come back (slide-conflict f5/f6,
  noisy f7 on 2 of 3 seeds), but others appear (pdf f7, f11, f14; noisy f15), and on s3
  Reduce loses four noisy-en tasks Map had (Map repeats them under decisions too). 11 segments
  instead of 8 on quick (Map ~35% slower per audio hour). D12 stays 300 s.
- Not tried: a second Map pass for decisions and tasks only (last resort; new prompt, merge,
  MapFile bump, ~2× Map time). Revisit only if long-en shows the same drops hurting notes.

### 3. Number guard: separators and days (kept, two baselines)
- **Option 3 (separators):** repair candidates use only the readings that follow the target
  language's decimal mark (`numbers.DECIMAL_MARK`, `native_readings`). Beyond the decision: a
  number no native reading fits ("3,36" in English) falls back to every reading, else the
  correct malformed repairs (3,36 → 336, 4,26 → 426, 7,40 → 740 on seed 1) would become flags.
  Spanish (written both ways) and ar/fa (own separators) aren't in the table. First version
  dropped every reading with a "," for English, thousands group included, and the fallback
  hid it; the "$770,000" test caught it. Seed 1 `--fresh`: recall 83.8% (=), repairs 9 → 8,
  unverified 2 → 3, unsupported numbers 2 → 1 ($740 gone from the notes). Baseline 39c67a1.
- **Due misread, prompt first (reverted):** a `due` rule "Keep every digit of a day and the
  order it was said in: "the 23rd of June" stays "the 23rd of June", never "June 3rd" or
  "June 23"". Map still wrote `2nd` on seeds 1–3 (seed 1 kept the order: "2nd of November");
  recall 88.2 / 79.4 / 89.7%. A copy error inside "22nd", out of the prompt's reach.
- **Guard for days:** a Map ordinal ("2nd") is compared only with the ordinals in the sources
  (`numbers.ordinals_in`: "22nd", "the twenty-second"; `mapper.source_days`). Against every
  source number 2 was one edit from 22, 12 and 1 (flagged); among the days said only 22 is.
  Plain numbers are unchanged. No MapFile change (`number_repairs` already records it).
  Seed 1 `--fresh`, Ask + judge: recall 83.8 → 85.3%, made-up items 1 → 0, wrong owner/due
  1 → 0, citation 100% (=), support 92.7 → 93.9%, Ask 96.2% (=), near-miss 7/12 (=).
  Baseline e59ff20. Only ordinals with a suffix are narrowed: "13 October" (no suffix) still
  uses every source number.

### 4. Speech/slide conflict (kept: marked, not replaced; MapFile v4)
- No prompt try: `map.md` already says to use the slide's value on a conflict, and Map has
  written the misspoken $80,000 on every seed since 6b.
- **User's choice: mark both values instead of replacing.** A speaker can be correcting an
  outdated slide, so a guard that always trusts the slide would write wrong notes there.
  `mapper.slide_conflicts`: a slide table row (Markdown or vision's "label\nvalue" rows) whose
  one value was never said, while the sentences naming every label word hold exactly one
  number on no slide. `mark_slide_conflicts` adds " (slide: $18,000)" after the spoken value
  in Map strings that name the label; `slide_conflicts` records it (MapFile v3 → v4).
- Dry run on the current runs (4 quick items + long-en, 36 segments): only the planted
  conflict.
- Seed 1, Ask + judge, gate pass: **slide_value_used false → true** (notes: "Quarterly budget
  approved at $80,000 (slide: $18,000) … [1:55] [slide 3]"; R7B kept the note), recall
  85.3 → 86.8%, citation 100% (=), made-up 0 (=), Ask 96.2% (=), near-miss 7/12 (=).
  Report-only: support 93.9 → 92.7% (the judge reads the marked claim as "contradicted by the
  slide"), unsupported numbers 1 → 2 (Map's flagged 6,900, kept by Reduce on the changed
  input). Baseline 63abd32.
- Carry-overs: Ask still answers "$80,000" (it reads transcript and slides, not Map notes);
  " (slide: …)" is English inside notes in any target language (like the `[slide N]`
  citation marker; check in 6d).

### 5. long-en (run 1 of 2 used; one Reduce-only rerun, reverted)
- **Run 1** (seed 1, full, gate pass vs 6c): recall 67.5 → **70.0%** (decisions 50 → 75,
  actions 40 → 40, numbers 90 → 82.5, open questions 75 → 75), citation 98.1 → 96.4%, uncited
  0, made-up 9 → 8, wrong owner/due 2, Ask 83.3 → **94.4%**, Ask citation 93.8 → 100%,
  near-miss 4/4, support 80.5% (report-only). Committed as `evals/baseline-long.json`.
- **Pass 2 no longer drops a third of the decisions.** Traced by text with the eval's match
  rule: decisions Map 18/20 → pass 1 18 → pass 2 15 (6c: 19 → 15 → 10); pass 2 drops 3
  (podcast series, quarterly planning, referral bonus).
- **Actions are misfiled, not lost:** pass 2 holds the text of 15 planted tasks, the eval
  credits 8. Map writes long-en tasks as decisions ("Tom will update the pricing page by the
  2nd of November."): 3 only under decisions (pricing page, moving quotes, contract to legal),
  3 under both (release notes, staging database, customer survey). Pass 1 copies the decision
  form; the 3 in both lose the action item (and its owner) by pass 2. Step 1's map.md rule
  holds on quick, not on long-en's segments.
- Reduce prompts' action example "[owner: ?] [due: ?] Update the release notes" overlaps the
  generator's "write the release notes"; pass 2 wrote exactly that line (owner lost) once.
- **Reduce-only rerun (reverted):** `reduce_part.md` and `reduce_pass2.md` got "A task someone
  will do (such as "Ana will print the event badges by Friday") is an action item with its
  owner and due, never a decision. When a task is listed both as a decision and as an action
  item, write it once, under ACTION_ITEMS." and the example "Translate the app store listing".
  R7B ignored it: pass 1 kept all 6 under DECISIONS word for word; recall 68.8%, made-up 8 →
  12, actions 40% (=); pass 2 moved `[owner] [due]` after the task. The example didn't leak.
  Not tried: `reduce_pass2.md` for the 3 dropped decisions (the misfiling is the larger loss).

### Next
- Carry-overs for the PR / 6e: Map files tasks as decisions on long segments (a pure-Python
  Map step could move or dedupe them; it would change quick: needs a re-baseline); the
  release-notes example in `reduce.md` (changing it re-runs quick's Reduce); Ask doesn't see
  slide conflicts; " (slide: …)" in non-English notes (6d). Step 0's tools are in `scripts/`
  (throwaway): `trace_facts.py <report.json>` and `map_tokens.py [seed]`.

## 2026-10-13 · Phase 6d split and 6d-1 kickoff prompt (no code)

### Plan changes
- §9 6d split (user): **6d-1** notes in other languages (output side), **6d-2** other source
  languages (`xling-{es,ja}`) and regional Map models (D18, now Phase 6d-2). Why: the generator
  is English-only (es/ja need scripts, templates and number words; macOS has es_ES, es_MX and
  ja_JP voices), "Other (beta)" translation doesn't exist yet, and D18 needs gated downloads
  and an HF token, while `en → {de,ar}` only changes ItemSpec's `target`.
- §21 kickoff prompt for 6d-1: a `lang` set (`conflict-de`, `clean-ar`, report-only baseline)
  measured first; scoring that reads numbers in the notes language; the 6b-2 "(slide: …)"
  note localized; Arabic right-to-left check (no custom CSS); "Other (beta)" with
  citation-preserving translation (design proposed before coding).
- Found while writing it: evals.py reads claim numbers with the English-only `numbers_in`, so
  German "42.500" would score as 42.5 (a correct fact counted as missed). Step 1 fixes it
  with the guard's `native_readings`.

### Next
- Phase 6d-1 in a new session with the §21 prompt.

## 2026-10-13 · Phase 6d-1 (notes in other languages)

### 0. Measurement (no fixes)
`lang` set (`evals/sets/lang.yaml`): `conflict-de` (slide-conflict-en's script and media,
target de) and `clean-ar` (clean-en's, target ar); a test checks both scripts equal their quick
twins. Seed 1, `--no-ask`, judge on (report `20261013-085449-lang`). Baseline not written yet:
step 0 is broken by the finding below, so `evals/baseline-lang.json` waits for its fix.

| item | recall (exact) | numbers | actions | judge recall | citation | made-up | lang ok | support |
|---|---|---|---|---|---|---|---|---|
| conflict-de | 23.1% (3/13) | 3/8 | 0/5 | 0/6 | 100% | 0 | de ✓ | 88.9% |
| clean-ar | 53.8% (7/13) | 7/8 | 0/5 | 0/5 | 100% | 0 | ar ✓ | 87.0% |

- **The five section markers don't survive in de or ar: everything lands in Summary.** R7B
  translates them and wraps them in citation spans: `## <co>Zusammenfassung</co: 0:[0]>`,
  `## <co>Schlüsselpunkte</co…>`, `## <co>Aktionspunkte</co…>`; Arabic `## <co>ملخص</co…>`,
  `نقاط رئيسية`, `قرارات`, `مهام`, `أسئلة مفتوحة`. The parser logs "unknown section markers" and keeps
  the text in Summary (§5.6), so Key points, Decisions, Action items and Open questions are
  empty in both notes. This alone explains actions 0/10 (home section empty) and judge
  recall 0/11 (the judge reads the right section's claims). `reduce.md` already says
  "in English … whatever language" and repeats it on the last line. The translated headings
  don't match render's `LABELS` either (de "Kernpunkte"/"Aufgaben", ar "الملخص").
- **Owner/due lines:** de keeps the English labels, inside a citation span:
  `- <co>[owner: Maria] [due: Mittwoch]</co: 0:[0]> <co>Vorbereitung der Kundensurvey</co…>`.
  ar drops the labels and every owner name: `- <co>إرسال العقد إلى قسم القانون بحلول يوم
  الجمعة.</co…>` (Map had `owner: Laura`, Latin script, for all 5 ar tasks). Map's `due` stays
  English in both ("Friday", "Next week"); de Map wrote "Mi", "Do. 4:44" for Wednesday and
  "end of next week" (Tom's task).
- **Digits:** de writes German separators, the `$` after: "80.000 $", "3.550", "10.700 $".
  ar writes Western digits with English separators ("42,500 دولار", "34%"), no Arabic-Indic
  digits or `٬`: English readings already fit Arabic, Arabic-Indic support isn't needed for
  what R7B writes (keep it out unless 6d-2 shows it).
- **Number guard in German:** 2 correct repairs (Map "800.000" → "80.000", "775.000" →
  "77.500", the decimal-mark rule from 6b-2 works in de); 2 flagged (4, 44). The eval's
  `unsupported_numbers` lists 10 (1.75, 3.55, 77.5, 80, …): the English reading of German
  separators, a scoring artifact like fact recall's.
- **Slide-conflict note: not added in German.** The conflict is detected (`Approved budget`,
  spoken 80000, slide `$18,000`), but `mark_conflicts` needs a label word in the Map string
  ("Quartalsbudget" ≠ {approved, budget}) and reads the spoken value with English `numbers_in`
  ("80.000" = 80.0). Both have to change before step 2's word matters.

**Each miss, scoring or real** (Map read by hand: trace_facts matches Map strings by English
text, so its Map column is blank for de/ar text facts):
| item | miss | cause |
|---|---|---|
| conflict-de | f1 3,550, f9 $10,700, f13 $140,000 | scoring: "3.550" read as 3.55 (in the notes, Summary is a home section for numbers) |
| conflict-de | f7 $18,000 | not marked (above); Map/notes say only 80.000 $ |
| conflict-de | f17 37% | real: in Map seg 1, Reduce wrote nothing from seg 1's key points |
| conflict-de | f5, f6, f8, f12, f16 actions | markers (empty section); f5 (Nadia) also missing in Map, f6 has no owner in Map |
| conflict-de | 6 judge facts | markers; f2, f14 also missing in Map, f10 written 3 times |
| clean-ar | f16 4,650 | real: in Map seg 1, dropped by Reduce |
| clean-ar | 5 actions | markers + owners dropped by Reduce (labels gone) |
| clean-ar | 5 judge facts | markers; ar Decisions repeat 3 tasks |

### 0b. Section markers: reply prefill (kept; user chose prompt first, then a fallback)
- **Prompt try A (reverted):** `reduce.md`'s last line "A program splits the notes on the
  markers, so copy them exactly as written above, in English, never translated: ## SUMMARY,
  …". Seed 1: R7B wrote the same translated markers, word for word.
- **Prefill (kept):** Reduce appends `<|START_RESPONSE|>## SUMMARY\n` to the prompt and puts
  it back in front of the reply (`reduce.REPLY_START`, single pass and both two-pass passes;
  Ask has no prefill, D22). Scratch check on the step-0 runs: all five markers in English on
  de and ar, seeds 1–3 (6/6), 18–21 citation spans per reply, notes still in de/ar. No prompt
  change, so REDUCE_HASH stays; a code-only change, so runs need `--fresh`.
- `lang`, `--fresh --no-ask`, judge on, seeds 1 / 2 / 3: recall **69.2 / 61.5 / 57.7%**
  (step 0: 38.5), actions 70 / 40 / 40%, numbers 68.8 / 75 / 68.8% (still English-read
  separators: step 1), judge recall 54.5 / 63.6 / 90.9% (was 0), citation 100%, made-up
  0 / 4 / 3 (all clean-ar), no marker warning in any run log. ar's action items now get their
  owners: R7B copies Map's Arabic task text, and the Map fallback (overlap ≥ 0.6) fills them.
- `quick` seed 1 `--fresh`, Ask + judge: **gate pass**, recall 86.8% (= baseline), citation
  100%, made-up 0, Ask 96.2%, near-miss 7/12, support 92.7%. Baseline unchanged.

### 1. Scoring that reads the notes language (kept)
- `numbers.notes_numbers(text, language)`: English (or None) is `numbers_in`, unchanged;
  other languages read digits with the guard's `native_readings` (de "3.550" = 3550; ar,
  not in DECIMAL_MARK, keeps both readings). `written_values` gives each written number's
  readings. Used for number facts, cross-language due dates, made-up items,
  `misspoken_value_used`, and `unsupported_numbers` (there a number counts if any reading is
  in the sources, the guard's keep rule, so ar "42,500" isn't listed as 42.5). No
  Arabic-Indic digits: R7B wrote none (step 0). Tests include the English reading not firing.
- Same runs re-scored (seed 1 / 2 / 3, judge cached): **recall 80.8 / 73.1 / 69.2%** (step 0b:
  69.2 / 61.5 / 57.7), numbers 87.5 / 93.8 / 87.5%; conflict-de 76.9% on every seed, its only
  number miss is now the slide conflict ($18,000, step 2). unsupported 2 / 0 / 1 (seed 3:
  750000, a real one). clean-ar's 4,650 (seeds 1, 3) is a real Reduce drop (Map seg 1 has it).
- `quick` seed 1 re-scored: gate pass, recall 86.8% (=), unsupported 2 (=).

### 2. Slide-conflict note in the notes language (kept)
- `mapper.SLIDE_WORD`: the note's word for R7B's 23 languages (English when missing), a static
  table like render's LABELS. Three changes were needed, not only the word (step 0):
  the spoken value is read with `native_readings` for the target language ("80.000" = 80000
  in de); in notes in another language than the meeting's the spoken value alone finds the
  string (a German string can't name "Approved budget"); and the slide value is written in
  the spoken value's style ("$18.000": German-read "$18,000" is 18, the eval would miss it).
  The note goes after a trailing currency or percent sign ("80.000 $ (Folie: $18.000)").
  Tests: the value alone doesn't mark same-language notes; "80,000" read as German isn't the
  spoken value; a language without a word gets "slide".
- Code-only Map change: Map's config hash doesn't move, so existing runs keep the old note
  until re-run (`--fresh` in the eval).
- `lang` `--fresh`, seeds 1 / 2 / 3: **slide_value_used true on every seed** (was false);
  R7B kept it in German style: "Quartalsbudget: 80.000 $ (Folie: 18.000 $) [1:55]". Recall
  80.8 / 73.1 / 73.1%, numbers 93.8% each; conflict-de 76.9 / 76.9 / 84.6%.
- `quick` seed 1 `--fresh`, Ask + judge: gate pass, recall 86.8% (=), slide_value_used true
  (=), Ask 96.2%, near-miss 7/12, support 92.7% (all =).

### 3. Arabic in the Notes page and notes.md (kept: Unicode isolates, D29)
- Checked with Claude in Chrome (user's choice) on `clean-ar` seed 3, app on 127.0.0.1 with
  `COHR_RUNS_DIR=evals/data/runs/clean-ar`. Streamlit lays every paragraph out left to right
  (left-aligned, bullets on the left). Lines in Arabic only read right: the line is one
  right-to-left run, so the citation (`0:12`) and the final period sit at its visual left, the
  end of the sentence for an Arabic reader. **A line with Latin text breaks**: Tom's task with
  "(المسؤول: Tom; الموعد: Monday)" had the citation on the far right and the task in the middle.
- Fix within the rules: `render.isolate` wraps each claim line, the summary paragraph and Ask
  answers in RLI … PDI (U+2067 / U+2069) for ar, fa, he, ur. After: Tom's line reads like the
  others (task, owner/due, citation at the end); clicking `5:40` still scrolls Sources to its
  segment. `notes.md` FORMAT_VERSION 1 → 2 (render re-runs on old runs). Left alignment and
  left bullets stay: changing them needs `dir="rtl"` or CSS. Test: isolates for ar in both
  modes, none for en/de, headings untouched.
- Found on the way: on seeds 2–3 R7B writes Arabic owners in Arabic script ("المسؤول: لورا" for
  Laura), so the eval's owner match misses them (clean-ar actions 40%); the name is right,
  only transliterated. Map keeps Latin owners. Not fixed (a Reduce rule "keep names as
  written" exists already); carry-over.

### 4. "Other (beta)" target languages (kept: own translate stage, D30)
- User's choices: own stage (not inside reduce); the New run page lists R7B's 23, then the 44
  beta languages labelled "(beta)" (one selectbox: the page is an `st.form`, so a second box
  shown on "Other" wouldn't appear before submit). Beta list = Tiny Aya card's 67 `language:`
  codes minus R7B's 23 (`config.BETA_LANGUAGES`, source and revision in the comment).
- `Config.notes_language` ("en" for a beta target) feeds Map and Reduce, and Ask's fallback
  language (R7B doesn't write Swahili). Equal to the target for R7B's 23: no existing hash moves.
- `translate` stage (Tiny Aya, `TranslateConfig`, temperature 0), `TranslatedNotes` v1 (new
  snapshot; NotesDoc unchanged). Nothing written for R7B's languages (a stale file is deleted,
  no model loaded). `render.shown_notes` picks it for render, the Notes page and the eval. New
  metrics: `translation_citations_same`, `untranslated_claims`. `lang` gets `clean-sw`.
- **Try 1 (prompt with an example, section line sent):** 18–24 of ~24 claims kept English per
  run, notes detected as English. Tiny Aya translated the example lines (printer, warehouse)
  along with ours and the "## …" line, so markers came back extra or missing; when lines did
  parse, money lost a digit ("$42,500" → "$4,500", "$11,800" → "$1,800", every seed).
  Translate 770–914 s per audio hour (retries).
- **Try 2:** no example, no section line, rules then the lines, target language last. Plus a
  number check: the line must keep the digits. 3–4 claims kept English, Swahili detected, 145
  s/h. But "34%" → "3.4%" passed the digit check (seed 2 lost the return-rate fact),
  `<|END_RESPONSE|>` stayed in the last line's text, and the labels came back translated
  ("[mmiliki: Laura] [tarehe: Ijumaa]"), so they were left inside the task text.
- **Kept (try 3):** `same_numbers` compares values (each source number, read the English way,
  matches one translated number under some reading: "42.500 $" passes, "3.4%" and "$4,500"
  don't); special tokens stripped; any two leading bracket labels read back (owner always
  the original name, due translated). `clean-sw` `--fresh`, seeds 1 / 2 / 3: **output language
  sw 3/3, citations identical 3/3, exact recall 92.3% each = the English notes' 92.3%** (same
  run, scored by the same rules), 4 claims kept English per run
  (the money key points), made-up 0, translate ~145 s per audio hour.
- **Limits (beta):** support rate 22–33% (report-only): the Qwen-4B judge reads Swahili claims
  against an English transcript; and on seed 1 the one-sentence summary came back invented
  ("Mr. Mkuki alishangaa…", no number for the check to catch). Headings stay English. A
  back-translation or length check could catch the invented sentence; not tried.

### Final gate (seed 1, `--fresh`)
- `quick` with Ask + judge: **pass**, recall 86.8%, citation 100%, made-up 0, uncited 0, Ask
  96.2%, Ask citation 100%, near-miss 7/12, support 92.7%, slide_value_used true (all = baseline).
- `lang` (no Ask): recall **84.6%** (conflict-de 76.9, clean-ar 84.6, clean-sw 92.3), numbers
  95.8%, actions 66.7%, citation 100%, made-up 0, language 3/3, clean-sw citations identical,
  4 untranslated, judge recall 56.2%, support 63.5% (clean-sw 21.7%). Written as
  `evals/baseline-lang.json` (report-only, like long).

### Next
- Carry-overs: R7B writes Arabic owner names in Arabic script on some seeds (eval owner match
  misses them); "Other (beta)" leaves ~4 money claims English per run and once invented the
  one-sentence summary (a back-translation or length check, not tried); beta headings English.
  From 6b-2 still open: long-en tasks filed as decisions; reduce.md's release-notes example;
  Ask doesn't see slide conflicts.
- Phase 6d-2 (other source languages, regional Map models, D18) in a new session; kickoff
  prompt to write from these findings.

## 2026-10-13 · Phase 6d-2 kickoff prompt (no code)
- §21 kickoff prompt for 6d-2, written from 6d-1's findings: generator in es/ja first (per-
  language tables, Latin owner names, `say` voices: es Mónica/Paulina + es_MX/es_ES, ja Kyoko +
  ja_JP), English scripts unchanged; `xling-es`/`xling-ja` (target en) join the `lang` set and are
  measured before any fix, including whether Transcribe writes numbers as digits or words.
- Why measure first again: numbers.py reads number words in English only, so a Spanish or
  Japanese transcript that spells numbers out could leave the guard with no source numbers
  (more flags, possible wrong repairs) and the eval with misses; 6d-1 showed scoring misses can
  hide a real problem (the translated markers).
- D18 (regional Map models) is step 2 and needs the user's approval for gated downloads and the
  HF token; `map_model_by_lang` doesn't exist in config.py yet. 6d-1's carry-overs are optional.
- §9's 6d-2 bullet now points to the prompt.

## 2026-10-13 · Phase 6d-2 (other source languages, regional Map models)

### 0. Generator in es/ja, measurement (no fixes)
- `evals/gen/script.py`: a `Lang` table per language (topics, decisions, tasks, open
  questions, line templates, dues, voices, slide formats), lined up with the English lists so
  a seed plants the same facts, owners, dues and values in every language. English scripts
  unchanged (QUICK_SCRIPTS and the lang twins pass). Owners keep Latin names. Numbers are
  plain digits in the script ("42500 dólares", "42500ドル"): `say` reads them the same way in
  every locale, and how they come back is Transcribe's choice. Slides: es "42.500 $" / "34 %",
  ja "42,500ドル". Truth dues stay English ("Friday", "October 20"): cross-language scoring
  compares only their digits. Distinguishable-texts test per language (ja: 3 texts reworded,
  katakana bigrams overlapped at 0.6–0.64).
- Voices: es Mónica (es_ES), Paulina (es_MX), Eddy (es_ES); ja Kyoko, Eddy (ja_JP), Reed (ja_JP).
- Found while building: PIL's default font has no CJK glyphs and no "é" (boxes on the slides).
  Slides use Helvetica (es) and Hiragino Kaku Gothic W3 (ja); English keeps the default, so its
  slides are unchanged. My first ja table had 来週末 (= next weekend) for "end of next week";
  Map translated it right ("next weekend"); fixed to 来週の終わり and xling-ja re-run.
- Eval: `asr_error` gives WER for es (NFKC, thousands separators dropped, "%" → "por ciento")
  and CER for ja/ko/zh (`cer` metric); English `wer` unchanged.
- Probe (say → Transcribe, 12 lines): Transcribe writes **digits** in both languages, never
  number words, except es "354 mil dólares" and ja 万 forms. Grouping follows the voice's
  locale: es_ES "42.500", es_MX "42,500" for the same "42500"; "34 por ciento" → "34%". ja:
  "42,500", "4万2500", "35万4000", and once kanji "四万二千五百" for the same sentence cut
  differently.

`lang` items `xling-es` (report 20261013-135635) and `xling-ja` (20261013-140133, re-run
after the 来週末 fix), seed 1, `--no-ask`, judge on:

| item | WER/CER | recall (exact) | numbers | actions | judge recall | citation | made-up | lang ok | support |
|---|---|---|---|---|---|---|---|---|---|
| xling-es | WER 2.1% | 69.2% (9/13) | 6/8 | 3/5 | 5/5 | 100% | 1 | en ✓ | 85.0% |
| xling-ja | CER 7.1% | 76.9% (10/13) | 5/8 | 5/5 | 4/5 | 70.0% | 0 | en ✓ | 66.7% |

Guard: es 0 repairs, 2 flagged (465, 260); ja 2 repairs (46,500 → 4,650 and 26,000 → 2,600,
both right, from the slide), 0 flagged. Owners: es Transcribe writes "María" for Maria
(Map and notes keep the accent); ja voices spell Latin names letter by letter, Transcribe
writes "OMAR", "VICTOR", "MARIA" (casefold matches), "Laura" and "Tom" as is.

**Each miss** (Map read by hand; trace_facts matches by English text):
| item | miss | cause |
|---|---|---|
| es | f7 $11,800 | **guard**: transcript "11.800 dólares" (es grouping), Map wrote "$11.80"; English `numbers_in` reads the source as 11.8, so the guard kept it |
| es | f15 4,650 | **guard**: Map "465" (and "260" for 2,600); flagged, not repaired: two candidates, 4650 (transcript) and 4.65 (slide "4.650" read the English way) |
| es | f12 Maria's survey | **scoring**: owner "María" ≠ "Maria" (also the 1 made-up item) |
| es | f8 Victor's release notes | Map: filed as a decision without a due date (the 6b-2 carry-over) |
| es | unsupported 6 (42500, 354000, …) | **scoring**: sources read the English way ("42.500" = 42.5) and "354 mil" not read |
| ja | f6 $42,500 | **ASR**: the sentence is dropped at a chunk start (both runs); Vision also lost the slide's values; Map wrote an invented "$195,500" (flagged) on run 1 |
| ja | f9 $354,000 | **Map + guard**: transcript "35万4000ドル" / "13万4000ドル", Map wrote "$34,000" / "$13,000"; the guard read the source as {35, 4000, 13} and kept both through their decimal-comma readings (34 from "34%", 13 from "13万") |
| ja | f7 $11,800 | Map: dropped (in the transcript as "11,800ドル"; seg 0 is dense, the 6b-2 coverage carry-over) |
| ja | citations | Maria's task −21.8 s, Tom's −143 s, 4,650 −104 s: Map seg 1 bundles them in one key point |

- **Transcribe drops sentences at chunk starts in Japanese:** 3 of 18 chunks lose their first
  line (42500ドル, 2600, the opening); es 0 of 15. Re-transcribing chunk 5 cut 0.4 s earlier
  or 9 s shorter keeps the sentence (as "42,500", "4万2500" or "四万二千五百"). Not a numbers.py
  problem; recorded for a chunking decision (outside 6d-2's scope unless the user wants it).
- Step 1 targets, in pure Python: read source numbers the way the source language writes
  them (ja 万/kanji numerals, es "mil" and "11.800" as 11800), in the guard and the eval; the
  eval's owner match ignoring accents.

### 1. Numbers read in the meeting's language (kept)
- `numbers.source_numbers_in(text, language)`: es reads a separator followed by exactly three
  digits as grouping ("11.800", "42,500", both voices' styles), one or two digits as a decimal
  ("3,5"), and "N mil/millones"; ja reads 万/億 and kanji numerals ("35万4000", "4万2500",
  "四万二千五百", "1万1800"). English (or None) is `numbers_in`, unchanged; other languages add
  their native digit readings (de "42.500" = 42500 too). Spanish number words are not read:
  Transcribe wrote none (step 0). Used by the guard's sources, the slide-conflict check (and
  its note's slide value) and the eval's `unsupported_numbers`; all pass the source language.
- **Keep rule narrowed** (D20): the guard keeps a number only by the notes language's
  readings (`native_readings`, as 6b-2 already did for repairs). English "$34,000" was kept
  as 34.0 because "34%" was said. A language not in DECIMAL_MARK (es, ar) keeps either reading.
  Risk, written into a test: with a lone small source number a factor of 1000 away, D20's
  factor rule now repairs instead of keeping ("$34,000" with only "34%" → "$34"); not seen on
  quick or lang (quick repairs 10 = baseline, all to the transcript's value).
- Eval: owner match folds case and accents (`evals.fold`: "María" = Maria, "OMAR" = Omar).
- Found on the first run: the slide-conflict check read the Spanish slide "4.650" as 4.65 and
  added "4,650 (slide: 4.65)" to the notes. It now reads the meeting's language; test with a
  real Spanish conflict too. Japanese conflicts can't be found (label words need spaces):
  a limit, no false notes.
- Results, seed 1 `--fresh`, `--no-ask`, judge on (reports 20261013-141759-lang,
  20261013-142909-lang for xling-es after the conflict fix):

| item | recall | numbers | citation | made-up | repairs / flagged | unsupported |
|---|---|---|---|---|---|---|
| xling-es | 69.2 → **92.3%** | 6/8 → 8/8 | 100% | 1 → 0 | 0/2 → 4/0 (11.80 → 11800, 7.70 → 7700, 465 → 4650, 260 → 2600) | 6 → 0 |
| xling-ja | 76.9% (=) | 5/8 | 70% | 0 | 2/0 → 3/1 (13,000 → 134,000; 34,000 flagged) | 2 → 1 |
| conflict-de, clean-ar, clean-sw | 76.9 / 84.6 / 92.3% (= baseline) | | 100% | 0 | | |

  xling-ja's remaining number misses aren't number reading: 42,500 is dropped by Transcribe
  at a chunk start, 11,800 dropped by Map, 354,000 written "$34,000" by Map (now flagged, not
  passed). xling-es's last miss is Victor's task filed as a decision (6b-2 carry-over).
- `quick` seed 1 `--fresh` with Ask + judge (20261013-142754): **gate pass**, recall 86.8%,
  citation 100%, made-up 0, Ask 96.2%, support 92.7%, repairs 10 / unverified 2 (all =
  baseline). No Map prompt change, so one seed (CLAUDE.md's seeds 1–3 rule is for prompts).

### 2. Regional Map model: Fire (D18; user: Fire only, keep weights only if adopted)
- Mechanism (kept, though nothing is adopted): `ModelSpec.quantize_bits` converts a snapshot
  to N-bit MLX in `cohr models download` (`mlx_lm.convert`, group size 64 like
  tiny-aya-global-8bit-mlx; written to `<hub cache>/cohr-converted/<repo>@<rev>-q8` through a
  temp dir); runtime loads never convert. `Config.map_model_by_lang` (empty) swaps
  `models["map"]` for a matching target before validation, so Map's and translate's
  `models.map` hash moves for that language only; an unset `quantize_bits` isn't dumped, so no
  existing hash moved (REDUCE_HASH and the snapshots unchanged).
- Fire: `mlx-community/tiny-aya-fire` (ungated, CC-BY-NC-4.0, bf16, 6.7 GB) at
  5d5ab5856168aff9ab495b448ef3725752d94351, converted to 8-bit (8.5 bits per weight, 3.3 GB,
  like Global's 3.6 GB). No HF token was needed or used.
- **Bug found:** the first Fire runs scored exactly like Global on every seed: `pipeline.run`
  never gave the run's config to the shared ModelManager, so the eval (one manager built with
  `Config()`) loaded the default models for any `-c models.…` or `map_model_by_lang` override.
  Fixed in `pipeline.run` (`models.set_config(config)`, test). Only those three reports had a
  model override (checked every report in evals/reports).
- Items: `clean-hi` (R7B writes Hindi; Map writes Hindi) and `clean-bn` ("Other (beta)": Map
  and Reduce in English, Tiny Aya translates), both clean-en's script, in the `lang` set.
  `--no-ask --no-judge` (D18 is exact-field recall and output language), seeds 1–3:

| item | Global recall (s1/s2/s3, mean) | Fire recall (mean) | Δ | language ok | made-up |
|---|---|---|---|---|---|
| clean-hi | 53.8 / 69.2 / 53.8 (59.0) | 53.8 / 61.5 / 53.8 (56.4) | −2.6 | 3/3 = 3/3 | 5/4/5 = 5/4/5 |
| clean-bn | 92.3 / 76.9 / 92.3 (87.2) | 84.6 / 92.3 / 92.3 (89.7) | +2.5 | 3/3 = 3/3 | 0 = 0 |

  **Not adopted:** the bar is ≥ 2× the D16 noise (2 × 2.9 = 5.8 points) and neither language
  reaches it. `map_model_by_lang` stays empty. Fire's download and conversion were deleted
  (huggingface_hub's cache API for the snapshot; ~10 GB freed). Earth and Water not tried
  (user's choice).
- Found on the way: clean-hi's recall is low for both models because R7B writes owners in
  Devanagari (लॉरा for Laura, ओमर for Omar): the exact owner match misses all 5 actions and
  counts them as made up, as 6d-1 saw for Arabic on some seeds (carry-over, now hi and ar).
  Also a real error: Tom's "Monday" became मंगलवार (Tuesday) in the Hindi notes.

### Final gate (seed 1, `--fresh`, final code)
- `quick` with Ask + judge (20261013-153253): **pass**, recall 86.8%, citation 100%, made-up 0,
  uncited 0, Ask 96.2%, Ask citation 100%, support 92.7%, repairs 10 (all = baseline).
- `lang` (no Ask, judge on; 20261013-152713), 7 items: recall **81.3%**, citation 96.0%,
  numbers 91.1%, actions 65.7%, made-up 5 (all clean-hi: Devanagari owners), language 7/7,
  WER 1.1% (es 2.1%), CER 7.1% (ja), judge recall 75.0%, support 77.0%.

| item | recall | citation | made-up | support |
|---|---|---|---|---|
| conflict-de | 76.9% | 100% | 0 | 86.4% |
| clean-ar | 84.6% | 100% | 0 | 88.9% |
| clean-sw | 92.3% | 100% | 0 | 21.7% |
| clean-hi | 53.8% | 100% | 5 | 90.5% |
| clean-bn | 92.3% | 100% | 0 | 95.7% |
| xling-es | 92.3% | 100% | 0 | 95.0% |
| xling-ja | 76.9% | 70.0% | 0 | 66.7% |

  The 6d-1 items equal their baseline. Written as `evals/baseline-lang.json` (report-only):
  four new items (xling-es, xling-ja, clean-hi, clean-bn), hence the made-up rise.

### Next
- Carry-overs for 6e: Transcribe drops the first sentence of some Japanese chunks (3 of 18;
  try a lead-in pad or overlap, all sets re-run); owners transliterated by R7B in hi and ar
  (eval misses them; a fix could match the owner against Map's Latin owner for the same
  task, or ask Reduce to keep Latin names); Hindi due date Monday → Tuesday; Map's coverage of
  dense segments and its 万 misreads (flagged now, not repaired: two candidates); no Japanese
  slide-conflict check; "Other (beta)" carry-overs from 6d-1 still open.
- CLAUDE.md lessons worth adding (separate PR, ask): eval model overrides need
  `pipeline.run`'s `set_config` (fixed); read a meeting's numbers in its own language.

## 2026-10-13 · Phase 6e split and 6e-1 kickoff prompt (no code)
- 6e split in two (user): 6e-1 = full set (FLEURS, AMI), robustness (ffmpeg/model errors in
  CLI and UI, sequential memory policy end to end), README update, first full run as a
  measurement; 6e-2 = the carry-over fixes the full run ranks highest, the release gate and the
  1.0.0 bump (only when asked). Why: quick already meets every v1 target (recall 86.8%,
  citation 100%, made-up 0, language 100%), so the ~15 carry-overs from 6b-2 to 6d-2 don't
  block v1, and the first full run (real AMI meetings, 14 FLEURS languages) is better input for
  choosing among them than the synthetic sets.
- The kickoff asks for a proposal first: §18's "~1 h" for full predates long-en (~40 min
  alone), and FLEURS/AMI download sizes aren't known yet. README.md is still Phase 1's (status,
  "ingest only for now"), so 6e-1 updates it alongside the cache size note.

## 2026-10-13 · Phase 6e-1 (full set + robustness, measured)

### 0. Proposal: what `full` contains (user approved all four choices)
- **AMI audio from the official mirror**, not `edinburghcstr/ami`: the Hub dataset holds one
  row per utterance per speaker (`begin_time`/`end_time`, uppercase text, no punctuation), so
  a meeting would have to be mixed back onto a timeline with no crosstalk, room noise or
  untranscribed speech. `amicorpus/<id>/audio/<id>.Mix-Headset.wav` is the whole meeting
  (16 kHz), from the same site and licence as the summaries.
- **FLEURS streamed**: the test tars are 159 MB (nl) to 658 MB (it), 5.9 GB for 14 languages,
  to use 280 clips. The parquet branch is larger (en 402 vs 290 MB) and the rows API refuses
  the dataset ("scan size limit exceeded"). Sample = 20 clips by seed from the first 100 tar
  members at the pinned revision `70bb2e84…`, one per sentence; ~0.9 GB read, ~10 MB kept.
- **`long-en` stays in `long`**: `full` ≈ 1 h (synthetic ~20 min, FLEURS ~4, AMI ~26 + judge
  ~9, estimated from quick's 513 s per audio hour without Vision); the release gate runs
  `full`, then `long`.
- **Meetings**: ES2004b, ES2004d, IS1009b, IS1009c, TS3003c (test split, 30–43 min each, 3.05 h,
  ~352 MB). ES2004c dropped after reading its summary (1 decision, 0 actions). Reference
  sentences: abstract 43, decisions 23, actions 11, problems 12; one sentence is 4–9 points
  of a section's coverage, so AMI coverage starts report-only (D31, proposed).
- Licences checked again: FLEURS and `edinburghcstr/ami` cards `cc-by-4.0`, ungated; the AMI
  licence page and `LICENCE.txt` in `ami_public_manual_1.6.2.zip` (downloaded, 22.9 MB, 143
  `abstractive/*.abssumm.xml`): "released under the Creative Commons Attribution 4.0
  International Public License agreement (CC BY 4.0)".

### 1. FLEURS (Transcribe only; report 20261014-060444-full, d1d5bbc)
- `evals/sets/full.yaml`: `include: [quick, lang]` + a `fleurs:` block (pinned revision, seed 1,
  20 clips from the first 100 tar members, Transcribe's 14 languages → FLEURS configs).
  `src/cohr/fleurs.py` streams the tar start through `HfFileSystem`; `evals.run_fleurs`
  transcribes each clip whole (no VAD); error is over a language's 20 clips together
  (`asr_corpus_error`). `--item fleurs` / `--item fleurs-ja` run FLEURS alone.
- Kept: 205 MB, not ~10 MB (step 0's estimate): the clips are 32-bit float WAVs, ~15 MB per
  language. Streaming a language took ~10 s; the whole FLEURS run ~4 min.
- Error (WER; CER for ja/ko/zh): pl 3.8, it 3.9, ja 4.0, fr 4.2, pt 4.2, en 5.3, es 6.0,
  nl 6.5, ko 7.1, vi 7.1, de 7.6, el 8.1, zh 12.5, ar 16.1%. Means: WER 6.6%, CER 7.9%.
  Transcribe alone: 149 s per audio hour (116–189 by language).
- Worst clips, two kinds:
  - Scoring form, not errors: de "10.000" → "zehntausend", fr "22 h 00" → "vingt-deux
    heures", pl "p.n.e." spelled out, ko a parenthetical "(Rossby number)" not spoken. The
    reverse of the expected direction: FLEURS writes digits, Transcribe writes words. Only
    English number words are read as digits (`wer_text`).
  - Real errors, numbers among them: ar "06:30 و 07:30" → "36 و37" (clock times lost), nl
    "5 en 100 dollar" → "$5.000 en $100", zh "World Wargate" made up twice, pt names
    (Goethe, Fichte, Schlegel → "Goth, Fish e Skeleton"). The Map number guard checks
    against the transcript, so a value wrong there passes it.
- Carry-overs (6e-2 ranking): Transcribe number errors in ar/nl (feeds the guard); number
  words → digits in the FLEURS scoring for every language (eval fairness; report-only now).

### 2. AMI (5 meetings, 3.06 h; report 20261014-071244-full, 7c389d4, `--no-ask`)
- `src/cohr/ami.py`: the mirror's `Mix-Headset.wav` + `abstractive/<id>.abssumm.xml` from the
  annotations zip (sentence counts 43/23/11/12, as in step 0). `evals.run_ami` runs the
  pipeline (audio only, English); `judge_ami` asks coverage per reference sentence in its home
  sections (`AMI_HOME`) and support per cited claim, in the synthetic items' judge session.
  The abstract is looked for in Summary **and** Key points (§18 said Summary): our Summary is
  one claim, AMI abstracts are 5–11 sentences. AMI rows stay out of `items` and the gated
  totals; their stage costs are a separate table.
- Download: the mirror cut the scripted download at 21 of 75 MB (~185 KB/s; the length check
  kept nothing), then answered 403 to every request from this machine, ranges included. The
  user fetched the 5 WAVs in a browser; all complete (sizes = header + data; ES2004b = the
  Content-Length the mirror had announced). IS1009b/c carry recorder chunks after `data`
  (cue, LIST, muma, chrp, bext) and are 5–10× louder (RMS 0.09 vs 0.01–0.02; 4% near-silent
  samples vs 37–50%): another room setup.
- **Speed (§9 definition of done): 326 s per audio hour = 5.4 min/h** (target ≤ 20),
  `resident`, no Vision. Per stage: transcribe 98, reduce 120, map 82, vad 25. Each meeting
  167–229 s. Peak 14.1 GB (limit 26.8). Judge: 334 calls, 632 s, 3.9 GB.
- Coverage (judge, report-only): all 44.9%; abstract 60.5, decisions 34.8, actions 18.2,
  problems 33.3%. Per meeting 22–63%. Support rate 80.2% (199 yes, 38 partial, 11 no of 248
  cited claims); 2 uncited claims.
- **Main finding: Reduce's reply is cut at `reduce.max_tokens` = 2048 on 4 of 5 meetings**
  (`notes.json` `raw`: 2053 tokens with the prefilled start; ES2004d ended at 1853 with
  `<|END_RESPONSE|>`). R7B wraps nearly every phrase in `<co>…</co: 0:[n]>`, so 34–63 claims
  fill the budget. Effects: TS3003c has no Decisions, Action items or Open questions at all
  (36 key points, then the cap: its 0% on those parts); ES2004b, IS1009c end mid-citation
  (`…compatibility</co: 0:[`: the 2 uncited claims); IS1009b ends on a bare `##` (the
  "unknown section markers ['#']" warning). The section order puts Decisions, Action items
  and Open questions last, so they lose first. quick (5 min) and long-en (two-pass,
  `TWO_PASS_MAX_TOKENS` 4096) never reach it.
- Other patterns: Decisions hold tasks and considerations ("The project will explore /
  consider / investigate …": ES2004b 15 decisions), the long-en carry-over in real meetings;
  the judge then misses decided items stated as "consider". AMI actions name roles ("The
  Marketing Expert will …"); our action items mostly have no owner (8 of 11 in ES2004b).
  Partial support is mostly numbers: ES2004b "34% take too much time to learn" where the
  transcript says "slightly more than a quarter" (a number in no source; `unsupported_numbers`
  isn't run on AMI).
- **Diagnostic (no code change): `-c reduce.max_tokens=4096`** (report 20261014-072637-full;
  only Reduce and the judge re-ran). Every reply ends with `<|END_RESPONSE|>`: 1853 (ES2004d,
  unchanged: temperature 0, never capped), 2103, 2259, 2665, 3470 tokens (TS3003c, 42.8 min:
  85% of 4096, so a 60 min single-pass meeting may reach it). Coverage 44.9 → **52.8%**:
  decisions 34.8 → 52.2, actions 18.2 → 36.4, abstract 60.5 → 62.8, problems 33.3 (same).
  TS3003c 22.2 → 55.6% (15 decisions, 10 actions, 6 open questions now). Uncited 2 → 0.
  Support rate 80.2 → 75.2%: 62 more cited claims, of which 34 yes (233 yes, 57 partial,
  20 no of 310): the added decisions and open questions are weaker. Reduce 120 → 139 s per
  audio hour; total 5.75 min/h; peak 13.8 GB. The runs in evals/data/runs/ami-* now hold the
  4096 notes; a default run re-runs Reduce (config hash).
- Carry-overs from AMI (6e-2 ranking input): (1) Reduce's cap cuts real 30–43 min meetings
  and nothing reports it (no finish-reason check: the notes look complete); raising it to
  4096 is measured above, and a check that warns or retries when the reply hits the cap is
  needed for longer meetings. (2) Tasks and considerations filed as decisions. (3) Action
  owners: roles are named in AMI, ours mostly have none. (4) Numbers in no source on real
  meetings (`unsupported_numbers` to run on AMI too).

### 4. Robustness (7566579, then the policy check)
- Paths checked: ffmpeg/ffprobe missing (CLI run, CLI eval, UI New Run), models missing,
  partly downloaded, deleted after a run, damaged, and the Hub unreachable during
  `cohr models download`. Already clear before 6e-1: a deleted model (`ModelsMissing` before
  any stage, so resume stops with the ingest still done), ffmpeg in `cohr run` (probe).
- Fixed: an **interrupted download passed as cached** (the snapshot folder exists with the
  files that finished), so `cohr models download` said "all cached" and the run failed in the
  loader. `models.check_complete` (offline): config.json + every shard the index names, or a
  `*.safetensors`; incomplete = missing, so the download resumes. All 6 cached models pass.
  No network: `DownloadError`, one line ("run it again to resume"), not a traceback. Damaged
  files: the load error says `hf cache rm model/<repo>` + `cohr models download`. UI: a page
  warning when ffmpeg/ffprobe are missing, and the install hint instead of "Could not read
  the recording". `cohr eval` checks ffmpeg before generating.
- **sequential vs resident** (clean-en, `--fresh` each, `--no-ask --no-judge`, same seed):
  transcript, map.json and Reduce's raw reply are **identical** byte for byte. Peak memory
  per stage (GB, resident → sequential): transcribe 5.63 → 5.63, vision 8.59 → 4.46, map
  11.68 → 4.40, reduce 16.76 → 5.92 (vad 0.05): resident adds each model to the last;
  sequential peaks at the largest single model. Stage times equal (reduce 34.7 vs 34.3 s):
  each model loads once per item either way; sequential costs reloads only across items.
  Both runs "fail" reduce +41–43% s/audio h against baseline.json: one item pays R7B's load
  that 4 items share; not a policy cost (a single-item run isn't comparable to the 4-item
  baseline).

### 6. First full run (report 20261014-085703-full; baseline-full.json written)
- First attempt stopped at xling-es: `KeyError: 'Tickets abiertos'`. Ask's question templates
  are keyed by the English script's labels, and lang had only ever run with `--no-ask`. Fix
  (7d63600): Ask is skipped on meetings not in English (logged). The rerun reused every stage
  and the 171 cached Ask answers; only xling-ja's pipeline, Reduce on AMI (back to 2048) and
  197 judge questions ran.
- **Gate (quick items, D31): pass, no regressions against baseline.json.** Fact recall 89.7%
  (decisions 75, actions 95, numbers 93.8), citation accuracy 98.4%, made-up 0, uncited 0,
  gap words 0, language 100%, slide alignment 100%, open questions 100%, support 95.2%, Ask
  98.1% (citations 100%, near-miss 58.3% = 7/12). **Every v1 target met** (§18), speed
  included: 14.7 min per audio hour with Vision (quick), 5.5 on AMI (audio only).
- lang items (report-only), seed 0: baseline-lang is seed 1 (`-c seed=1`), so the two aren't
  comparable item by item (Map samples at 0.3; D16: means of seeds 1–3). Recall 82.4%,
  citations 90.7%, made-up 6 (clean-hi 5: owners in Devanagari, the 6d-2 carry-over),
  language 7/7, judge recall 75%. clean-sw support 21.7%, as in its baseline: the judge reads
  Swahili claims against the English transcript and answers "no" ("the claim is in
  Swahili…"); recall there is 92%. xling-ja recall 69.2%, citations 77.8%.
- FLEURS: identical to step 1 (WER 6.6%, CER 7.9%; Transcribe deterministic). AMI: identical
  to step 2's first run (coverage 44.9%, support 80.2%, 2 uncited), so the 2048 cap shows the
  same way. AMI peak 17.3 GB (14.1 alone): Vision stays resident from the synthetic items.
- Judge: 608 questions (197 new, 337 s). The run took ~35 min with the caches; from empty
  caches, ~1 h (step 0's estimate held: FLEURS 2 min, AMI 18 min, judge 10 min, synthetic
  with Ask 30 min).

### Carry-overs ranked by the full run (input for the 6e-2 kickoff)
1. **Reduce's output cap cuts real meetings, silently** (AMI 4 of 5 at exactly 2048 tokens;
   TS3003c has no Decisions, Action items or Open questions). 4096 measured: coverage
   44.9 → 52.8%, no cuts, Reduce +16% time; TS3003c used 85% of 4096, so 60 min meetings need
   either more or a check. Needs: a finish check (warn, or retry larger) + a new default.
   Hits every real meeting over ~30 min; nothing synthetic reaches it.
2. **Tasks and considerations filed as decisions** (AMI ES2004b: 15 "decisions" of the form
   "The project will explore/consider…"; long-en since 6c; quick decision recall 75%).
3. **Owners**: R7B writes owners in Devanagari/Arabic script (clean-hi recall 53.8%, made-up
   5: partly the eval's owner match); AMI actions name roles, ours mostly have no owner.
4. **Numbers in the transcript and notes**: Transcribe errors (FLEURS ar "06:30 و 07:30" →
   "36 و37", nl "5" → "$5.000"), Map's 万 misreads, numbers in no source on AMI ("34%" for
   "a quarter"; `unsupported_numbers` to run on AMI too).
5. **Japanese**: xling-ja recall 69.2% (chunk-start drops, dense-segment Map drops) and no
   slide-conflict check for Japanese.
6. Eval gaps (no product change): judge on "Other (beta)" notes (judge the English notes
   before translation: clean-sw 21.7%); Ask questions for es/ja meetings; FLEURS number words
   per language; D31's memory row (no threshold exists).
7. Ask near-miss 58% (7/12, unchanged since 6b) and Ask with slide conflicts; the
   "Other (beta)" limits.

### Next
6e-2 (separate session, kickoff written from the ranking above): carry-over fixes, then the
full release gate and the 1.0.0 bump only when asked. D31 confirmed by the user (2026-10-14):
`full` gates on its quick items' metrics and stage speeds; memory stays ungated.

## 2026-10-14 · 6e-2 kickoff prompt (no code)
- PLAN §21 "Kickoff prompt for Phase 6e-2": carry-overs 1–4 from the 6e-1 ranking (Reduce's
  output cap with a finish check, decisions vs tasks, owners, numbers), each measured on the
  sets it reaches, then the release gate (full + long; the 1.0.0 bump asked separately).
  5–7 (Japanese, eval gaps, Ask near-miss) default to after v1; step 0 may argue otherwise.

## 2026-10-14 · Phase 6e-2 (carry-over fixes + v1)

### 0. Proposal (user approved, with the choices below)
- **Reduce's cap (1)**: AMI's Reduce prompts are 8–11K tokens for 30–43 min, so single pass
  runs to ~2 h (32K) while the reply grows ~81 tokens per minute (TS3003c): ~4.9K at 60 min,
  ~10K near 2 h. Plan: `stream_generate`'s finish reason; a reply is cut when it stopped on
  "length" with no `<|END_RESPONSE|>`. Default 8192 for single and two-pass (temperature 0:
  replies that already finished are unchanged, as ES2004d showed at 2048 and 4096). A reply
  still cut is flagged (`truncated` in notes.json, run log, UI), never retried (at
  temperature 0 a retry writes the same start again) and never completed.
- **Decisions (2)**: not only Map. ES2004b's 15 notes decisions, each matched to its closest
  map.json item: 8 are Map *action items*, 3 key points, 4 Map decisions; in the other AMI
  meetings and long-en they come from Map decisions, about half of them task-like ("will
  explore/consider"). Plan: Python in Reduce first (a decision matching a Map action item at
  ACTION_MATCH_MIN moves to Action items with its citations); a Map prompt rule only if the
  trace still shows it (seeds 1–3, then long-en and AMI). Traced again after step 1: today's
  AMI notes are the cut 2048 ones.
- **Owners (3)**: only clean-hi in the last run (clean-ar is Latin now); Map already writes
  "लॉरा" in map.json, Reduce copies it, and the eval's exact owner match makes all 5 a wrong
  owner. User's choice: fix the notes (the prompt says keep names as they appear): an owner
  not in its cited segment is transliterated (`anyascii`, ISC) and matched to the names said
  there; kept as written when nothing matches.
- **Numbers (4)**: `unsupported_numbers` on AMI, report-only (ES2004b's "34%" is a Map key
  point the guard kept). 万: the guard flags "$34,000" by design (one edit from both 354,000
  and 134,000). User's choice: Map's input gets Japanese numbers as digits ("35万4000ドル" →
  "354,000ドル", `_ja_value`) when the notes language isn't Japanese; the transcript and
  citations are unchanged.
- **After v1**: 5 (Japanese: 万 is covered by 4; chunk-start and dense-segment drops have no
  single fix), 6 (eval-only gaps; the gate doesn't depend on them), 7 (Ask near-miss,
  unchanged since 6b; Ask is 98.1% on the gate).

### 1. Reduce's output cap (carry-over 1)
- `CommandR7B.complete` runs `stream_generate` (the same text as mlx-lm's `generate`) and
  keeps the finish reason; `generate` stays for Ask. `reduce.cut_off`: "length" with no
  `<|END_RESPONSE|>`; a reply closed by its last allowed token is complete (test where it
  must not fire), and so is an end of turn without the marker.
- `reduce.max_tokens` 2048 → 8192, single and two-pass (`TWO_PASS_MAX_TOKENS` removed).
  A reply still cut: `NotesDoc.truncated` (count, v5), a run-log warning, a warning on the
  Notes page, and a "cut" column + `ami_truncated` in the AMI report. The cut claim is kept as
  written and uncited. REDUCE_HASH moved (reason in its comment).
- Existing replies (R7B tokenizer): quick 688–1098 tokens, long-en parts 824–1692 and pass 2
  **3848 of 4096** (94%), every one closed with `<|END_RESPONSE|>`, so at temperature 0
  none changes; long-en isn't re-run for this step.
- quick (20261014-093829): **gate pass**; Reduce re-ran on all 4 items and every notes.json
  is byte-identical to before (`raw` and sections), `truncated` 0.
- AMI (`--item ami --no-ask`, report 20261014-094639-full): coverage 44.9 → **52.8%**
  (abstract 62.8, decisions 52.2, actions 36.4, problems 33.3%), every number equal to 6e-1's
  4096 diagnostic (the same replies at temperature 0), cut 0, uncited 0, support 75.2%.
  Reduce 139 s per audio hour, AMI total 5.75 min/h.
- New tooling: `cohr eval --rerun STAGE` (3ec3fbb), for code-only changes (step 2 on).

### 2. Tasks filed as decisions (carry-over 2; Python, `reduce.filed_task`)
- A notes decision moves to Action items (citations kept; owner/due from the Map item, as for
  action items already) when it matches an action item of a cited segment at
  ACTION_MATCH_MIN and **better** than any of those segments' Map decisions, or **as well**
  when that action item names an owner. When an action item already states it, its
  citations are added there (`add_task`): no task listed twice.
- Why the owner on a tie, from the scores of every candidate move: strict wins are all
  ES2004b's "The project will explore/consider/investigate/prepare …" (6, all tasks); ties
  *with* an owner are synthetic duplicates of a listed task ("Laura to send contract to legal
  by Friday": clean-en 4 of 7 decisions, noisy-en 3, clean-bn/sw 4, xling-es/ja 1); ties
  *without* one were AMI decided designs Map also wrote as tasks ("Design the single curved
  side remote control" = TS3003c's reference decision "The case will be single curved";
  IS1009b "Ensure corporate image…"). A first version moving every tie moved those 5.
  (Map also gave several of those ownerless AMI tasks a due "End of next week"; not acted on.)
- Replay of the stored replies (no model): AMI ES2004b 15 → 9 decisions, IS1009c 7 → 6 (the
  marketing manager's presentations), ES2004d, IS1009b, TS3003c unchanged.
- quick `--rerun reduce` (20261014-094845; it ran the tie-moves-always version, which moves
  the same claims on quick): **gate pass**, every metric equal but support 95.2 → 94.7%:
  76 judged claims instead of 83, the 7 merged duplicates were all "yes".
- Not covered (step 2b if needed): a task Map files only as a decision (xling-es "Victor will
  write the notes for the next version"). AMI is measured with step 4's AMI run.

### 3. Owners in the notes' script (carry-over 3; Python, `names.py`, `reduce.restore_owners`)
- New dependency `anyascii` 0.3.3 (ISC, pure Python, no dependencies; checked in its
  classifiers): transliteration only.
- An owner with non-Latin letters, one word, not in the transcript or slides as written, is
  matched to a capitalised Latin word said in the meeting: same consonants after
  transliteration (blurred pairs merged: c/k, l/r, v/b/f, Arabic و/ي as vowels, aspiration
  h dropped), closest spelling breaking ties (Arabic "ماريا" → Maria, not Omar), equal ties
  kept as written. Candidates are words never also written in lowercase ("Team" when "the
  team" is said). Kept: Latin owners, roles ("मार्केटिंग मैनेजर"), names nobody said.
  Matches all 19 forms of the eval names in Devanagari, Arabic, Bengali and Japanese except
  Japanese ビクター (no final r): Japanese owners in Japanese notes aren't in any set.
- Applied once on the final notes (single and two-pass); logged ("लॉरा → Laura").
- lang seed 1 (20261014-100217, with step 2; baseline-lang pass): recall 81.3 → **86.8%**,
  clean-hi 53.8 → **92.3%** (actions 0 → 100%), made-up 0, every other item's recall equal.
  Citation accuracy 95.9 → 89.9%: clean-hi 100 → 58.3%, because its 5 action items are now
  found and scored. **New carry-over (cross-language jump times):** their Hindi text shares
  no token with the English transcript, so `refine_time` falls back to the segment start
  (seg 0 @ 0.002 s, 46–246 s early). Arabic notes escape it only because their task text
  keeps the Latin names. Not caused by step 3; hidden until the facts matched.

### 4. Numbers (carry-over 4)
- **AMI report** (eval only): `unsupported_numbers` per meeting (count, list in the .md,
  `ami_unsupported_numbers`), read as English. Measured with the next AMI run.
- **万 in Map's input: tried, not kept.** `numbers.ja_grouped` gave Map "354,000ドル" for
  "35万4000ドル" when the notes aren't Japanese (digits + 万/億 only: kanji-only runs can be
  words, "十分", "万一"). xling-ja `--no-ask --no-judge`, seeds 1–3, before → after: recall
  76.9/53.8/76.9 → 84.6/53.8/76.9% (mean 69.2 → 71.8), but **354,000 missed on every seed
  either way**: given the grouped digits, Tiny Aya still wrote "3,400,000" (seed 3; before,
  "345,000"). The one fact gained is 11,800, said in digits all along (a different Map sample
  from a changed input). So the 万 misread is Map's digit copying, not its reading of 万; the
  guard flags the result (no unique repair), as before. The patch is kept out of the tree.
  The xling-ja runs on disk hold the trial's Map output for seeds 1–3: re-run Map
  (`--rerun map`) when xling-ja is next measured at those seeds.
- Transcribe's own number errors (FLEURS ar, nl): recorded in 6e-1, not fixed.

### 5. Release gate (full, then long; user-approved; `--rerun reduce`)
- **full** (20261014-113730, `baseline-full.json`): **gate pass**. Against 6e-1's run:
  recall 85.5 → **88.7%**, made-up items 6 → 1, action recall 85.5 → 94.5%, citation accuracy
  94.1 → 90.8% (clean-hi 100 → 58.3%: cross-language jump times, step 3), support
  85.1 → 85.3%, Ask 98.3%. AMI: coverage 44.9 → **52.8%** (abstract 62.8, decisions 52.2,
  actions 36.4, problems 33.3%), support 80.2 → 76.6%, uncited 2 → 0, cut 0, numbers in no
  source 0, 5.8 min per audio hour, peak 17.3 GB. xling-ja's Map re-ran from clean code
  (config stale after the step 4b trial). Baseline rewritten (b6ddfaf).
- **long** (20261014-123008, report-only): "FAIL" against a baseline from 6b-2. long-en was
  regenerated (evals/gen changed in 6d-2 and long hadn't run since), so every stage ran on
  new audio: recall 70.0 → 67.5%, made-up items 8 → 12, made-up open questions 1 → 4, Ask
  94.4 → 77.8%, support 80.5 → 75.7%. Attribution, Reduce replayed with main's code on the
  same run (no model change, 2048 cap):
  - main: recall 63.75%, made-up items 8, open questions 4, action recall 40%.
  - branch: recall 67.5%, made-up items 12, open questions 4, action recall 55%.
  - The four pass-1 replies are byte-identical (each closed with `<|END_RESPONSE|>`), so
    the 8192 cap changes nothing here. The difference is step 2: pass 1's moved decisions
    reach pass 2's input as action items, and pass 2 then keeps more tasks: 3 planted ones
    found (migrate the staging database, write the release notes, prepare the customer
    survey), 3 vague Map tasks ("Continue monitoring retention metrics…", "Review and
    clarify the reporting of cloud costs", "Address the support queue backlog…") and one
    with a wrong date (competitor research "1 November", said "11 November").
  - Ask reads map.json, not the notes: its drop is the new audio's Map, not this branch.
  `baseline-long.json` rewritten from this run (b6ddfaf).
- **lang** seed 1 (20261014-124855, `--no-ask`, report-only): recall 81.3 → **86.8%**
  (clean-hi 53.8 → 92.3%, every other item equal; xling-ja 76.9% with Map re-run from clean
  code), made-up items 5 → 0, citation accuracy 95.9 → 89.9% (clean-hi 58.3%, jump times),
  support 77.0 → 76.8%. `baseline-lang.json` rewritten.
- quick: `baseline.json` rewritten from step 2's run (20261014-094845; step 3 doesn't reach
  Latin owners, step 4 is eval-only). Gate green: ruff, ty, pytest 526 passed.

### After v1 (post-v1 list)
1. ~~**Long-en made-up tasks**~~: done in Phase 8b (D40, due guard + vague-task move;
   made-up items 12 → 7). Left over: see 8b's "Open".
2. ~~**Cross-language jump times**~~: done in Phase 8a (D39, language-neutral anchors;
   clean-hi 58.3 → 100%, xling-ja 70 → 90%). Left over: see 8a's "Open".
3. ~~**Map files a task only as a decision**~~: done in Phase 8d, widened to every missed
   action item Map reached (D42, Python on map.json and the final notes, no prompt rule;
   9 of 17 found). Left over: see 8d's "Open".
   Map's never-written tasks: done in Phase 8e (D43, flagged sentences → one Tiny Aya call;
   quick action recall 100% at seeds 0/2/3, long-en 70 → 90%). Left over: see 8e's "Open".
4. ~~**Map miscopies digits in ja meetings**~~ (D35): widened in Phase 8g to every missed
   number (D45: short numbers never repaired, figures Reduce lost re-added; long-en 77.5 →
   85.0%). xling-ja f9 stays flagged. Left over: see 8g's "Open".
5. ~~**Japanese ビクター**~~: done in Phase 8i (D51: a final ー restores to -r; new lang item
   `clean-ja`, every owner restored; Japanese notes' numbers read by the eval and the guard).
6. ~~**Map's ownerless "End of next week" dues**~~: dropped from the notes in Phase 8b
   (D40: 34 unsaid, 4 placeholders; 1 of AMI's 39 left), and from map.json in Phase 8c (D41).
7. ~~**Ask on the regenerated long-en**~~: done in Phase 8c (D41, best-span placement and
   Map's dates and dues checked; 77.8 → 88.9%). Left over: see 8c's "Open".
8. Carry-overs 5–7 from 6e-1: ~~Japanese chunk-start drops~~ (Phase 8i, D48: the decoder's
   lost openings on unpunctuated chunks, restored by a probe), dense-segment drops, eval-only
   gaps, Ask near-miss (55.6% on full).
9. ~~The 1.0.0 bump~~: done on request (`uv version --bump major`, pyproject + uv.lock;
   README status), merged as 32576c1. **No release followed:** the §16 `release` job was
   left out in Phase 1 ("Not in Phase 1 (by plan)") and never added, so main's CI ran only
   `check` and the repo had no tags. Added on `ci-release-job` (user's choice over a hand
   release); its merge to main tags v1.0.0 and attaches the wheel and sdist.
10. ~~**Japanese chunks without punctuation**~~ (8a): done in Phase 8i (D49: placed at the
    first hit inside the chunk; xling-ja MARIA −21.8 → −1.4 s).
11. ~~**Missed decisions**~~: done in Phase 8f (D44, cue sentences → decisions after Reduce;
    quick 100% at seeds 0–3, long-en 70 → 95%, lang 21 → 24/28). Left over: see 8f's "Open".
12. ~~**AMI coverage**~~ (8g's "Open"): done in Phase 8h (D46 unknown headings and repeats,
    D47 segments a section skipped; 44.9 → 50.6%, decisions 30.4 → 47.8%). Left over: see 8h's "Open".

## 2026-10-14 · Phase 7 kickoff prompt (no code)
- PLAN §21 "Kickoff prompt for Phase 7": cross-meeting search from §13. Step 0 checks the
  planning sketch against today's code: what's indexed (shown notes, citations), the
  tokenizer for zh/ja/ko (reuse Ask's D17 trigram windows), who syncs the index, how a hit
  opens Notes at its moment, and an optional model-free search check on the eval runs. Then
  index.py, sync + `cohr index rebuild`, the Search page. No models, no eval runs; the 6e-2
  post-v1 list stays open. The 1.1.0 bump is asked separately.

## 2026-10-27 · Phase 7: cross-meeting search
Branch `phase-7-search`. No models, no eval runs; the 6e-2 post-v1 list stays open.

### 0. Proposal (approved as proposed)
- Index transcript items per chunk (≤ 30 s, not 300 s segments), slides, `shown_notes`
  claims with citations, action items in their own table. Sync on read, keyed by file stats
  (D36). Two FTS5 tables, query picks (D37). Ask's term code moved to `fts.py`. Hits open in
  Ask's drawer or in Notes with a banner link (anchors only work within a page). Plain SQL
  with `schema_version`, rebuilt atomically. `cohr index rebuild` and `cohr search`. A
  report-only search check on the eval runs, narrowed to what English truth can test.

### 1–3. Built
- `src/cohr/index.py`, `src/cohr/fts.py`, `app_pages/search.py` (+ nav, Notes banner,
  `ui.Focus`), CLI. Ask's ranking is unchanged (its tests pass as they were; the
  `query_terms` test moved to `test_fts.py`).
- **Found by a test (D37):** unicode61 reads a Japanese run with digits in it
  ("チケットは390件") as one token, so "390" never matched the xling-ja transcript. Latin
  queries now search ja/zh/ko rows through the trigram table (3+ character terms).
- Browser check (Claude in Chrome, eval runs through a scratch `COHR_RUNS_DIR` of symlinks):
  at 1512 px, "42,500" lists each meeting's key point; Listen opens the drawer at 1:33 with
  the segment's slides; Open notes switches the run, shows "From Search: 1:33" and cues
  that segment's player; "採用計画" finds the xling-ja transcript at 2:11; the Action items
  tab lists long-en's tasks with owner, due and time. **Not checked: the narrow window.**
  `resize_window` reported success but the viewport stayed 1568 px (the window was likely
  maximized); the page uses only stacking-safe layout (one column of bordered results, a
  wrapping horizontal filter row), but nobody has looked at it below 640 px.

### 4. Search check (`scripts/search_check.py`, report-only)
Index of all 17 eval runs (AMI included as noise), each planted fact of the 12 gen items
queried by its number (as typed: "14,500"), its owner and its text, across every run;
scored on the fact's run: a transcript hit / a note within ±15 s among its top 5.

| query | transcript | notes |
|---|---|---|
| number | 119/126 | 100/126 |
| owner | 67/75 | 53/75 |
| words | 252/288 | 109/288 |

- **Transcript hits at the chunk start were up to 25 s early** (first run: clean-en words
  10/18 in the transcript). Search now places a transcript hit at the chunk's sentence that
  best matches the query (`reduce.sentence_offset`, D19's even speaking rate): 17/18.
- **Bengali digits (D38):** Tiny Aya writes "৯৫০", "$৪২,৫০০"; numbers in clean-bn notes
  2/8 → 8/8 after folding every script's digits to ASCII (index and query).
- Notes in another language than the query find only names and numbers (words: ar, hi, de
  0/18–19), as expected of keyword search; the same meeting's English transcript is found.
- clean-hi owners in the notes 0/5: found, but every action item cites t = 0.002 / 291 s, the
  segment start: post-v1 item 2 (cross-language jump times), not search.
- Spoken number forms: es "354 mil", ja "35万4000" don't match "354,000" (3 of 126; one more
  ja miss is a fact said in the chunk before its span). Candidate below.
- long-en owners 14/20 in the transcript: names said often push the fact's chunk out of the
  top 5 (ranking, not matching).

### Follow-ups (not done)
1. Spoken number forms in the index: a second FTS column with `numbers.source_numbers_in`
   readings of each row (es "354 mil" → 354000), searched, not shown.
2. The narrow-window browser check of the Search page (420 and 800 px).
3. The 1.1.0 bump: done 2026-10-27 (release-1.1.0).

## 2026-10-28 · Phase 8a kickoff prompt (no code)
- PLAN §21 "Kickoff prompt for Phase 8a" and a §9 line: cross-language citation times
  (post-v1 item 2). Picked over item 1 (long-en made-up tasks) because it's pure Python with no
  prompt change, users see it (and Search's Open notes since 1.1.0), and lang + quick measure it.
- Found while writing it: **Map writes in the notes language** (clean-hi `map.json` is Hindi),
  so the post-v1 candidate "refine on the English Map item" has no English text to use. The
  prompt asks step 0 to replay `refine_time` offline on the stored lang runs first, then pick
  language-neutral anchors (numbers read per language, owners through anyascii) or argue for more.
- Baseline-lang citation accuracy: clean-hi 58.3%, **xling-ja 70%** (English notes, Japanese
  transcript: same cause the other way round), the other five 100%.

## 2026-10-28 · Phase 8a: cross-language citation times
Branch `phase-8a-jump-times`. No prompt change (REDUCE_HASH unchanged), no FORMAT_VERSION bump.

### 0. Measurement and proposal (approved as proposed)
- Offline replay (D19's approach): rebuilt notes from each stored lang run's `raw` through
  `build_notes` + `restore_owners` with an instrumented refiner. Reproduces the stored notes
  exactly, so its before/after equals `--rerun reduce`.
- What each wrong citation shares with its true chunk: **clean-hi**, the 5 action items: only
  the owner, in Devanagari (`[owner: लॉरा]`); its numbers already land (digits overlap).
  **xling-ja**: 4650 is shared but scores 0.08 (long span, < 0.1); Tom is said as トム; MARIA
  is on the right chunk (0.11) but placed 21.8 s early (below).
- Names in claim text, outside `[owner: …]`, aren't usable anchors: through D34's consonants
  تم "was done" = Tom, مللي "milli" = Maria, سرعة "speed" = "Sorry" in the stored runs.
- Two traps the replay caught in the prototype: an owner resolved per chunk (मारिया matched
  Omar, the only m-r name in that chunk; resolve once per segment), and substring names ("Tom"
  in "cus**tom**ers"; whole words, but kana next to "MARIAさん" is no word character).

### 1. Fix (D39)
- `reduce.refine_time`: unchanged while a chunk scores ≥ 0.1; otherwise the chunk with the
  most anchors (span numbers in the notes' language, digits folded, vs `source_numbers_in` per
  chunk; the `[owner: …]` restored for the segment and matched by `names.said_in`), ties by
  word overlap then earliest, `t` at its sentence with the most anchors; segment start when
  nothing anchors. `segment_refiner` takes the notes' and meeting's languages (Reduce:
  `notes_language`, `source_language`; Ask: the answer language).
- Tests: Hindi claim by owner and by a Bengali-digit number, English claim on Japanese text by
  number and by katakana; must-not-fire: an overlapping English claim keeps today's time, a
  number in two chunks loses to the chunk the words match, no chunks / no overlap at
  `min_score` 0 keep the segment start (both crashed in a first draft).

### 2. Measured
| | before | after |
|---|---|---|
| lang seed 1, citation accuracy | 89.9% | **98.7%** |
| clean-hi | 58.3% | **100%** |
| xling-ja | 70% | **90%** |
| other lang items | 100% | 100% |
| lang recall / made-up items | 86.8% / 0 | unchanged |
| quick (gate) | pass | pass, every metric identical |
- Segment-start citations (all, not only planted facts): clean-hi 14 → 9, xling-ja 12 → 9,
  xling-es 9 → 7, conflict-de 4 → 3, clean-ar 7 → 7 (no digits or owners in its fallbacks).
- quick: no citation falls back, so the anchors never run there. long-en not run: two-pass
  code unchanged and English on English.
- clean-hi with Ask and the judge: Ask accuracy, citations, cross-language alone and with
  history all 100% (19 questions). xling-es Ask is skipped by the eval (es meeting,
  English-only questions), so Ask's cross-language path is covered by clean-hi only.
- `baseline-lang.json` rewritten (seed 1, `--no-ask`, as before).

### Open
- **xling-ja MARIA, −21.8 s:** word overlap finds the right chunk, but Transcribe wrote 28 s
  of Japanese without 。, so the "sentence" starts at the chunk start. A fix would change
  placement on the path every language uses (e.g. place at the anchor inside a long
  unpunctuated sentence); measure on quick and long-en first. Added as post-v1 item 10.
- Claims with no number and no owner (summaries, decisions in Hindi or Arabic) still cite
  the segment start, by design.

- 1.2.0 bump: done 2026-10-28 on request (release-1.2.0).

## 2026-10-28 · Phase 8b kickoff prompt (no code)
- PLAN §21 "Kickoff prompt for Phase 8b" and a §9 line: made-up action items on long meetings
  (post-v1 item 1, merged with item 6). Picked over item 10 (ja chunks without 。), which moves
  placement on every language's path for one citation.
- Checked the post-v1 candidate on the stored run before writing (CLAUDE.md lesson): "no owner,
  no due" fits only 3 of long-en's 11 made-up action items. 9 have no owner, and 6 of those 9
  carry the due "end of next week", which their cited segment doesn't say (1 of long-en's 7
  owner-less dues is said there). AMI: 18 owner-less dues, none said in a cited segment.
- Cause candidate: map.md's example due is "end of next week" (line 12), and
  evals/gen/script.py plants it (RELATIVE_DUES), so Map copies the example and long-en can
  reward it. map.json has it on 22 lines in AMI; the AMI transcripts say "next week" once.
- quick and lang stored runs have no owner-less action items except one in conflict-de, so
  neither set measures this; the prompt asks for an offline replay on long-en and AMI
  (`evals.made_up` is judge-free), then an AMI-only run (ask) for coverage.

## 2026-10-28 · Phase 8b: made-up action items on long meetings
Branch `phase-8b-made-up-tasks`. Post-v1 items 1 and 6; D40.
- Step 0 (stored runs, no models): every real long-en due (12) is said in its cited segment;
  the 6 copied "end of next week" dues are said only elsewhere in the meeting, and "1
  November" is "11 November" in its segment. AMI: 1 of 39 dues said where cited. A
  `due_key ⊆ words` check over a whole segment was too loose ("End of meeting" matched "end"
  and "meeting" far apart), hence the window (`dues.said`, 4 extra words).
- Rejected filters: an ask-cue check on the cited sentence (long-en's "Let's move on to…"
  trips it; AMI's ASR has no sentence breaks); "no owner and no due" alone (9/9 on long-en,
  but 25 of AMI's 47 tasks: it fits the generator, where every planted task has an owner).
- Fix on the final notes, after `restore_owners`: `guard_dues` then `move_vague_tasks`
  (English notes only, Continue/Ensure/Keep/Monitor/Maintain). Placeholders ("Not
  specified", "TBD") parse as unknown. Cross-language: digits only; clean-hi's Hindi dues and
  conflict-de's German ones stay. `due_key` and `due_in` moved from evals to `dues.py`.
- Eval: `_stated_due` reads a due written in the task text when the field is empty, so the
  guard clearing "1 November" leaves "Summarise competitor research by 1 November" wrong, not
  a match.
- Offline replay (one-pass runs rebuilt from `raw`; long-en's steps on the stored notes)
  matched the model runs item for item.

| | before | after |
|---|---|---|
| long-en made-up items | 12 | **7** |
| long-en recall / support rate | 67.5% / 75.7% | 67.5% / **79.4%** |
| AMI coverage (5 meetings) | — | unchanged on every meeting |
| AMI support rate | 69.6–88.6% | up in 4 of 5 (TS3003c 69.6 → 73.9%) |
| AMI dues | 39 | 1 (34 unsaid dropped, 4 placeholders cleared) |
| quick (gate), lang seed 1 | — | every metric identical, guard never fires |

- Runs: quick and lang seed 1 `--no-ask` with `--rerun reduce`; long-en `--rerun reduce`
  (Map reused); AMI items of the full set `--rerun reduce` with Ask and the judge.
- `baseline-long.json` rewritten (made-up items 12 → 7 on purpose). quick, lang and full
  baselines unchanged (identical, or AMI report-only).

### Open
- Map still copies "end of next week" into map.json, and Ask's segment snippets show it
  (`_action_line`). Fixing it means changing map.md's example (seeds 1–3, every set) or the
  guard on Map's items (Reduce input moves: a long-en run).
- 5 long-en made-up items remain: "Explain quarterly budget reporting", "Update revenue
  dashboard", "Review and clarify…", "Address the support queue backlog", "Standardise on
  PostGIS" (a decision). No source signal separates them from real owner-less tasks yet.
- "Summarise competitor research by 1 November": the text still says 1 (said 11). A digit
  repair in task text, like Map's number guard, is the candidate.

- 1.3.0 bump: done 2026-10-28 on request (release-1.3.0).

## 2026-10-28 · Phase 8c kickoff prompt (no code)
- PLAN §21 "Kickoff prompt for Phase 8c" and a §9 line: Ask on long meetings (post-v1 item
  7). Picked over item 3 (Map task-as-decision, a prompt rule on seeds 1–3) and item 10 (ja
  chunks without 。, moves placement on every language's path): Ask is what users query, and
  its misses share causes with 8b's "Open".
- Checked the post-v1 candidate on the stored run before writing (CLAUDE.md lesson): ranking
  isn't the cause. `ask.rank` puts the answer's segment at rank 0, 0, 0 and 2 for the 4
  misses (identical in the 6e-2 and 8b reports, cached replies unchanged).
- The misses: a-f26 right answer, citation −197.7 s (R7B cited `<co>Tom</co>` alone first;
  `refine_time` puts "Tom" at 1820.88, the full span at 2019.56). a-f61: 8 of the 12
  top-ranked documents carry a `[due: end of next week]` Map line (map.md's example), and R7B
  answered Maria's. a-f82: Map wrote "November 1" (said "11 November"), R7B answered another
  Grace task. a-f15: ASR wrote "Oma", Map missed the lunch task, R7B answered Omar's Map line.
- So the prompt asks step 0 to measure name-only spans across the stored notes too (the same
  `resolve` path places Reduce's citations), and to weigh a due/date check on the Map lines in
  Ask's documents against changing map.md's example.
- Steer added after review: the Map-values fix goes on Map's items (map.json) rather than only
  on Ask's snippets, so every reader sees one value and 8b's "1 November" task is reached too
  (user).

## 2026-10-28 · Phase 8c: Ask on long meetings
Branch `phase-8c-ask-long`. Post-v1 item 7; D41.
- Step 0 (stored runs and cached replies, no models): ranking isn't the cause (rank 0, 0, 0,
  2). a-f26: `<co>Tom</co>` then the task, both citing seg 6; the first span placed it at
  Tom's first chunk (−197.7 s). a-f82: Map wrote "November 1" three times in seg 26 (said
  "11 November"); the number guard kept it because the segment says 1 elsewhere. a-f61: 6 of
  the 9 "end of next week" Map lines sent were owner-less copies of map.md's example. a-f15:
  "Oma … lunch" is in seg 4, where Map wrote no task and which wasn't sent.
- Multi-span claims: 33 in the cached Ask replies (7 placing > 15 s apart), 179 of 541
  one-pass notes claims (139 apart, mostly AMI one-word spans like "meeting"). Replayed
  (rebuild matched the stored notes item for item): best span fixes a-f26 and lifts clean-en
  citations 93.75 → 100%, nothing drops on any item.
- Map's dues checked on every stored run: fires only on long-en's 7 copies, "November 1" and
  conflict-de's made-up "Do. 4:44" (said "end of next week"); AMI 44 of 45 (D40's drops).
  The date check fires only on seg 26's three "November 1"s.
- Fixes: `reduce.best_span` (Refine takes every span citing the segment; one rule for
  `resolve`, `pass2_resolver` and Ask). `mapper.guard_dates` after the number guard:
  `dues.check_dates` repairs a date one digit edit from exactly one said date of its month,
  else flags it; `dues.due_said` (shared with `reduce.guard_dues`) drops an unsaid due.
  MapFile v5 (`dropped_dues`). map.md's example left as is: the check removes every copy.
- Map reproduced its stored output field for field on every item at the same seed (diffed
  against copies of the runs, outside the guarded fields). The new field moves every Ask
  cache key (`sources_hash`), so all answers were re-asked.

| | before | after |
|---|---|---|
| long-en Ask accuracy / citation | 77.8% / 93.3% | **88.9% / 100%** |
| long-en made-up items | 7 | **4** |
| long-en support rate | 79.4% | **82.0%** |
| long-en recall, notes citation accuracy | 67.5%, 98.1% | unchanged |
| quick citation accuracy | 98.4% | **100%** (clean-en 93.75 → 100) |
| quick other metrics, Ask | — | identical (76 answers re-asked) |
| lang seed 1 | — | identical but conflict-de support: 1 partial → no |

- Runs: quick (Ask and judge), lang seed 1 `--no-ask`, long-en (Ask and judge); Map re-ran
  by the format bump (+145 s on long-en). `baseline.json` and `baseline-long.json`
  rewritten on purpose; `baseline-lang.json` unchanged.

### Open
- a-f15 stays a miss: ASR wrote "Oma", Map has no task in seg 4, so it isn't sent. Only Map
  recall reaches it.
- a-f61 still answers Maria's task: the copies are gone, but 4 real "end of next week" tasks
  are sent (Maria ×2, Omar, Grace). An Ask prompt rule ("answer for the person named") is the
  remaining candidate; it moves every cached answer.
- long-en `ask_xlang_history` 1 → 0: one question. a-f4's answer changed wording (its documents
  lost copied dues), so de-history's history changed; both before and after R7B listed every
  metric instead of answering, and the old list happened to include 4,550.
- conflict-de: with Map's made-up due gone, R7B wrote Tom's task without "bis Ende nächsten
  Woche" and the judge called the shorter German claim unsupported (its phrase "not in the
  cited text").
- `pass2_group` 0 → 5 on long-en (R7B's new pass-2 spans matched no pass-1 claim); notes
  citation accuracy unchanged.

- 1.4.0 bump: done 2026-10-28 on request (release-1.4.0).

## 2026-10-29 · Phase 8d kickoff prompt (no code)
- PLAN §21 "Kickoff prompt for Phase 8d" and a §9 line: missed action items (post-v1 item 3,
  widened). Picked over item 10 (ja chunks without 。, moves placement on every language's
  path for one citation) and 8c's a-f61 Ask rule (moves every cached answer for one question):
  action recall is the core metric and item 3 was passed over twice.
- Checked the post-v1 candidate on the stored runs before writing (CLAUDE.md lesson): Map
  writing a task only as a decision is 2 of the 17 action items missed in the latest quick,
  lang and long reports (long-en f22 "Tom will update the pricing page", xling-es f8 Victor).
  The rest: 3 Map tasks with no owner (clean-sw, clean-bn, conflict-de), 5 Map tasks the
  final notes lost or changed (long-en f31, f40, f82; clean-ar f17; conflict-de f12), 7 Map
  never wrote (5 on long-en). So the phase covers all three Map-reachable groups, Python first.
- Scan of every stored map.json for "<subject> will/to/should …" decisions with no Map task in
  the segment: 22, of which 17 AMI and 3 xling-ja are product decisions with no person. A
  rule needs the subject to be a name said in the meeting; on the stored runs that fires on
  exactly the 2 real cases.
- long-en notes.json keeps no pass-1 notes, so which pass lost f31 and f82 needs a replay.

## 2026-10-29 · Phase 8d: missed action items
Branch `phase-8d-missed-tasks`. Post-v1 item 3, widened; D42.
- Step 0 (stored runs, no models): the kickoff's 17 misses held. long-en's pass-1 replies
  *are* stored (`notes.json` `two_pass.groups[].raw`), so no replay was needed to place them:
  f82 (Grace) was a correct owned item in pass 1 that pass 2 dropped; f22 (Tom) a pass-1
  decision no Map task matched (D33 can't move it), dropped by pass 2; f31 (Laura) a Map
  task with its owner inside the text ("Laura will clean up …", owner null), never in pass 1.
  clean-ar f17 is a one-pass R7B omission of a Map task; conflict-de f12 an `[owner: ?]` line
  whose Map task names Tom. f40 is ASR ("Omar Crews read" for "Omar proofread"), not notes.
- The cited sentences of the owner-less tasks: clean-sw/bn "Maria, can you prepare the customer
  survey …?"; conflict-de "Grace will write the release notes by Wednesday." (German Map task:
  no word overlap with it, so left open). xling-es says "Víctor", which is why the scan's name
  check skipped Victor: names are compared with accents folded.
- `said_names` keeps "Okay", "Thanks", "Action", "Monday", "I": counting names anywhere in a
  sentence would misfire, so the owner comes only from a cue position ("X, can you", "X will",
  "action item for X"), and weekdays, months and pronouns are no one.
- Fixes (user: all four, changes logged with no MapFile field): `mapper.guard_owners` (A1
  decision → task, A2 owner from the task text or its one-person sentence), `reduce.fill_owners`
  (B1, merged into a duplicate the owner already has: long-en's pass 2 wrote Laura's board
  presentation twice) and `reduce.add_lost_tasks` (B2). B1 and B2 place their citations with
  an `[owner: …]` span: without it clean-ar f17 landed −48 s and conflict-de f12 −223 s
  (D39 anchors only on owners and numbers); the anchors fold accents too (exact match first:
  folding drops katakana voicing marks).
- Replay first (17 stored runs rebuilt to the stored notes item for item): 9 of 17 found,
  made-up unchanged, AMI 0 changes in decisions or action items. Then the runs matched it.

| | before | after |
|---|---|---|
| lang seed 1 action recall / fact recall | 80.0% / 86.8% | **94.3% / 92.3%** |
| lang made-up items, citation accuracy | 0, 98.7% | 0, 98.8% |
| long-en action recall / fact recall | 55.0% / 67.5% | **70.0% / 71.3%** |
| long-en made-up items | 4 | 6 (see Open) |
| long-en citation accuracy, support, Ask | 98.2%, 82.0%, 88.9% | 98.3%, 81.9%, 88.9% |
| quick (Ask and judge) | — | identical |

- Runs: lang seed 1 `--no-ask` and quick with `--rerun map` (Map reproduced its output at the
  same seed), long-en `--rerun map` then `--rerun reduce` for the merge. `baseline-lang.json`
  and `baseline-long.json` rewritten on purpose; `baseline.json` unchanged.
- A cut reply (no ACTION_ITEMS section) now gets Map's owned tasks back (B2); the cut-reply
  test says so.

### Open
- long-en made-up 4 → 6: "Review and clarify the reporting of cloud costs" and "Address the
  support queue backlog", owner-less pass-1 tasks (seg 25) in both runs; the old pass 2
  dropped them, the new one (different pass-1 input from group 0) keeps them. Replayed with
  D42 off, still 6.
- conflict-de f6: a German Map task of an English meeting; its sentence can't be found by
  words. Needs the task's time or a cross-language match.
- Map never wrote 7 (long-en f14, f15, f19, f71, f86; noisy-en f7; conflict-de f5): Map recall,
  seg 4 densest (3 tasks and a decision, none written). f40 is ASR.
- B2 inserts in segment order before the first item citing a later segment; pass 2's own
  order isn't chronological, so additions can sit mid-list.

- 1.5.0 bump: done 2026-10-29 on request (release-1.5.0).

## 2026-10-29 · Phase 8e kickoff prompt (no code)
- PLAN §21 "Kickoff prompt for Phase 8e" and a §9 line: action items Map never wrote (8d's
  "Open"), with long-en's made-up 4 → 6 measured alongside. Picked over item 10 (ja chunks
  without 。, moves placement on every language's path for one citation) and items 4 and 5
  (Japanese only, no eval set): action recall is the core metric, and 8d's Python on Map's
  output reached all it could.
- Checked the candidate on the stored runs before writing (CLAUDE.md lesson): "dense segments"
  is wrong. long-en segments are 261–299 s and 726–864 words each; seg 4 wrote 0 of 3 tasks,
  seg 6 all 3; a miss's position in its segment doesn't predict it.
- All 8 misses (long-en f14, f15, f19, f71, f86; noisy-en f7; conflict-de f5, f6) are
  cue sentences. A scan of `names.addressee` against each segment's Map owners reaches 7.
  noisy-en f7 is lost to "F site" (F reads as a second person). The scan also flags "Okay,
  will do." (9 runs; "okay" isn't in `_NOT_PEOPLE`), "Laura Schedule" (capitalised verb read
  as a surname) and all of clean-hi (Devanagari owners compared as Latin).
- AMI has 0 name-cued sentences, so it can't test the rule's must-not-fire side; the prompt
  asks for must-not-fire tests from real speech and a seed 2/3 check on quick before choosing
  between a Python net, a small model call on the flagged sentences, and a Map prompt change.

## 2026-10-29 · Phase 8e: action items Map never wrote
Branch `phase-8e-map-recall`. 8d's "Open"; D43.
- Step 0 (no models): with three fixes in the scan ("Okay" and other replies are no one, a
  one-letter word is no one, owners compared by name: by consonants Omar = Maria = "mr"),
  the scan flags exactly the 8 misses on the 17 stored runs, 0 elsewhere (AMI: no sentence
  names anyone). The stored quick runs are pipeline seed 0 (`Config.seed`'s default, item
  seeds 1–4 are the generator's); `-c seed=N` changes only the pipeline seed, media stay.
- Map's misses change with the seed: quick action recall 95 / 80 / 75% at seeds 0 / 2 / 3
  (Omar's hiring plan written at 0, dropped at 2 and 3; Nadia, Grace, Maria likewise). Every
  missed action fact at every seed was a flagged sentence. A prompt change would move which
  tasks drop; a net catches whichever do.
- long-en's 2 extra made-up tasks (seg 25) don't share the cause: Map wrote them from filler
  ("the way cloud costs is reported is a bit confusing"), and no sentence names anyone.
- Fix (user: scan + model over a Python-only net and a map.md rule): `names.addressee` fixes
  above, plus a two-word name cut to its first word when that is a person ("Action item for
  Laura Schedule …"); `mapper.uncovered_cues` → one `map_tasks.md` call per flagged segment,
  temperature 0; Python owns the owner and the values. `map.cue_tasks` switches it off.
- Found while measuring: Tiny Aya closed noisy-en's one entry with `}}}` (each `{"line": …}`
  is now read alone; a temperature-0 retry would repeat it); it judged conflict-de's "Grace
  will write the release notes by Wednesday." no task when writing German (the same sentence
  in English is one); and it cut f86 to "order laptops". Rewording the prompt ("with every
  detail") left f86 short and put Nadia's name into the German task, so it was reverted and
  `mapper.whole_task` completes the task from the sentence (English notes, words said
  verbatim; replayed on every stored English run, it changed only f86).
- Map owners "Omar Crews" (ASR, f40) cover Omar: an owner's first word counts as covered.

| | before | after |
|---|---|---|
| quick action recall, seeds 0 / 2 / 3 | 95.0 / 80.0 / 75.0% | **100 / 100 / 100%** |
| quick seed 0 fact recall, Ask, citation accuracy | 89.7%, 98.1%, 100% | 91.2%, 98.1%, 100% |
| quick made-up items, wrong owner/due (3 seeds) | 0 / 1 / 0, 0 | 0 / 1 / 0 (same item), 0 |
| lang seed 1 action recall / fact recall | 94.3% / 92.3% | **97.1% / 93.4%** |
| long-en action recall / fact recall | 70.0% / 71.3% | **90.0% / 78.8%** |
| long-en Ask, citation accuracy | 88.9%, 98.3% | **94.4%**, 98.4% |
| long-en made-up items | 6 | 7 (f15: owner "Oma", ASR for Omar) |
| AMI | — | 0 flagged sentences: no call, no change |

- Runs: quick seeds 0/2/3 and lang seed 1 with `--rerun map --no-ask`, then the final quick
  (Ask and judge), lang seed 1 `--no-ask` and long-en, each `--update-baseline`. All three
  baselines rewritten on purpose (Map now writes the tasks it dropped).
- Long runs cost: 4 of 30 long-en segments make the extra call; quick 0–2 per meeting.

### Open
- conflict-de f6: the model says Grace's sentence gives no task when writing German; Map's
  own German task has no owner and no word overlap (8d).
- f15 and f40: the transcription's names ("Oma", "Omar Crews read" for "Omar proofread").
- Cues are English only (xling-es/ja have none); Japanese and Spanish meetings get no net.
- The 5 long-en made-up items from 8b/8d remain (seg 25's two are Map writing filler as tasks).

- 1.6.0 bump: done 2026-10-29 on request (release-1.6.0).

## 2026-10-29 · Phase 8f kickoff prompt (no code)
- PLAN §21 "Kickoff prompt for Phase 8f" and a §9 line: missed decisions. Picked over
  long-en numbers (77.5%) and made-up items (7), which show only on the 40 min long set, and
  the Japanese items (4, 5, 10): decisions are the weakest kind in every set (quick 75%,
  long-en 70%, lang 21/28 judge-covered) after 8b–8e took action recall to 90–100%.
- Kickoff check on the stored runs (CLAUDE.md lesson): of quick seed 0's 4 misses and
  long-en's 6, 7 are in no map.json, clean-en f2 is a Map key point, long-en f49 a Map
  decision the notes lost, and long-en f55 is ASR ("PostGIS"). All are planted in three
  cue forms; a scan of those against each segment's Map decisions flags exactly the English
  misses. noisy-en f3 and slide-conflict-en f2 are missed at seeds 0–3, so unlike 8e a
  prompt change isn't ruled out.
- AMI has 6 such sentences in 5 meetings, none a missed decision (4 already in Map, 2 refer
  back), so a cue rule fits the generator; the prompt asks for must-not-fire tests from real
  speech. lang misses need a hand read of map.json (Map writes the notes language).

## 2026-10-29 · Phase 8f: missed decisions
Branch `phase-8f-decisions`. D44.
- Step 0 (no models): the cue scan on all 17 stored runs flags exactly the English misses, 0
  elsewhere; every cue in the generated sets is a planted decision. 6 of the 18 misses were
  Reduce's, not Map's: long-en f49 (pass 1's group reply has no decision citing seg 13),
  clean-sw/bn f15 (Map's English "Weekly sync moved to Thursdays." filed as a key point),
  clean-ar f15/f18 (Map has both in Arabic; the notes kept no seg-1 decision). clean-en f2 is
  in a bundled Map key point. Map never wrote noisy-en f3, slide-conflict-en f2/f14, long-en
  f18/f27/f30/f33, conflict-de f2/f14 and clean-ar f12. long-en f55 is ASR ("PostGIS").
- AMI: the three cue forms occur once in 5 meetings (ES2004d's kinetic charger, already a Map
  and notes decision). Real speech says "we decided upon …", "we decided that …", "Do we agree
  on that?", "I think we all agreed on that.": a rule keyed to the forms fits the generator, so
  it only adds where a cue is uncovered, and AMI is its must-not-fire evidence.
- Re-adding every Map decision the notes lack would add 8 AMI items (several not decisions:
  "Discuss the feasibility of kinetic energy…") and long-en's "Leave Bitcoin acceptance
  decision open": Map's decisions list holds tasks and open questions too. Rejected.
- Fix (user: Python over a model call or map.md change): `reduce.decision_cues` +
  `reduce.add_said_decisions`, after `add_lost_tasks`. English notes: the sentence's own words,
  or the key point that states only the decision moves; a bundle stays. Other notes languages:
  the segment's unstated Map decisions up to its cue count, skipping Map's tasks and open
  questions. Code only: `--rerun reduce`, REDUCE_HASH and Ask keys unchanged.
- Found in the replay: in Arabic notes Refine has nothing to match, so both clean-ar additions
  sat at seg 1's start (roadmap 55 s early). When no decision cited the segment, each addition
  now takes its cue sentence's time (297.8 s, 346.0 s).
- The offline replay on the stored notes.json (the steps after it don't touch Decisions or Key
  points) matched the eval runs item for item.

| | before | after |
|---|---|---|
| quick decisions, seeds 0 / 1 / 2 / 3 | 75 / 81.25 / 81.25 / 87.5% | **100 / 100 / 100 / 100%** |
| quick seed 0 fact recall, Ask, citation accuracy | 91.2%, 98.1%, 100% | **97.1%**, 98.1%, 100% |
| quick made-up items, seeds 0–3 | 0 / 0 / 1 / 0 | same |
| quick open questions, seeds 0–3 | 100 / 80 / 80 / 100% | same |
| lang seed 1 judge-covered decisions, judge recall | 21/28, 75.0% | **24/28, 83.3%** |
| lang fact recall, citation accuracy | 93.4%, 98.8% | same |
| long-en decisions, fact recall | 70%, 78.8% | **95%, 85.0%** |
| long-en Ask, citation accuracy, made-up, wrong owner/due | 94.4%, 98.4%, 7, 2 | 94.4%, 98.5%, 7, 2 |
| AMI | — | replay: 0 changes (no AMI run) |

- Runs: quick seeds 1–3 `--rerun reduce --no-ask --no-judge`, quick seed 0 with Ask and the
  judge, lang seed 1 `--no-ask`, long-en, each `--rerun reduce`; then each `--update-baseline`.
  All three baselines rewritten on purpose (decisions the notes lost are now added). quick
  seed 1's action recall 95 → 100% predates this change (its earlier report was before 8e's fix).
- Cost: none (no model call).

### Open
- conflict-de f2/f14 and clean-ar f12: Map never wrote them, and in German/Arabic notes Python
  can't tell which English cue is uncovered (a model call could; Tiny Aya misjudged German cue
  sentences in 8e).
- clean-sw f15: the English notes have it; Tiny Aya translated "sync" as "sasisho" (update) and
  the judge didn't count it. Translation, beta language.
- long-en f55: ASR ("PostGIS" for Postgres), counted as made-up too.
- Real meetings' decision forms ("we decided upon/that …") are left to Map and R7B.
- Seed-1 "before" in the table is 8e's mid-phase report; spare copy of the pre-8f stored runs in
  evals/data/runs-pre-8f (gitignored, safe to delete).

- 1.7.0 bump: done 2026-10-29 on request (release-1.7.0).

## 2026-10-29 · Phase 8g kickoff prompt (no code)
- PLAN §21 "Kickoff prompt for Phase 8g" and a §9 line: missed numbers (post-v1 item 4,
  widened). Planned as a digit repair (item 4 and 8b's "1 November"), but the stored runs
  show that work mostly done: 8c's `guard_dates` fixed "1 November" (the notes say 11), and
  Map's guard repairs 30+ miscopied values. One wrong digit is left (xling-ja f9: "$34,000"
  is one edit from both 354,000 and 134,000, so it's flagged). User's choice: widen to every
  missed number fact, the weakest kind on long-en (31/40).
- Kickoff check on the latest reports (CLAUDE.md lesson): of 17 missed number facts, Map
  never wrote 8, Reduce lost 6 that are in Map's key points (long-en f43/f66/f79,
  slide-conflict-en f7, clean-ar/hi f16), 1 is the digit, and xling-ja f6 is ASR. Quick's
  number misses move with the seed (seeds 0–3), so a map.md change would only move them.
- AMI IS1009b's one repair is wrong: "the rest" (60%) became "40% regular buttons", one digit
  edit from the 40 said. The prompt asks step 0 to check AMI's repairs before widening D20.
- long-en's Map writes exactly 3 key points in 17 of 28 segments (quick/lang 4–8); cause unknown.

## 2026-10-30 · Phase 8g: missed numbers
Branch `phase-8g-numbers`. Post-v1 item 4, widened; D45.
- Step 0 (no models) confirmed the kickoff's 17: Reduce lost 6 that Map wrote as key points
  (long-en f43 and f66 in pass 1, f79 in pass 2; slide-conflict-en f7 kept "$80,000" and
  dropped "(slide: $18,000)"; clean-ar/hi f16), Map never wrote 8, xling-ja f9 is the digit,
  f6 is ASR. Where Map never wrote one, it still wrote a key point on the topic ("Quarterly
  budget figures are steady"), just without the figure.
- Repairs: 41 stored, AMI's only one (IS1009b 60 → 40) wrong; the other one- and two-digit
  repairs are days (pdf-slides "2nd" → 22nd). So no plain number under three digits is
  repaired now. AMI says 33 digit numbers no Map item or notes state, mostly scoring
  fragments ("35.", "50?", "Do 5."); a filtered sentence net still fired 5 times on AMI (2
  garbled) and on every "That's compared with …" sentence of long-en: not built.
- f9 has three candidates, not two: "34%" (×1000) too. Dropping the one the claim states
  (134,000) leaves two. A tiebreak (one digit edit beats a factor) fixed it, but at quick
  seed 1 it repaired pdf-slides' "6,900 candidates" (69 said, ×100) to the $8,900 cloud bill,
  and the eval matched the cloud-bill fact to that claim (citation −109 s). Dropped; f9
  stays flagged. The stated-candidate rule stays (no wrong repair on any run).
- `reduce.add_lost_numbers` after the decisions. The replay on the stored notes.json and the
  eval runs added the same 7 key points (6 eval facts + AMI ES2004d "50% profit"). The
  replace path kept Reduce's seg-only citation, so the judge read "(slide: $18,000)" against
  seg 0 and said no; the step now also cites the segment's slides showing an added figure,
  and the judge then said the slide contradicts $80,000 (the conflict note, as §5.5 writes it).
- Code only: Map's text changes only where the guard does (IS1009b seg 6), so Ask keys move
  there only; REDUCE_HASH and MapFile unchanged.

| | before | after |
|---|---|---|
| quick numbers, seeds 0 / 1 / 2 / 3 | 93.8 / 93.8 / 93.8 / 96.9% | **96.9** / 93.8 / **96.9 / 100%** |
| quick seed 0 fact recall, Ask, citation accuracy | 97.1%, 98.1%, 100% | **98.5%**, 98.1%, 100% |
| quick seed 0 judge support | 95.1% | 93.8% (the conflict note) |
| quick made-up items, unsupported numbers, seeds 0–3 | 0 | 0 |
| lang seed 1 numbers, fact recall, citation | 91.1%, 93.4%, 98.8% | **94.6%, 95.6%**, 98.9% |
| lang unsupported, unverified | 1, 3 | 1, 3 |
| long-en numbers, fact recall | 77.5%, 85.0% | **85.0%, 88.8%** |
| long-en Ask, citation, made-up, wrong owner/due | 94.4%, 98.5%, 7, 2 | 94.4%, 98.6%, 7, 2 |
| AMI (first Map run since Oct 14) | 52.8% coverage | 44.9% (not 8g: see Open) |

- AMI, 8g's part: 6 key points added (ES2004b's survey figures "80% would spend more…",
  "50% only use 10% of the buttons", "34% take too long to learn", "35 and under…";
  ES2004d "50% profit"; IS1009b "€25"), all said; IS1009b seg 6 keeps "60%" and flags it, so
  ami_unsupported_numbers 0 → 1 ("the rest" has no digit).
- Runs: quick seeds 1–3 `--no-ask --no-judge`, quick seed 0 with Ask and the judge, lang seed
  1 `--no-ask --rerun map`, long-en `--rerun reduce`, AMI (`--item ami --no-ask`); seed 1 and
  lang rerun after dropping the tiebreak. quick, lang and long baselines rewritten on purpose
  (figures the notes lost are added); baseline-full.json not (no full run).
- Cost: none (no model call).

### Open
- AMI coverage 52.8 → 44.9% (decisions 52.2 → 30.4%): the first Map run on AMI since Oct 14,
  so 8b–8e's Map steps (only replayed on AMI) and 8g together. IS1009b drove most of it: R7B
  wrote a second block under "## <co>Remote Control Design</co…>" after OPEN_QUESTIONS, and
  §5.6's fallback filed it as Summary (34 summary claims, 5 key points). With 8g's "60%" put
  back to "40%" the reply is the same, so 8g didn't trigger it. Needs its own look: which
  Map change moved the prompt, and whether a heading after the last marker should continue
  the section before it.
- Map's 8 never-written numbers (long-en f8/f11/f16/f45/f50/f70, pdf-slides f12, xling-ja f7).
- xling-ja f9 flagged ("34%" is a third candidate); a unit check (% vs $) could settle it.
- The judge reads a "(slide: …)" conflict note as a contradiction.
- long-en's Map writes exactly 3 key points in 17 of 28 segments: cause still unknown.

- 1.8.0 bump: done 2026-10-30 on request (release-1.8.0).

## 2026-10-30 · Phase 8h kickoff prompt (no code)
- PLAN §21 "Kickoff prompt for Phase 8h" and a §9 line: AMI coverage (8g's "Open"; user's
  choice over Map's never-written numbers, because AMI is the only set evals/gen didn't make
  and it went down).
- Kickoff check on the stored runs (CLAUDE.md lesson): IS1009b's reply changed shape. R7B
  wrote all five sections for segment 0 only, then segments 1–6 as one flat list under
  "## <co>Remote Control Design</co: …>" (14 lines twice), all filed as Summary. That flat
  list holds key points and decisions both, so "continue the section before it" (8g's
  candidate) would have filed them as open questions.
- Cause of the flip: Reduce's prompts are unchanged since Oct 13; its input lost 44 tokens
  (8c's dropped dues, written `[due: ?]`). The fallback fired in 3 of IS1009b's 7 Reduce runs,
  and both earlier ones ("#", Oct 14) are the runs where it scored 56.2% instead of 62.5%.
  No other stored run has an unknown heading.
- The other four meetings kept the shape and moved by one or two claims; one reference
  sentence is 11–17 points, and TS3003c scored 22.2% and 55.6% in Oct 14 runs. Step 0 splits
  notes changes from noise with the cached judge.
- The pre-Oct-30 AMI runs were only in an old session's /tmp scratchpad; copied to
  evals/data/ami-before-8g/ (git-ignored) for step 0.
- IS1009b's s/h 2056 in 8g's report is wall-clock: Reduce took 77.7 s but started 17 minutes
  earlier (another run held the GPU).

## 2026-10-30 · Phase 8h: AMI coverage
Branch `phase-8h-ami`. 8g's "Open"; D46, D47.
- Step 0 (no models): every judge answer for both states was cached, so the Oct 28 notes
  (evals/data/ami-before-8g/) and 8g's were scored offline: 47 → 40 of 89, as reported. Net
  IS1009b −3, ES2004d −2, ES2004b −1, TS3003c −1. No Python step moved a lost sentence (8g's
  figures only added; the decision → action move touched none).
- Noise: ES2004d's problem sentence flipped when two open questions lost their "?" and
  nothing else changed; its lost decision went with a claim that doesn't state it. TS3003c's
  battery claims were still there; the judge flipped on the list around them.
- Shared cause: R7B's one-pass replies over AMI's 7–9 segments leave whole segments out of a
  section, and some loop. IS1009b wrote no decisions for seg 1 and no questions for segs 3/5/6;
  TS3003c wrote segs 4–6's decisions three times and none of its own for segs 7–8; ES2004b no
  key points for segs 3–4. The Oct 28 IS1009b reply had it too (no seg 2, seg 5's questions
  three times). Never on the 2-segment eval items or long-en's groups. Map's output is the
  same as Oct 14 apart from dues, so restoring `[due: ?]` wouldn't fix it.
- IS1009b's dump is Map's key points word for word: filing it by Map item gives 20 key
  points and 2 decisions, a cleaner page but none of the lost decisions back. Hence the
  second step on the final notes.
- D46 (parser): list items under an unknown heading filed by the cited segments' best Map item
  (≥ 0.6), prose stays Summary; a repeated claim (same text and citations) kept once.
  D47 (`reduce.add_skipped_segments`): one-pass notes; a segment no decision cites gets its Map
  decisions (not its tasks or questions), one no question cites its Map questions; a key point
  that states only the decision moves; Map's "[]" items skipped.
- Replay (main's code reproduced every stored notes.json first): only the AMI notes change.
  The AMI `--rerun reduce` wrote byte-identical replies, and its notes equal the replay.

| | before (8g) | after |
|---|---|---|
| AMI coverage (Oct 28 notes: 52.8%) | 44.9% | **50.6%** |
| AMI abstract / decisions / actions / problems | 60.5 / 30.4 / 45.5 / 16.7% | 58.1 / **47.8** / 45.5 / **33.3%** |
| AMI support, unsupported numbers | 80.3%, 1 | 80.1%, 1 |
| AMI per meeting ES2004b / ES2004d / IS1009b / IS1009c / TS3003c | 46.7 / 23.8 / 43.8 / 63.2 / 50.0% | 53.3 / 28.6 / 56.2 / 63.2 / 55.6% |
| AMI Reduce s/h | 456 (wall-clock artifact) | 129 |
| quick seeds 0–3 fact recall | 98.5 / 97.1 / 98.5 / 100% | same; notes identical (replayed per seed) |
| quick seed 0 Ask, citation, judge support | 98.1%, 100%, 93.8% | same (summary identical) |
| lang seed 1 | 95.6% recall, 98.9% citation | summary identical |
| long-en | — | notes identical (two-pass: D47 skips; no heading, no repeats) |

- AMI, sentence by sentence (8g → 8h): +6 (IS1009b teletext, whole-house, budget; TS3003c
  battery; ES2004b Video Plus; ES2004d's noise sentence back), −1 (IS1009b "plastic cases",
  its claim moved from Summary to Key points: noise).
- Runs: quick seed 0 with Ask and the judge (`--rerun reduce`), quick seeds 1–3 `--no-ask
  --no-judge` (seed-0 runs copied first and put back), lang `-c seed=1 --no-ask --rerun
  reduce`, AMI `--item ami --no-ask --rerun reduce` (asked). long-en not rerun (asked; the
  replay shows it can't change). No baseline rewritten: quick, lang and long don't move, and
  `--update-baseline` after an `--item ami` run would replace baseline-full.json with an
  AMI-only report.
- Cost: no model call; Ask keys, MapFile, NotesDoc and REDUCE_HASH unchanged.

### Open
- AMI's 2 points to the Oct 28 notes: ES2004b's 3 abstract sentences and "young,
  business-class buyer" (reply drift: segs 3–4's key points, R7B's own seg 5–6 decisions);
  IS1009b "corporate image" (Map lists it as a task too, so D44's rule skips it, and 8b's
  vague-task move made the task a key point); ES2004d's production decision (noise).
- Key points of skipped segments aren't filled (D47): revisit if a set shows Summary missing.
- Map writes "[]" as an open question (5 AMI segments, 2 long-en): skipped by D47, still a
  `- []` bullet in Reduce's input.
- 8g's Open (Map's 8 never-written numbers, xling-ja f9, the judge on slide-conflict notes,
  long-en's 3 key points) and post-v1 items 5, 8, 10 unchanged.

- 1.9.0 bump: done 2026-10-30 on request (release-1.9.0).

## 2026-10-30 · Phase 8i kickoff prompt (no code)
- PLAN §21 "Kickoff prompt for Phase 8i" and a §9 line: Japanese meetings (post-v1 items 5
  and 10, carry-over 5 of item 8, 8g's xling-ja f7/f9). Picked over Ask's near-miss (no cause
  found yet) and AMI's leftovers (~2 points, mostly judge noise): xling-ja is the lang set's
  weakest item on every measure (seed 1: recall 76.9%, citation 90.0%, support 65.0%).
- Kickoff check on the stored run (CLAUDE.md lesson): Transcribe wrote 5 of xling-ja's 18
  chunks without punctuation, and 4 of them lost their opening (chunk 5 the whole 42,500
  budget line, f6, so Map never saw it; chunk 7 "先月は7700ドル"; chunk 11 "昨日"; chunk 15
  "先月は2600"). No punctuated chunk lost one, and all chunks start 0–40 ms before speech.
  #10's −21.8 s is in chunk 11, so the two items may share a cause.
- The decoder already sends <|pnc|>: no switch to flip. No generated set has an unpunctuated
  chunk; AMI has 10–26 per meeting (spontaneous speech: step 0 checks them against AMI's
  reference words).
- Map: f7 (11,800, chunk 6) in no Map item; f9 "35万4000" written "$34,000" and only flagged;
  one seg-0 key point with no source (budget "for team spending, user testing, and
  onboarding").
- #5 confirmed in Python: Victor "fktr", ビクター "fkt"; Omar and Maria share "mr" in katakana.
  No set has Japanese notes; a clean-ja lang item (clean-en seed 1, target ja) would plant
  all five owners.

## 2026-10-30 · Phase 8i: Japanese meetings
Branch `phase-8i-japanese`. Post-v1 items 5 and 10, carry-over 5 of item 8, 8g's f7/f9; D48–D51.
- Step 0 (no models): the kickoff's chunk table reproduces from the stored run. Chunks 5, 7, 11,
  13, 15 have no sentence mark inside; 5, 7, 15 lost their whole first line, 11 lost "昨日".
  AMI against its reference words: unpunctuated chunks (87 in 5 meetings) find 84% of their
  opening words, punctuated ones 82%, word recall the same: there it's spontaneous speech.
- Step 0 (Transcribe): greedy decoding is deterministic (the 5 chunks decode to the stored
  replies). Pre-roll and silence padding give identical text (leading silence is what
  matters) and act as a coin flip: they restored chunks 5 and 7 but took control chunk 9's
  opening (0.78 → 0.11) and 1, 6, 9's punctuation; chunk 15's "先月は2600でした" came back
  under no variant, a split at a pause included. Decoding the chunk's first span alone
  found each lost line (coverage in the reply 0.08–0.33; every kept opening ≥ 0.81).
- D48 (`asr.py`): a reply over 40 characters with no sentence mark before its end gets that
  probe; under 60% of the span at the reply's head → its text goes in front (minus words the
  reply opens with). Fires: xling-ja chunks 5, 7, 15; no generated-set chunk qualifies; AMI 7
  (5 real lost speech, e.g. "That brings us to the chip"; a "Yeah." and "b uh tha" noise).
- D49 (`reduce._placed`): MARIA's span took the word path ("maria" scores exactly the 0.1
  minimum), not the anchors 8a planned for. Both paths now place 1 s before the first hit in
  an unpunctuated chunk over 80 characters and 10 s. The character floor keeps the test
  fixtures' one-sentence 30 s chunks at their start.
- D50 (guard): units ($/ドル vs %) settle f9's two candidates; after D48, Map's new input made
  it "$345,000", so adjacent swaps are near too. 77 stored Map files replayed per rule: f9 only.
- D51 (Japanese notes): clean-ja (new) showed the eval and the guard saw no number in Japanese
  notes ("390件" read as nothing, "4,650件" as 4, 万 never) and 四半期 read as 4.
  `notes_numbers` uses the meeting reader for ja; WRITTEN lets kana and kanji (not 万億千百十)
  touch a number; `restore` maps a final ー to -r (R7B wrote ローラー, ヴィクトール, マリア: all
  restored); the eval's owner match works on "Victorさん".

| | before | after |
|---|---|---|
| xling-ja recall / citation / support | 76.9 / 90.0 / 65.0% | **92.3 / 100 / 94.7%** |
| xling-ja numbers, CER, f11 time | 62.5%, 7.1%, −21.8 s | **87.5%, 5.2%, −1.4 s** (f7 left) |
| clean-ja (new) recall / actions / numbers | — | 84.6 / 100 / 75% (f6 450,000 flagged, f16 never written) |
| clean-ja Ask, near-miss | — | 100% (19), 66.7% |
| lang seed 1 recall / citation / support / judge recall | 95.6 / 98.9 / 77.1 / 83.3% | 96.2 / 100 / 81.5 / 85.4% (other 6 items identical) |
| quick seed 0 with Ask and the judge | 98.5%, Ask 98.1% | identical (`--rerun transcribe`) |
| AMI coverage | 50.6% | 50.6% |
| AMI abstract / decisions / actions / problems | 58.1 / 47.8 / 45.5 / 33.3% | 60.5 / 39.1 / 45.5 / 41.7% |
| AMI per meeting ES2004b / ES2004d / IS1009b / IS1009c / TS3003c | 53.3 / 28.6 / 56.2 / 63.2 / 55.6% | 46.7 / 28.6 / 62.5 / 63.2 / 55.6% |
| AMI support | 80.1% | 78.8% |

- AMI, sentence by sentence: Map changed only in the 6 segments holding a restored chunk.
  ES2004b's 2 changed segments gave R7B a new reply (5 of 9 decisions reworded, "Focus on one
  product design" gone): −2 decisions, +2 abstract, −1 abstract. IS1009b dumped under an
  unknown heading again (D46 filed it): +1 problem. ES2004d ±1 (noise).
- Runs: lang `-c seed=1 --no-ask --rerun transcribe`, then `--rerun map` after D50/D51 (stored
  runs copied first), xling-ja + clean-ja with Ask and the judge, quick seed 0 `--rerun
  transcribe` with Ask and the judge, AMI `--item ami --no-ask --rerun transcribe` (asked; run
  by the user: the guard hook blocks the set name). long-en and FLEURS not run: no long-en
  chunk is unpunctuated and the guard replay leaves it unchanged; FLEURS calls the model
  directly, not `run_stage`. baseline-lang.json rewritten from the whole seed-1 run (gains
  clean-ja); baseline-full.json not (an `--item` run).
- Cost: one extra decode of 1–8 s per probed chunk (~0.3 s; xling-ja Transcribe 117.6 →
  116.0 s/h, noise). No schema, prompt or REDUCE_HASH change; Ask keys move only where the
  transcript or notes did (xling-ja, clean-ja, AMI).

### Open
- xling-ja f7 (11,800) and clean-ja f16 (4,650): Map never wrote them (8g's never-written
  numbers). clean-ja f6: Map's "450,000" is a swap and ×10 at once, flagged.
- ES2004b's decisions after its new Reduce reply (reply drift from 2 changed Map segments).
- AMI's 2 noise restorations ("Yeah.", "b uh tha"): harmless so far; a minimum-content rule
  would need AMI's reference words to tune.
- `names.said_names` finds no name in a Japanese transcript ("VICTORさん" is one word): only
  matters for a Japanese meeting with Japanese notes (no set has one).
- 8h's Open (ES2004b's drift, IS1009b "corporate image", key points of skipped segments, "[]"
  questions), the judge on slide-conflict notes, long-en's 3 key points, the Japanese
  slide-conflict check and Ask's near-miss (item 8) unchanged.
