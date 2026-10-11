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
