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
