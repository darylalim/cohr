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
- **GitHub (§17), state before this PR:** repo public, default branch `main`, description set,
  Actions token read-only. Not yet: topics, merge settings (squash only, delete branch on merge,
  no wiki/projects), branch protection.
- **Not in Phase 1 (by plan):** release job (§16), ModelManager and download disk check (Phase
  2), upload streaming (Phase 5).

### Next
- §17 steps 2, 3, 5 and 6 after confirmation; first CI run on the Phase 1 PR. Then Phase 2
  (perception: `vad.py`, `asr.py`, `vision.py`, ModelManager).
