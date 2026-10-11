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

### Decisions (PLAN §20)
| # | Result | Evidence |
|---|---|---|
| D1 | **mlx-speech** (Auto, go/no-go row 1). mlx-audio can't load the int8 checkpoint correctly | Comparison table above |
| D2 | `max_chunk_s` = 30 s kept; no truncation up to 60 s (runtime chunks internally) | Words/s probe |
| D3 | No auto-detection in either runtime: the user picks the source language | Runtime APIs |
| D5 | Defaults kept: threshold 0.5, 250/100/30 ms, merge < 0.5 s | Speech F1 0.979, 0 s in gap; re-check on eval |
| D8 | Voices for all 14 languages. ≥2 real voices: en, de, es, fr, it, ja, ko, nl, pt, zh. Only 1: ar (Majed), el (Melina), pl (Zosia), vi (Linh) | `say -v '?'` (counts include Eloquence voices like Eddy/Flo) |
| D6 | Pinned revisions below (downloaded at these SHAs); mlx-vlm `952d4f6bc65bd5095e74abe71c78038764ee08aa` | Hub `model_info` 2026-10-10 |
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
- §5.2/§2: Transcribe runtime is mlx-speech; mlx-audio only for VAD (§1, §6 wording).
- §11 dependency rule: add MPL-2.0 (transitive, unmodified), user-approved.
- §6 ruff config gains `extend-exclude = ["*.md"]` (to update in PLAN §6 in the Phase 0 PR).

### Next
- Download running (user approved). Then per-model load checks, D1–D7, D10, spike.
