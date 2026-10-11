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

### Decisions (PLAN §20)
| # | Result | Evidence |
|---|---|---|
| D8 | Voices for all 14 languages. ≥2 real voices: en, de, es, fr, it, ja, ko, nl, pt, zh. Only 1: ar (Majed), el (Melina), pl (Zosia), vi (Linh) | `say -v '?'` (counts include Eloquence voices like Eddy/Flo) |
| D9 | No torch/torchvision/torchaudio in `uv.lock`; `torch` not in `sys.modules` after importing every runtime package | `grep '^name = "torch' uv.lock`; import check |

### Plan changes
- §6 ruff config gains `extend-exclude = ["*.md"]` (to update in PLAN §6 in the Phase 0 PR).

### Next
- Ask the user before the ~15 GB model download; then per-model load checks, D1–D7, D10.
