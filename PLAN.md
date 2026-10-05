# Plan: Cited Meeting Notes (Tier A pipeline)

A local Streamlit app that turns a **recording plus slides** (or a screen-recording video) into
**multilingual notes in which every claim cites its timestamp or slide**. It uses four Cohere
models on MLX, all of which fit in memory together on an **M2 Max with 32 GB**.

Status: draft, verified against Hub metadata on 2026-10-04. Nothing is implemented yet.

---

## 1. Goals and non-goals

**Goals**
- Inputs: one audio or video file, plus optional slide images or PDF pages.
- Output: `notes.md` with a summary, key points, decisions and action items. Each claim cites
  `[mm:ss]` and/or `[slide N]` and links back to the source segment.
- Output in a target language the user picks; the source speech can be in any of Transcribe's 14 languages.
- An "Ask" tab: questions about a processed run, answered with citations.
- Fully local and offline once the models are downloaded. Each stage saves its output, so
  re-running a stage doesn't redo the earlier ones.

**Non-goals (v1)**
- Live or streaming transcription, and identifying who is speaking.
- Commercial use (Tiny Aya and R7B are CC-BY-NC; see §11).
- Tier B models (North Mini Code) and any model too large for 32 GB.

---

## 2. Models (exact repos, measured sizes)

| Stage | Model | MLX repo | Weights | Runtime | License |
|---|---|---|---|---|---|
| Speech → text | Cohere Transcribe 2B | `mlx-community/cohere-transcribe-03-2026-mlx-8bit` (files in `mlx-int8/`) | 4.13 GB | `mlx-speech` (PyPI, ≥0.5.3) | Apache 2.0 |
| Slides → text | North Micro Vision Instruct 2.4B | `mlx-community/North-Micro-Vision-Instruct-8bit` | 3.15 GB | `mlx-vlm` **from git main** (Cohere Compass support isn't released yet) | Apache 2.0 |
| Map (per segment) | Tiny Aya Global 3.35B | `mlx-community/tiny-aya-global-8bit-mlx` | 3.56 GB | `mlx-lm` | CC-BY-NC |
| Reduce + Ask | Command R7B | `mlx-community/c4ai-command-r7b-12-2024-4bit` | 4.52 GB | `mlx-lm` | CC-BY-NC |
| **Total** | | | **~15.4 GB** | | |

**Smaller fallbacks** if memory is tight: `beshkenadze/cohere-transcribe-03-2026-mlx-4bit` (1.51 GB;
check it works with `mlx-speech` in Phase 0), `mlx-community/North-Micro-Vision-Instruct-4bit`
(2.17 GB), `Siarhei/tiny-aya-global-4bit` (1.88 GB).

**Context limits that shape the design**
- Tiny Aya: 8K tokens in and 8K out, so it gets one segment per call.
- Micro Vision: validated up to 8K multimodal tokens, so one image per call.
- R7B: 128K context, so the final step sees the whole meeting.

---

## 3. Architecture

```
INGEST (ffmpeg, no models)
  audio/video ─► 16 kHz mono wav ─► chunks of ~25–30 s cut at silences ─► chunks/*.wav + offsets
  video       ─► frames at scene changes (pts timestamps) ─► perceptual-hash dedupe ─► frames/*.png
  slide files ─► page images (PDF via pdftoppm or PyMuPDF) ─► frames/*.png (no timestamp)

PERCEIVE (models, one input at a time)
  Transcribe(chunk, language) ─► transcript.json   [{chunk_id, t0, t1, text}]
  MicroVision(frame)          ─► slides.json       [{slide_id, t, title, text_md, tables_md, chart_notes}]

FUSE (no models)
  group transcript chunks into ~5 min windows; attach slides by timestamp
  (the slide on screen at the window start, plus any shown during it) ─► segments.json

MAP (Tiny Aya Global, once per segment, output in the target language)
  segment ─► {summary, key_points[], decisions[], action_items[], terms[]} ─► map.json

REDUCE (Command R7B, RAG template, citations on)
  documents = one per segment (time range + slide ids + map notes + raw transcript excerpt)
  ─► final notes with <co>…</co: 0:[i,…]> spans ─► parse ─► notes.md (+ notes_raw.txt)

ASK (Command R7B, same documents) ─► answer with citations
```

**Run directory** (`runs/<YYYYmmdd-HHMMSS>-<slug>/`)
```
manifest.json        inputs, config, model repos, stage status + config hashes
source/              copied or linked originals
audio.wav            16 kHz mono
chunks/              000.wav, 001.wav, …  (+ chunks.json with offsets)
frames/              000.png, … (+ frames.json with pts, phash)
transcript.json
slides.json
segments.json
map.json
notes_raw.txt        R7B output exactly as generated (keep it for debugging)
notes.md             rendered output
```
A stage is skipped when its output exists and its input and config hash match `manifest.json`.

---

## 4. Memory policy

- **Default `resident`:** keep all four models loaded (~15.4 GB weights + ~3–5 GB of KV and
  working memory, about 19–20 GB). That's under the ~21.5 GB default GPU limit and makes the Ask
  tab and re-runs instant.
- **`sequential`:** the stages run one after another anyway, so you can unload each model when its
  stage finishes (`del`, `gc.collect()`, `mx.clear_cache()`). Peak memory is then the largest
  single model plus KV, about 8 GB. Use this if the machine is busy with other things.
- Optional, documented but never applied automatically: `sudo sysctl iogpu.wired_limit_mb=26000`
  raises the GPU limit until reboot.
- Cap R7B's input at ~32K tokens by default. Above that (inputs longer than ~2 h), reduce in two
  passes: summarize groups of segments, then summarize the group summaries.
- Show peak memory (`mx.get_peak_memory()`) in the UI after each stage.

---

## 5. Stage details

### 5.1 Ingest
- Normalize: `ffmpeg -i in -ac 1 -ar 16000 -c:a pcm_s16le audio.wav`.
- Chunking: run `ffmpeg -af silencedetect=noise=-30dB:d=0.4`, then cut at the silence nearest a
  target of ~27 s. Force a cut at the maximum length (default 30 s; set it from the Transcribe
  limit found in Phase 0). Store `{chunk_id, t0, t1}`. Cutting at silences avoids having to merge
  overlapping text.
- Video frames: `ffmpeg -vf "select='gt(scene,0.3)',showinfo" -vsync vfr frames/%03d.png` and
  parse each frame's `pts_time` from the `showinfo` output. Always keep the first frame.
- Dedupe: `imagehash.phash`, dropping a frame if its Hamming distance to the last kept frame is
  ≤ 6. Expect 20–60 frames per hour of talk.
- Slide files without timestamps get `t = null`. They're used as unaligned documents in Reduce
  and are left out of segment windows.

### 5.2 Transcribe (mlx-speech)
```python
from mlx_speech.generation import CohereAsrModel
model = CohereAsrModel.from_path(local_dir / "mlx-int8")   # local_dir from snapshot_download
result = model.transcribe(audio_f32_16k, sample_rate=16000, language=lang)
```
- The user picks the source language in the UI. **Phase 0:** check whether `language` can be
  left out for auto-detection, and find the maximum input length.
- Supported languages: ar de el en es fr it ja ko nl pl pt vi zh.

### 5.3 Describe slides (mlx-vlm, Micro Vision)
- Temperature 0, max ~1024 tokens, one image per call.
- The prompt asks for JSON only:
  `{"title": str, "text_md": str (verbatim, reading order), "tables_md": [str], "chart_notes": str, "visual_notes": str}`.
- If parsing fails, retry once with "Return only valid JSON". After that, keep the raw text in `text_md`.

### 5.4 Fuse
- Window length is configurable (default 300 s). Each segment holds `{seg_id, t0, t1,
  transcript_text, slide_ids[]}`.
- Budget check: `len(tokenize(transcript_text + slide text))` must stay ≤ ~5K tokens for Tiny Aya.
  Split the window in half if it's over.

### 5.5 Map (Tiny Aya Global)
- One call per segment, temperature ~0.3, output in the target language, JSON only:
  `{"summary": str, "key_points": [str], "decisions": [str], "action_items": [{"owner": str|null, "task": str, "due": str|null}], "terms": [str]}`.
- The prompt says to use only the content in the segment and to prefer the slide text's spelling
  for names and numbers (speech recognition often gets these wrong).

### 5.6 Reduce (Command R7B with citations)
- Use the tokenizer's **`rag`** chat template, which is present in the MLX repo and supports
  `documents` and `enable_citations`. Each document:
  `{"title": "12:30–17:30 · slides 4,5", "snippet": <map notes + slide text + raw transcript excerpt>}`.
  Unaligned slides become their own documents (`"title": "slide 7"`).
- **Phase 0:** confirm the exact `apply_chat_template(..., documents=..., chat_template="rag", enable_citations=True)`
  arguments, and the citation format R7B produces (expected: `<co>span</co: 0:[i,j]>`).
- The prompt asks for these sections: Summary · Key points · Decisions · Action items · Open questions.
- Citation parsing: regex `<co>(.*?)</co:\s*([^>]+)>` → list of document indices → render as
  `span [12:30](#seg-3) [slide 4](#slide-4)`.
- Target language: R7B covers 23 languages. If the target isn't one of them, reduce in English
  and translate the final notes section by section with Tiny Aya.

### 5.7 Ask
- Same documents as Reduce, the user's question, and the `rag` template with citations. Answer
  rendered with the same citation links. No re-processing needed.

---

## 6. Project layout (uv, Python 3.12)

```
pyproject.toml
src/cohr/
  __init__.py
  config.py          pydantic settings: model repos, window sizes, thresholds, memory policy
  models.py          ModelManager: lazy load, resident/sequential policy, memory stats, one lock
  ingest.py          ffmpeg wrappers: normalize, silence chunking, scene frames, pdf pages, dedupe
  asr.py             Transcribe wrapper
  vision.py          Micro Vision wrapper + JSON repair
  fuse.py            segment windows + slide alignment + token budget split
  mapper.py          Tiny Aya per-segment notes
  reduce.py          R7B rag documents, generation, citation parsing
  render.py          notes.md rendering with anchors
  pipeline.py        stage runner: manifest, hashing, skip logic, progress callbacks
  cli.py             `cohr run <audio> [--slides …] [--video …] --lang en --target en`
app.py               Streamlit UI
tests/
  fixtures/          generated, not committed if large (see §8)
  test_ingest.py  test_fuse.py  test_citations.py  test_render.py  test_pipeline_smoke.py
runs/                gitignored
```

**Dependencies:** `mlx`, `mlx-lm`, `mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git`,
`mlx-speech>=0.5.3`, `streamlit`, `huggingface_hub`, `soundfile`, `numpy`, `pillow`,
`imagehash`, `pydantic`, `pymupdf` (PDF pages), `pytest`, `ruff`.
**System:** `ffmpeg` (already at `/opt/homebrew/bin/ffmpeg`) and `uv` (already installed).

---

## 7. Streamlit UI (`app.py`)

- Load the **`developing-with-streamlit`** skill before writing UI code.
- `ModelManager` is created once with `@st.cache_resource`, which survives reruns and keeps
  reruns from loading the models twice.
- Run the pipeline in a background thread that reports progress through a queue. The UI polls
  with `st.status` and a progress bar per stage, so a rerun doesn't cut off a long job. Only one
  job at a time (enforced by the ModelManager lock).
- Tabs:
  1. **New run:** upload audio or video and slides; pick source language, target language,
     window length and memory policy; Run.
  2. **Notes:** rendered `notes.md`. Clicking a citation scrolls to the segment, which has an
     `st.audio` player that starts at `t0` plus the slide thumbnails.
  3. **Ask:** chat over the selected run, with citations.
  4. **Runs:** list of previous runs, with per-stage re-run buttons ("re-reduce with new prompt").
- Sidebar: models loaded, peak memory per stage, GPU limit warning.

---

## 8. Testing

- **Unit tests (no models):** silence-cut selection, frame dedupe, segment windowing and token
  budget split, citation regex and index mapping, render anchors, manifest skip logic.
- **Fixtures generated on the fly** (no binaries committed):
  - Audio: `say -o /tmp/x.aiff "…"` → ffmpeg → wav (macOS TTS), two speakers via `-v`
    voices, ~2 min total.
  - Slides: PIL draws 3 images with a title, bullets and a small table with known numbers.
- **Smoke test** (marked `@pytest.mark.models`, opt in): full run on the fixtures. Checks that
  the transcript contains known words, the slide JSON contains the table numbers, `notes.md`
  has at least one citation per section, and every citation resolves to an existing segment or slide.
- Lint and type-check with `ruff` (and `ty` if wanted).

---

## 9. Implementation phases (acceptance criteria)

**Phase 0: checks before building (do first; record findings in `NOTES.md`)**
- [ ] Each model loads and runs alone with the commands above; record load time and peak memory.
- [ ] Transcribe: maximum input length, whether language auto-detection works, and
      whether the 4-bit `beshkenadze` repo works with `mlx-speech`.
- [ ] Micro Vision through mlx-vlm main: a slide image produces valid JSON with the prompt in §5.3.
- [ ] R7B `rag` template: exact kwargs; citations appear in the output in the expected format.
- [ ] Tiny Aya: the JSON-only prompt follows the schema reliably in 3 languages.
- [ ] All four loaded together: peak memory is under the GPU limit with a 32K R7B prompt.
- **Exit:** a single `scripts/spike.py` that runs all four and prints memory use.

**Phase 1: scaffold + ingest:** uv project, config, ingest functions, unit tests pass.
**Phase 2: perception:** `asr.py`, `vision.py`, ModelManager, `transcript.json` and `slides.json`, plus skip logic.
**Phase 3: fuse + map:** `segments.json` and `map.json`; token budget enforced.
**Phase 4: reduce + render:** `notes.md` with working citations; CLI end to end on fixtures.
**Phase 5: Streamlit UI:** all four tabs, background job, progress, Ask.
**Phase 6: hardening:** sequential memory policy, inputs over 2 h (two-pass reduce), error
messages for missing ffmpeg or models, a cache size note for the README.

**Definition of done:** a 30–60 min real recording plus slides produces cited notes in a chosen
target language. All citations resolve. Re-running only Reduce takes under a minute. Peak memory
stays under the limit with the `resident` policy.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `mlx-vlm` main changes or breaks Compass support | Pin the git commit that works in Phase 0 |
| Small models produce malformed JSON | Low temperature, schema in the prompt, one retry, fall back to raw text |
| R7B citation format differs from expected | Phase 0 check; parser isolated in `reduce.py` with tests |
| Speech recognition mangles names and numbers | Map prompt prefers slide spelling; terms list passed to Reduce |
| Background thread + MLX + Streamlit reruns | One ModelManager lock, `cache_resource`, one job at a time |
| Memory pressure from other apps | `sequential` policy switch; sidebar warning when free memory is low |
| First-run downloads (~15 GB) | Download step in the UI with progress via `huggingface_hub.snapshot_download` |

---

## 11. Licensing

Transcribe and Micro Vision are Apache 2.0. **Tiny Aya and Command R7B are CC-BY-NC-4.0.**
This app is therefore **for personal and research use only**; state that in the README. A
commercial version would have to replace Map and Reduce with commercially licensed models.
