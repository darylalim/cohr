# Plan: Cited Meeting Notes (Tier A pipeline)

A local Streamlit app that turns a **recording plus slides** (or a screen-recording video) into
**multilingual notes in which every claim cites its timestamp or slide**. It uses four Cohere
models plus a small voice-detection model, all on MLX and all fitting in memory together on an
**M2 Max with 32 GB**.

Status: Phase 0 done 2026-10-10 (all checks passed; findings in `NOTES.md`, results in §20).
Phase 1 done 2026-10-11 (scaffold + ingest; §17 GitHub setup complete, CI green).
Phase 2 done 2026-10-11 (perception: VAD, Transcribe, Vision, ModelManager; smoke test passes offline).
Phase 3 done 2026-10-11 (fuse + map: slide text alignment, segments, Tiny Aya Map).
Hub metadata verified 2026-10-04 and 2026-10-10; packages, licenses and Streamlit config checked
2026-10-05.

---

## 1. Goals and non-goals

**Goals**
- Inputs: one audio or video file, plus optional slide images or PDF pages.
- Output: `notes.md` with a summary, key points, decisions, action items and open questions. Each claim cites
  `[mm:ss]` and/or `[slide N]` and links back to the source segment.
- Output in a target language the user picks; the source speech can be in any of Transcribe's 14 languages.
- An "Ask" page: questions about a processed run, answered with citations.
- Fully local and offline once the models are downloaded, with **no telemetry**: Streamlit usage
  stats off, Hugging Face telemetry off, and Hub offline mode on after downloads (§7 config, §12).
  Each stage saves its output, so re-running a stage doesn't redo the earlier ones.
- **All model inference runs on MLX** (`mlx-audio` and/or `mlx-speech`, `mlx-vlm`, `mlx-lm`): one framework, one
  unified-memory allocator, one set of memory stats. No PyTorch, GGUF/llama.cpp or CoreML
  inference. Non-model steps (ffmpeg, imagehash, pypdfium2, soundfile) run on the CPU and aren't
  covered by this rule. `transformers` is allowed for tokenizers, chat templates and image
  processors only.

**Non-goals (v1)**
- Live or streaming transcription, and identifying who is speaking.
- Meetings that switch language mid-conversation: Transcribe takes one source language per run,
  so speech in another language is transcribed poorly. Known limitation; pick the main language.
- Commercial use **with the default models** (Tiny Aya and R7B are CC-BY-NC; see §11). The code
  itself is Apache-2.0.
- Tier B models (North Mini Code) and any model too large for 32 GB.

---

## 2. Models (exact repos, measured sizes)

| Stage | Model | MLX repo | Weights | Runtime | License |
|---|---|---|---|---|---|
| Speech → text | Cohere Transcribe 2B | `mlx-community/cohere-transcribe-03-2026-mlx-8bit` (files in `mlx-int8/`) | 4.13 GB | `mlx-speech` (≥0.5.3), chosen in Phase 0 (§5.2, D1) | Apache 2.0 |
| Slides → text | North Micro Vision Instruct 2.4B | `mlx-community/North-Micro-Vision-Instruct-8bit` | 3.15 GB | `mlx-vlm` **from git main** (Cohere Compass support isn't released yet) | Apache 2.0 |
| Map (per segment) | Tiny Aya Global 3.35B | `mlx-community/tiny-aya-global-8bit-mlx` | 3.56 GB | `mlx-lm` | CC-BY-NC |
| Reduce + Ask | Command R7B | `mlx-community/c4ai-command-r7b-12-2024-4bit` | 4.52 GB | `mlx-lm` | CC-BY-NC |
| Speech detection | Silero VAD v6 | `mlx-community/silero-vad-v6` | ~2 MB | `mlx-audio` (`mlx_audio.vad`) | MIT |
| **Total** | | | **~15.4 GB** | | |

**Smaller fallbacks** if memory is tight: `mlx-community/North-Micro-Vision-Instruct-4bit`
(2.17 GB), `Siarhei/tiny-aya-global-4bit` (1.88 GB); untested, and not needed so far (Phase 0
measured 8.6 GB headroom, §4). There is **no usable 4-bit Transcribe**:
`beshkenadze/cohere-transcribe-03-2026-mlx-4bit` was converted for `mlx-audio-swift` and loads
in neither Python runtime (D4).

**Context limits that shape the design**
- Tiny Aya: 8K tokens in and 8K out, so it gets one segment per call.
- Micro Vision: validated up to 8K multimodal tokens, so one image per call.
- R7B: 128K context, so the final step sees the whole meeting.

**Considered and not used** (all 55 repos in the `CohereLabs` Hub org checked 2026-10-09; don't
re-check unless the org adds new models)
- **No local weights:** `Cohere-embed-*-v3.0`, `rerank-*`, `multilingual-22-12`, `Command-nightly`.
  These are 0 GB placeholders for Cohere's online API, so they break "offline". This is also why
  §13 has no semantic search.
- **Too big for 32 GB:** North-Small-Translate-1.0 (218B), the Command A family (A, A Plus,
  A Reasoning, A Vision, A Translate), Command R / R+ (35B / 104B), Aya Expanse 32B, Aya 23 35B,
  Aya Vision 32B.
- **Replaced by newer, smaller models:** Aya Vision 8B (17 GB, CC-BY-NC; Micro Vision instead),
  Aya 23 8B, Aya Expanse 8B, aya-101 (Tiny Aya instead).
- **Wrong kind for Map:** `tiny-aya-base`, `tiny-aya-base-32K` (base models, not
  instruction-tuned); `tiny-aya-en-thinker`, `tiny-aya-l2-thinker` (reasoning models: slower, and
  their reasoning text gets in the way of JSON-only output).
- **Arabic specialists:** `cohere-transcribe-arabic-07-2026` (Apache-2.0, MLX ports exist) and
  `c4ai-command-r7b-arabic-02-2025`. Base Transcribe and R7B already cover Arabic. Reconsider only
  if Arabic meetings become a main use case.
- **Deferred (optional, Phase 6):** the regional Tiny Aya variants `tiny-aya-earth`, `-fire` and
  `-water`, as per-language Map models (§5.5, D18).

---

## 3. Architecture

```
INGEST (ffmpeg, no models)
  audio/video ─► 16 kHz mono wav (audio.wav)
  video       ─► frames at scene changes (pts timestamps) ─► perceptual-hash dedupe ─► frames/*.png
  slide files ─► page images (PDF via pypdfium2) ─► frames/*.png (no timestamp)

SPEECH DETECT (Silero VAD on MLX, ~2 MB; fallback: ffmpeg silencedetect)
  audio.wav ─► speech runs ─► merge + pack into chunks ≤ max length ─► chunks.json
              (non-speech regions are skipped, recorded as `gaps`)

PERCEIVE (models, one input at a time)
  Transcribe(chunk, language) ─► transcript.json   [{chunk_id, t0, t1, text}]
  MicroVision(frame)          ─► slides.json       [{slide_id, t, title, text_md, tables_md, chart_notes}]

FUSE (no models)
  PDF/image slides (no timestamp) ─► text-align to transcript ─► estimated t (or unaligned)
  group transcript chunks into ~5 min windows; attach slides by timestamp
  (the slide on screen at the window start, plus any shown during it) ─► segments.json

MAP (Tiny Aya Global, once per segment, output in the target language)
  segment ─► {summary, key_points[], decisions[], action_items[], open_questions[], terms[]} ─► map.json

REDUCE (Command R7B, RAG template, citations on)
  documents = one per segment (time range + slide ids + map notes + raw transcript excerpt)
              + one per slide
  ─► final notes with <co>…</co: 0:[i,…]> spans ─► parse ─► refine citation times (no models)
  ─► notes.json (raw reply kept inside) ─► render ─► notes.md

ASK (Command R7B, same documents) ─► answer with citations
```

**Run directory** (`runs/<YYYYmmdd-HHMMSS>-<slug>/`, local time; `-2`, `-3` … on a clash)
```
manifest.json        inputs, config, model repos, stage status + config hashes, app version,
                     per-stage format versions (§19)
source/              symlinks to originals by default, or copies (see §12)
audio.wav            16 kHz mono, the only audio copy
chunks.json          {chunker, chunks: [{chunk_id, t0, t1}], gaps: [{t0, t1}]}: offsets into
                     audio.wav, no chunk WAVs; gaps = skipped non-speech
frames/              000.png, … downscaled to ≤1600 px long edge (+ frames.json with pts, phash)
transcript.json
slides.json
segments.json
map.json
notes.json           parsed notes with structured citations (§19); what Notes/Ask/evals read;
                     `raw` holds R7B's reply exactly as generated (debugging, evals)
notes.md             rendered from notes.json as a portable file (plain citations + Sources
                     appendix, §5.6); this is what Download gives you
log.txt              run log (INFO by default, DEBUG via config), §7 job lifecycle
```
A stage is skipped when its output exists and its input and config hash match `manifest.json`.
A stage's config hash includes the hash of its prompt file (`src/cohr/prompts/`, §6), so editing
one prompt re-runs only that stage and the ones after it. Every output is written to a temp file
and then `os.replace`d, so a crash never leaves a half-written file that the skip logic would
trust.

**Reproducible generation:** `config.seed` (default 0) is passed to `mx.random.seed()` before
every generating stage (Vision, Map, Reduce, Ask). It's part of the config (and so of each stage's
config hash) and recorded in the manifest. Same inputs + config + seed give the same output, so eval
run-to-run noise (D16) comes from real differences, not sampling luck.

---

## 4. Memory policy

- **Default `resident`:** keep the four Cohere models loaded (~15.4 GB weights + ~3–5 GB of KV and
  working memory, about 19–20 GB). That's under the default GPU limit (**26.8 GB measured** on the
  M2 Max 32 GB in Phase 0; Phase 0 peak with all five models and a 34K-token R7B prompt was
  18.2 GB) and makes the Ask
  page and re-runs instant. Silero VAD (~2 MB) is loaded only for its step and then released.
- **`sequential`:** the stages run one after another anyway, so you can unload each model when its
  stage finishes (`del`, `gc.collect()`, `mx.clear_cache()`). Peak memory is then the largest
  single model plus KV, about 8 GB. Use this if the machine is busy with other things.
- Optional, documented but never applied automatically: `sudo sysctl iogpu.wired_limit_mb=26000`
  raises the GPU limit until reboot.
- Cap R7B's input at ~32K tokens by default. Above that (inputs longer than ~2 h), reduce in two
  passes: summarize groups of segments, then summarize the group summaries. **Citations survive
  both passes** (§5.6): pass 1 is a normal cited reduce per group, and its citations are written
  into the group summary as inline markers `[seg-N]`. Pass 2 cites group summaries; each pass-2
  citation resolves to the `[seg-N]` markers inside the cited span, so final links still point to
  segments. If a cited span has no marker, cite all segments of that group (coarse but valid).
- Show peak memory (`mx.get_peak_memory()`) in the UI after each stage.

---

## 5. Stage details

### 5.1 Ingest
- Normalize: `ffmpeg -i in -ac 1 -ar 16000 -c:a pcm_s16le audio.wav`.
- Chunking happens in the next step (§5.1a), not here. Ingest uses no models.
- Video frames: `ffmpeg -vf "select='gt(scene,0.3)',showinfo" -vsync vfr frames/%03d.png` and
  parse each frame's `pts_time` from the `showinfo` output. Always keep the first frame. Add
  `scale='min(1600,iw)':-2` (for portrait sources, scale the long edge instead) so 4K screen
  frames aren't stored at full size.
- Dedupe: `imagehash.phash`, dropping a frame if its Hamming distance to the last kept frame is
  ≤ 6. Expect 20–60 frames per hour of talk.
- PDF slides: `pypdfium2.PdfDocument(path)[i].render(scale=…).to_pil()`, with the scale chosen so
  the long edge is ≤1600 px (same cap as video frames), saved as `frames/NNN.png`. Image slides
  (PNG/JPG) are downscaled the same way.
- Slide files without timestamps get `t = null` here. Fuse then tries to place them in time by
  matching their text to the transcript (§5.4); slides it can't place stay unaligned documents in
  Reduce.

### 5.1a Speech detection and chunking (`vad.py`)
**Why:** a fixed volume threshold mistakes quiet speech for silence and finds no silence in noisy
rooms. Worse, speech recognition models tend to **make up text from non-speech audio** (music,
room noise, long breaks), and in this app made-up text would turn into cited "decisions".
Voice detection sends only speech to Transcribe.

**Default chunker: `vad`** (Silero VAD v6 via `mlx_audio.vad`, MLX-native, ~2 MB, MIT).
- `speech_runs = detect(audio.wav)` with threshold 0.5, minimum speech 250 ms, minimum silence
  100 ms, padding 30 ms (the `mlx-audio` defaults; tune in Phase 0).
- Merge runs separated by < 0.5 s. Pack consecutive runs into chunks up to `max_chunk_s`
  (default 30 s, capped by the Transcribe limit found in Phase 0), cutting only *between* runs,
  and never across a pause longer than `max_inner_gap_s` (default 2 s), so a long non-speech
  stretch between two nearby runs is never sent to Transcribe (Phase 2).
- A single run longer than `max_chunk_s` is split at the frame with the lowest speech probability
  in its last ~5 s.
- Regions with no speech go into `gaps` and are never transcribed. Fuse still uses real time, so
  windows, timestamps and citations are unaffected. The Notes page shows long gaps as
  "no speech 00:00–02:10".
- Write `chunks.json` only; `asr.py` slices `audio.wav` in memory (`soundfile.read(start=, stop=)`).
  Cutting between speech runs avoids having to merge overlapping text.
- Model lifetime: loaded for this step and released afterwards (~2 MB, no effect on the memory
  budget in §4).

**Fallback chunker: `silence`** (`config.chunker = "silence"`, also used automatically if the VAD
model can't load): a **relative** threshold computed in numpy on `audio.wav` (Phase 2; replaces
`ffmpeg silencedetect`, same rule): noise floor = 10th percentile of 1 s window levels,
`noise = floor + 6 dB`, not a fixed -30 dB; quiet dips < 0.3 s inside speech are filled. The
runs then go through the same packing; long runs are cut at their quietest frame.

**Packing logic is pure Python** (input: list of speech runs and probabilities), so it is unit
tested without the model (§8).

### 5.2 Transcribe (runtime chosen in Phase 0)

Two MLX runtimes can run Cohere Transcribe. **Phase 0 chose `mlx-speech` (Option B, D1):** on the
§2 8-bit repo it gave WER 0.11 on the fixture, 1.96 s per audio minute and 5.5 GB peak, while
`mlx-audio` loads the `mlx-int8/` checkpoint non-strictly and outputs the same gibberish for any
input. `mlx-audio` stays a dependency for Silero VAD only. `asr.py` keeps one interface
(`transcribe(chunk_f32_16k, lang) -> str`) and loads with strict weight checks.

**Option A: `mlx-audio`** (not chosen; kept for the record)
- `mlx_audio/stt/models/cohere_asr/`: batched transcription, language check, `max_audio_clip_s = 35`.
- It has its own optional Silero segmentation (`_segment_with_vad`). We **don't** use it; we pass
  our own chunks from §5.1a, so chunk offsets stay under our control for citations.
- Exact load call and weight compatibility with `mlx-community/cohere-transcribe-03-2026-mlx-8bit`:
  Phase 0.

**Option B: `mlx-speech`** (MIT, © 2026 AppAuto; needs Python ≥3.13)
```python
from mlx_speech.generation import CohereAsrModel
model = CohereAsrModel.from_path(local_dir / "mlx-int8")   # local_dir from snapshot_download
result = model.transcribe(audio_f32_16k, sample_rate=16000, language=lang)
```
Neither runtime auto-detects the language (D3); the decoder prompt always carries a language token.
- The user picks the source language in the UI. **Phase 0:** check whether `language` can be
  left out for auto-detection, and find the maximum input length.
- Supported languages: ar de el en es fr it ja ko nl pl pt vi zh.

### 5.3 Describe slides (mlx-vlm, Micro Vision)
- Temperature 0, max ~1024 tokens, one image per call.
- The prompt asks for JSON only:
  `{"title": str, "text_md": str (verbatim, reading order), "tables_md": [str], "chart_notes": str, "visual_notes": str}`.
- If parsing fails, retry once with "Return only valid JSON". After that, keep the raw text in `text_md`.

### 5.4 Fuse
- **Slide text alignment** (for PDF/image slides, the common "audio + deck" input): score every
  slide against every transcript chunk by token overlap (casefolded, numbers kept, stopwords
  dropped, using `title + text_md + tables_md` from `slides.json`). Score = shared tokens over
  the smaller token set, never dividing by fewer than 8; Chinese/Japanese runs become character
  bigrams; with ≥ 4 chunks, tokens in more than half the chunks are dropped (any language). Choose a **monotonic**
  assignment (slides are usually presented in order) with dynamic programming that maximizes the
  total score, allowing slides to be skipped. A slide whose best score is below a threshold
  (default 0.15, tuned on the eval) stays unaligned.
  Store `t`, `t_source: "pts" | "aligned" | "none"` and `align_score` in `slides.json` (Fuse
  rewrites it; re-running Fuse recomputes every non-`pts` slide, so it's idempotent). The UI marks
  text-aligned slides with "≈" before the time. Pure Python, unit tested (§8), evaluated by the
  `pdf-slides-en` variant (§18).
- Window length is configurable (default 300 s). Each segment holds `{seg_id, t0, t1,
  transcript_text, slide_ids[]}`.
- Slides per segment: every timed slide shown from the end of the previous segment to the end of
  this one (the first and last are open-ended), plus the slide on screen when it starts.
- Budget check: `transcript_text + slide text` must stay ≤ `max_segment_tokens` (5K) for Tiny
  Aya, estimated without a tokenizer (1 token per CJK character, per 3 other characters; Fuse
  loads no model). Split the window in half (by chunks) until it fits; a single chunk over budget
  is kept with a warning. Map logs the real token count and warns if it's over.

### 5.5 Map (Tiny Aya Global)
- One call per segment, temperature ~0.3, output in the target language, JSON only. The prompt
  repeats the target-language instruction as its **last line**: with it only at the top, Phase 0
  got English output for de (2/4) and ja (4/4), because the transcript at the end wins:
  `{"summary": str, "key_points": [str], "decisions": [str], "action_items": [{"owner": str|null, "task": str, "due": str|null}], "open_questions": [str], "terms": [str]}`.
  `open_questions` gives Reduce real material for its Open questions section, which the AMI
  `problems` reference is scored against (§18).
- The prompt says to use only the content in the segment and to prefer the slide text's spelling
  for names and numbers (speech recognition often gets these wrong).
- **Invalid JSON:** same rule as §5.3. Retry once with "Return only valid JSON". If it still fails,
  store the raw text as `summary`, leave the lists empty, set `raw: true` on that segment's entry,
  and log a warning. The retry prompt (`prompts/map_retry.md`) also ends with the target language.
- Due dates are copied as spoken, never computed (Phase 0 saw "end of next week" → "10月27日"). One bad segment never fails the stage.
- **Optional regional Map models (Phase 6, decision D18):** Cohere's regional Tiny Aya variants
  have the same size (3.35B), chat template and license (CC-BY-NC) as Global. The release
  describes their focus as Earth: Africa + West Asia, Fire: South Asia, Water: Asia-Pacific +
  Europe. The model cards are gated, so confirm this when adopting one. `config.py` gets
  `map_model_by_lang: dict[str, str]` (target language → repo), **empty by default**, so every
  language uses Global. A matching variant **replaces** Global for Map and for "Other (beta)"
  translation (§5.6), so memory use stays the same. Only languages where the variant wins the D18
  eval get an entry. The model repo + revision are part of Map's config hash (§3, §19), so
  switching a language's model re-runs only Map and the stages after it. MLX weights:
  `mlx-community/tiny-aya-fire` exists. Earth and Water need `mlx_lm.convert` (8-bit) from the
  pinned CohereLabs revision, uploaded to a personal Hub repo or converted locally on first use.
  Decide this in Phase 6 and document it in `NOTES.md`.
  **Access:** the CohereLabs originals are gated. Converting them needs the license accepted on
  the Hub and an HF token from the `HF_TOKEN` environment variable or `.env` (never committed,
  §6, §14 H2). The default models need no token: all six default MLX repos were ungated on
  2026-10-09.

### 5.6 Reduce (Command R7B with citations)
- Use the tokenizer's **`rag`** chat template, which is present in the MLX repo and supports
  `documents` and `enable_citations`. Each document:
  `{"title": "12:30–17:30 · slides 4,5", "snippet": <map notes + raw transcript>}`.
  **Every slide is its own document** (`"title": "slide 7: <title>"`, snippet = slide text +
  chart notes), aligned or not: in Phase 0, with slide text inside the segment's document, R7B
  cited only the segment (Phase 4a). Slide numbers shown to people are `slide_id + 1`.
- **Over the cap (until two-pass reduce, Phase 6):** if the prompt exceeds `max_input_tokens`,
  every segment's transcript is trimmed (25% at a time); if it still doesn't fit at 300
  characters per segment, the stage fails with a clear error. Map notes and slides are never trimmed.
- **Phase 0:** confirm the exact `apply_chat_template(..., documents=..., chat_template="rag", enable_citations=True)`
  arguments, and the citation format R7B produces (expected: `<co>span</co: 0:[i,j]>`).
- **Fixed section markers:** the prompt asks for five sections under fixed English markers, whatever
  the target language: `## SUMMARY`, `## KEY_POINTS`, `## DECISIONS`, `## ACTION_ITEMS`,
  `## OPEN_QUESTIONS`. The parser splits on these exact markers, so German or Japanese notes
  parse the same way as English ones. `render.py` shows each heading in the target language from
  a static table (`LABELS[lang][key]`: the five headings plus "Sources", "owner", "due" and
  "uncited", for all 23 languages; English if the language is missing). Missing or unknown
  markers: log them, and put unmatched text in Summary rather than dropping it.
- **Claims:** one per list item (continuation lines join the item; `- none` is skipped) and one
  per sentence in Summary (`.!?` + space, or `。！？`). A `<co>` span is never split, so its
  citations stay with the claim that contains it. A claim's citations are its spans' targets,
  first occurrence of each; the refined `t` comes from the first span citing that segment.
- **Action item line format:** one item per line, `- [owner: <name or ?>] [due: <date or ?>] <task>`
  (`?` = unknown). The parser fills `owner`/`due` (§19). If a line doesn't match, match the task
  text against `map.json` action items from the cited segments (token overlap ≥ 0.6) and copy their
  owner and due date; otherwise leave them null. Phase 0 got `Priya: screenshots by 20 October`
  without an example, so `reduce.md` shows two example lines.
- Citation parsing: regex `<co>(.*?)</co:\s*([^>]+)>` → list of document indices → `citations[]`.
- **Citation rules (every claim cites):**
  - A document index outside the documents sent is dropped and logged. It never becomes a link.
  - A claim left with no valid citation is kept, with `citations: []`. The Notes page shows it
    grayed out with an "uncited" badge (gray, §7), and the eval counts it (§18). Never invent a
    citation.
- **Citation time refinement** (pure Python, after parsing): a segment citation points at a window
  of up to 5 minutes, which is too coarse to play from. For each segment citation, score the cited
  span against every transcript chunk in that segment (the token overlap from §5.4) and store the
  best chunk's `t0` as the citation's `t`. If no chunk scores ≥ 0.1 (D19), use the segment's `t0`.
  Slide citations need no refinement. Two-pass citations (§4) are refined after marker resolution,
  against the resolved segments.
- **Rendering**, two modes from the same `notes.json`:
  - *In app* (Notes page, §7): `span [12:47](#seg-3) [slide 4](#slide-4)`, where the label is the
    refined `t` and the anchor is the segment.
  - *Portable* (`notes.md`, the Download button): citations as plain text `[12:47]` / `[slide 4]`
    (no in-app anchors, which are dead links outside the app), plus a **Sources** appendix with one
    entry per cited segment (time range, slide ids, first ~200 characters of its transcript) and per
    cited slide (number and title).
  - Both modes flag uncited claims (`[uncited]` in the target language; the app adds the badge),
    and action items show `(owner: …; due: …)` when known. Rendering is its own `render` stage
    (no model), so a label change re-renders without re-running R7B.
- **Two-pass reduce** for inputs over the 32K cap: see §4. Pass 2 uses `prompts/reduce_pass2.md`;
  the marker resolution is pure Python and unit tested (§8); the `long-en` eval checks it (§18).
- **Target languages:** the UI lists **R7B's 23 languages** (exact list from the model card,
  recorded in `config.py` in Phase 0). An **"Other (beta)"** option accepts any language Tiny Aya
  supports: Reduce then runs in English and Tiny Aya translates the notes section by section.
  Translation must **keep citations intact**: only the text between citation markers is sent for
  translation, and the markers are put back in the same order. Anything else would break links.

### 5.7 Ask
- Same documents as Reduce, the user's question, and the `rag` template with citations. Answer
  rendered with the same citation links, using the same citation rules and time refinement as
  §5.6. No re-processing needed.
- **Follow-ups:** the last 3 question/answer turns are sent as chat history (answers with
  citation markup removed, to save tokens), so "and who owns that?" works. History counts toward
  the ~32K cap and is dropped oldest-first if needed.
- **Answer language:** the language of the question, detected with `lingua-language-detector`
  (moved from the eval group to runtime deps for this). If detection is unsure, use the run's target
  language. The documents stay as they are; R7B answers across languages.
- **Long meetings:** if the documents exceed the ~32K cap, rank them for the question with an
  in-memory SQLite FTS5 table (BM25, stdlib; `trigram` tokenizer for zh/ja/ko) and send the
  top-ranked ones that fit. The Ask page then says "Answer based on the N most relevant
  segments". The ranking is pure Python and unit tested.

---

## 6. Project layout (uv, Python 3.13)

```
.gitignore           see §6 .gitignore below
CLAUDE.md            rules + index for implementation sessions (§21)
NOTES.md             Phase 0 findings and per-phase notes; results for §20 decisions
LICENSE              Apache-2.0 full text (§11)
NOTICE               copyright + third-party model license notice (§11)
pyproject.toml       includes `license = "Apache-2.0"`, `license-files = ["LICENSE", "NOTICE"]`
src/cohr/
  __init__.py
  atomic.py          temp file / temp dir + os.replace for every output (§3)
  config.py          pydantic settings: model repos + pinned revisions, window sizes, thresholds,
                     memory policy, storage paths (§12)
  storage.py         run dirs, source link/copy, disk-space checks, prune/delete, size report
  schemas.py         pydantic models for every run file + FORMAT_VERSION per stage (§19)
  models.py          ModelManager: lazy load, resident/sequential policy, memory stats, one lock
  ingest.py          ffmpeg wrappers: normalize, scene frames, pdf pages, dedupe
  vad.py             speech detection (Silero via mlx-audio) + silence fallback + chunk packing
  asr.py             Transcribe wrapper (mlx-audio or mlx-speech, §5.2)
  vision.py          Micro Vision wrapper + JSON repair
  fuse.py            segment windows + slide alignment + token budget split
  mapper.py          Tiny Aya per-segment notes
  reduce.py          R7B rag documents, generation, citation parsing
  render.py          notes.md rendering with anchors
  pipeline.py        stage runner: manifest, hashing, skip logic, progress callbacks, cancel
                     flag, interrupted-stage recovery, atomic writes, run log
  prompts/           vision.md  map.md  reduce.md  reduce_pass2.md  ask.md  translate.md  judge.md
                     (plain files; each hash feeds its stage's config hash, §3)
  cli.py             `cohr run <audio> [--slides …] [--video …] --lang en --target en`, `cohr eval`
  evals.py           eval runner: stage metrics, planted-fact matching, judge, baseline compare
app.py               Streamlit entry: page config, st.navigation, sidebar (§7)
app_pages/           new_run.py  notes.py  ask.py  runs.py  (search.py in Phase 7)
.streamlit/          config.toml: theme, upload limit, localhost (§7)
evals/                 quality evaluation (§18)
  README.md            how to run evals + dataset attribution (CC BY)
  sets/                quick.yaml, full.yaml: which items each set runs
  gen/                 planted-facts meeting generator (scripts, slides, truth.json)
  baseline.json        committed metrics the release gate compares against
  data/                generated + downloaded eval data (gitignored)
  reports/             per-run reports (gitignored)
scripts/
  spike.py           Phase 0 exit check: all models chained on fixtures, one cited note, memory
                     and timings (§9)
tests/
  fixtures/          generated, not committed if large (see §8)
  test_ingest.py  test_vad.py  test_fuse.py  test_align.py  test_ask_rank.py  test_pipeline.py  test_citations.py  test_render.py  test_storage.py  test_theme.py
  test_app.py  test_evals.py  test_schemas.py
  schemas/           committed JSON Schema snapshots of every run file (§19)
  test_pipeline_smoke.py
runs/                gitignored
```

**Python 3.13 is required:** `mlx-speech` 0.5.3 declares `Requires-Python >=3.13` and does not
install on 3.12 (checked 2026-10-05). `mlx`, `mlx-audio` and `mlx-speech` were installed and
imported on 3.13; `mlx-lm`, `mlx-vlm` and the rest are confirmed by `uv sync` in Phase 0.

**Dependencies:** `mlx`, `mlx-audio` (Silero VAD; possibly Transcribe, §5.2), `mlx-lm`, `mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@<commit sha pinned in Phase 0>`,
`mlx-speech>=0.5.3`, `streamlit>=1.65` (theme and layout keys were checked on 1.65), `huggingface_hub`, `soundfile`, `numpy`, `pillow`,
`imagehash`, `pydantic`, `pypdfium2` (PDF pages; Apache-2.0/BSD-3. **Not PyMuPDF**, which is AGPL-3.0; see §11),
`lingua-language-detector` (Ask answer language, §5.7, and the eval language check; Apache-2.0).
**System:** `ffmpeg` (already at `/opt/homebrew/bin/ffmpeg`) and `uv` (already installed).

### Development tooling

| Tool | Role | Setup |
|---|---|---|
| **uv** | Python 3.13, venv, lockfile, running commands | `uv init --package`, `.python-version` = 3.13, `requires-python = ">=3.13"`, commit `uv.lock` |
| **ruff** | lint + format | `[tool.ruff]` in `pyproject.toml`: `line-length = 100`, `target-version = "py313"`, rules `E,F,I,UP,B,SIM`, `extend-exclude = ["*.md"]` (ruff ≥0.17 formats code blocks in Markdown) |
| **ty** | type check | `[tool.ty]` in `pyproject.toml`; untyped MLX packages are fine, don't add stubs |
| **pytest** | tests | `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, marker `models`, default `-m "not models"` |

- Dev tools go in a dependency group, not in runtime deps:
  `uv add --dev ruff ty pytest`, which writes `[dependency-groups] dev = [...]`.
- Eval tools go in a separate group: `uv add --group eval jiwer pyyaml` (`lingua-language-detector`
  is a runtime dependency, above)
  (§18). CI installs every group (`uv sync --locked --all-groups`) so `ty check` can resolve
  `evals.py`'s imports; the eval *tests* still need no models.
- Everyday commands, all through uv (never a bare `pip` or the global `python`):
  ```
  uv sync                          # install runtime + dev
  uv run ruff format .             # format
  uv run ruff check --fix .        # lint
  uv run ty check                  # type check
  uv run pytest                    # unit tests (no models)
  uv run pytest -m models          # smoke test with real models (opt in)
  uv run streamlit run app.py      # UI
  uv run cohr run …                # CLI (entry point in [project.scripts])
  uv run --group eval cohr eval --set quick   # quality eval, models required (§18)
  ```
- **Gate for every phase:** `ruff format --check`, `ruff check`, `ty check` and `pytest` all pass.

### .gitignore

Short and project-specific instead of GitHub's ~200-line `Python.gitignore` template. Create it
**as the very first step of Phase 0, before any other file is created.** Comments must be on
their own lines (in `.gitignore`, `#` after a pattern is part of the pattern).

```gitignore
# --- Python / uv ---
.venv/
__pycache__/
*.py[cod]
dist/
build/
*.egg-info/

# --- Tool caches ---
.ruff_cache/
.pytest_cache/
.ty_cache/
.coverage
htmlcov/

# --- Pipeline outputs (§3, §12); includes runs/index.db (§13) ---
runs/

# --- Eval data and reports (§18); evals/baseline.json stays committed ---
evals/data/
evals/reports/
NOTES.local.md

# --- Media: never commit recordings or frames (privacy + size) ---
*.wav
*.mp3
*.m4a
*.aiff
*.flac
*.mp4
*.mov
*.mkv
*.webm

# --- Model weights: never redistribute (§11, CC-BY-NC) ---
*.safetensors
*.gguf
*.npz
models/
.cache/

# --- Secrets & local config ---
.env
.env.*
.streamlit/secrets.toml
.claude/settings.local.json

# --- macOS ---
.DS_Store
```

**Committed on purpose:** `uv.lock`, `.python-version` (CI `uv sync --locked`, §15),
`.streamlit/config.toml` (upload limit, §12), `.claude/settings.json` + `.claude/hooks/` (§14),
`NOTES.md` (Phase 0 findings), `LICENSE`, `NOTICE` (§11), `evals/baseline.json`, `evals/sets/`,
`evals/gen/` (§18).

The media and weights rules are a safety net: tests write to `tmp_path` and weights live in the HF
cache outside the repo, so normally nothing matches them. They protect against an accidental
`git add .` of a private recording or CC-BY-NC weights. `.ty_cache/` is harmless if ty doesn't
create it; drop the line if it never appears.

**Check (Phase 0):** `git check-ignore -v runs/x meeting.mp4 model.safetensors .env` prints a
rule for each; `git check-ignore uv.lock .claude/settings.json NOTES.md` prints nothing.

---

## 7. Streamlit UI (`app.py`)

- Load the **`developing-with-streamlit`** skill before writing UI code.
- `ModelManager` is created once with `@st.cache_resource`, which survives reruns and keeps
  reruns from loading the models twice.
- Run the pipeline in a background thread that reports progress through a queue. The UI polls
  with `st.status` and a progress bar per stage, so a rerun doesn't cut off a long job. Only one
  job at a time (enforced by the ModelManager lock). While a job runs, Ask on *other* runs waits
  for the lock and says so.
- **Job lifecycle:**
  - **Cancel** button: sets a `threading.Event` that stages check between chunks, frames and
    segments. The current stage is marked `cancelled`; its temp output is discarded (atomic
    writes, §3), and earlier stages stay valid.
  - **Crash or restart recovery:** the manifest stores each running stage's status plus a
    heartbeat timestamp updated every few seconds. On app start, a `running` stage with a heartbeat
    older than 60 s becomes `interrupted`, and the Runs page offers **Resume**, which re-runs from
    the first incomplete stage via the normal skip logic.
  - **Logging:** Python `logging` to `runs/<id>/log.txt` (stage start/end, timings, peak memory,
    warnings, tracebacks). Errors in the UI show a short message plus "see log".
- Pages, not tabs: `st.tabs` runs every tab's code on each rerun, and Notes, Ask and Runs each do
  real work. The app uses `st.navigation` with one file per page in `app_pages/` (see Layout below).
  1. **New run:** upload audio or video and slides, or give a local path; pick source language,
     target language, window length and memory policy; Run.
  2. **Notes:** notes rendered from `notes.json` (in-app mode, §5.6) next to a Sources panel.
     Clicking a citation jumps to its segment in Sources, which has an `st.audio` player plus the
     slide images. The player starts at the segment's earliest cited moment (refined `t`, §5.6).
     The block also lists every cited moment in that segment, and the drawer (Ask and the D14
     fallback) plays from the exact `t` of the clicked citation.
  3. **Ask:** chat over the selected run, with citations.
  4. **Runs:** list of previous runs with size, per-stage re-run buttons ("re-reduce with new
     prompt"), Delete and Prune media (§12), and the peak-memory chart for the selected run.
  5. **Search** (Phase 7, §13).
- Sidebar: navigation, the **current run** picker (a global filter used by Notes and Ask), a compact
  model and memory status line with the GPU limit warning, and the footer disclaimer (§11). No
  charts or tables in the sidebar.

### Layout that adapts to different displays

**Displays this app actually runs on:** it's local-only (`server.address = "localhost"`, below), so
"different displays" means different **window widths on the Mac**, not phones:
| Window | Typical width | Example |
|---|---|---|
| Narrow | < 640 px | Split View, Stage Manager tile, small window |
| Medium | 640–1200 px | Half of a laptop screen |
| Wide | 1200–1800 px | Full-screen MacBook Pro 14"/16" (1512/1728 pt) |
| Very wide | > 1800 px | External 27" or ultrawide monitor |

**Principle: use Streamlit's built-in responsive behavior; never measure the screen.** Python can't
see the viewport width (`st.context` has no width; checked in 1.65), and measuring it with a custom
JS component would add a rerun on every resize. Instead:
- `st.columns` (default `wrap=True`) **stacks vertically at ≤ 640 px**.
- `st.container(horizontal=True)` button rows **wrap onto new lines** when space runs out.
- `initial_sidebar_state="auto"` **hides the sidebar on small viewports** (opened via the menu).
- `width="stretch"` (the default) lets dataframes, charts, audio and images fill whatever column
  they're in. Never use the deprecated `use_container_width`.
- **Per-page layout:** `st.set_page_config(layout=…)` is called in each page file, so form and chat
  pages use `"centered"` (a built-in readable max width) and the two-panel pages use `"wide"`.

**Structure**
```
app.py                       st.set_page_config(page_title="cohr", initial_sidebar_state="auto")
                             st.navigation([...], position="sidebar"); sidebar run picker + status
app_pages/
  new_run.py                 layout="centered"
  notes.py                   layout="wide"
  ask.py                     layout="centered"
  runs.py                    layout="wide"
  search.py                  layout="wide"      (Phase 7)
```

**Per page**

*New run* (`centered`)
- `st.form`, so changing options doesn't rerun anything until Run.
- Source and target language: `st.columns(2)`, side by side, stacked when narrow.
- Window length, memory policy and source mode (§12) go in an `st.expander("Advanced")`.
- After Run: one `st.status` per stage, full width, with the progress bar inside.

*Notes* (`wide`)
- Header: run title, then an `st.container(horizontal=True)` with Download `notes.md`
  (`st.download_button`; the portable version, §5.6), Re-reduce and Open in Ask. The buttons wrap
  when narrow. If any claim is uncited, a gray caption says how many.
- Body: `st.columns([3, 2], gap="large")`.
  - Left, **notes**: the rendered notes inside `st.container(width=880)`, so lines stay readable
    on very wide monitors (~75–90 characters). Phase 5 check: at window widths below 880 px the
    container must shrink with the column. If it doesn't, use `width="stretch"` and accept long
    lines on very wide screens.
  - Right, **sources**: `st.container(height=…, border=True)`, scrolling independently, one
    block per segment with its anchor, time range, `st.audio(start_time=t0, end_time=t1)` and slide
    images (`width="stretch"`, `alt=` slide title). Unaligned slides come after.
  - At ≤ 640 px the columns stack: notes first, then sources. Citation anchors still jump to the
    right block, because they're anchors, not layout.
  - **Phase 5 check:** in-page anchors may scroll only the page, not the Sources panel's own
    scroll area. If a citation click doesn't bring its block into view, use the Ask pattern
    instead: citations become small buttons that open the right-side drawer with that segment.

*Ask* (`centered`)
- `st.chat_message` history, `st.chat_input` pinned at the bottom (built in).
- Under each answer: the cited sources as `st.pills`. Selecting one opens
  `@st.dialog(position="right")`, a side drawer the user can resize, with that segment's audio and
  slides. A drawer overlays the page instead of needing a column, so it works at every width.

*Runs* (`wide`)
- `st.dataframe` (stretch, `column_config` for size, date, status) with single-row selection.
- The selected run's actions go in an `st.container(horizontal=True)` (re-run stage, Prune media,
  Delete). Delete and Prune confirm in an `@st.dialog`.
- The selected run's peak-memory-per-stage bar chart (`st.bar_chart`, theme chart colors 1–4) is
  below the table.

**Not done (on purpose):** custom CSS media queries (the skill's no-CSS rule), viewport-measuring
components, more than 2 content columns anywhere, and separate "mobile" pages.

**Tests**
- `tests/test_app.py` uses `st.testing.v1.AppTest` with a fake ModelManager and a fixture run
  directory (no models; runs in CI): every page renders without exceptions, the Notes page has
  one Sources block per segment, and Ask shows pills for cited sources.
- **Manual check (Phase 5):** resize the browser window to ~420, 800, 1280, 1512 and 2560 px wide,
  in light and dark mode. At each width: nothing scrolls horizontally, the sidebar behaves (hidden
  when narrow), notes and sources stack at ≤ 640 px, button rows wrap, and the notes column stays
  readable at 2560. Claude in Chrome can take these screenshots (`resize_window`) if wanted.

### Theme: "Paper & Ink" (light + dark)

**Concept:** the app produces *notes*, so light mode is warm paper with ink-blue accents and dark
mode is the same desk at night: ink-dark background, the same blue raised in lightness. Citations
(`[12:30]`, `[slide 4]`) are links, so **`linkColor` is the most-used accent** and is tuned
separately from the button color in each mode.

**Rules (from the Streamlit skill):**
- Theme only through `.streamlit/config.toml`. **No custom CSS.**
- Define both `[theme.light]` and `[theme.dark]`. Streamlit then follows the OS setting by default,
  and users can switch in the settings menu. Defining only `[theme]` would lock the app to one mode.
- Not Cohere's brand colors (coral, dark green), per the §11 naming rule.

**How light and dark complement each other** (the same design, mirrored):
| Property | Light | Dark |
|---|---|---|
| Hue family | ink blue ~225° | same hue, lighter |
| Neutrals | warm grays (paper) | the same warm grays, inverted |
| Layering | sidebar *darker* than main (recessed) | sidebar *darker* than main (recessed) |
| Accent role split | button = link = mid blue | button = mid blue (white text), link = light blue (on dark bg) |
| Semantic colors | same 7 hues, darkened to ≥4.5:1 on paper | same 7 hues, lightened to ≥4.5:1 on ink |
| Chart order | blue, green, orange, violet, red, teal, gray | same order and hues |

**Fonts:** `font = "sans-serif"` and `codeFont = "monospace"`, Streamlit's **bundled** fonts.
No Google Fonts: the app must work offline (§1), and a fonts URL fails without network. For
Arabic, Chinese, Japanese, Korean, Greek and Vietnamese text, the browser automatically falls back
to macOS system fonts glyph by glyph, which covers all 14 Transcribe languages without bundling
large CJK fonts.

**`.streamlit/config.toml`** (one file, also holds the upload limit from §12 and the local-only
server address)
```toml
[server]
maxUploadSize = 4096                     # MB, §12
address = "localhost"                    # local only: no LAN access to recordings (§7 Layout)

[browser]
gatherUsageStats = false                 # Streamlit sends usage stats by default (checked 1.65)

[theme]
font = "sans-serif"                      # bundled, offline-safe
codeFont = "monospace"
baseRadius = "medium"
showWidgetBorder = true
showSidebarBorder = true
linkUnderline = false

[theme.light]
primaryColor = "#2F5BD3"                 # white text 5.90:1
backgroundColor = "#FBFAF7"              # warm paper
secondaryBackgroundColor = "#F0EEE8"
textColor = "#1E2024"                    # 15.6:1
linkColor = "#2A52BE"                    # citations, 6.6:1
codeBackgroundColor = "#F0EEE8"
borderColor = "#D9D5CB"
dataframeBorderColor = "#D9D5CB"
blueColor = "#2F5BD3"
greenColor = "#1F7A45"
orangeColor = "#A8560C"
violetColor = "#6E46C2"
redColor = "#C0322B"
grayColor = "#5E6168"
chartCategoricalColors = ["#2F5BD3", "#1F7A45", "#A8560C", "#6E46C2", "#C0322B", "#0E7C86", "#5E6168"]

[theme.light.sidebar]
backgroundColor = "#F0EEE8"
secondaryBackgroundColor = "#E6E3DB"

[theme.dark]
primaryColor = "#3D66DD"                 # white text 5.07:1, 3.6:1 vs background
backgroundColor = "#14161B"              # ink
secondaryBackgroundColor = "#1E2128"
textColor = "#E8E6E1"                    # 14.5:1
linkColor = "#8DAEFF"                    # citations, 8.3:1
codeBackgroundColor = "#1E2128"
borderColor = "#343842"
dataframeBorderColor = "#343842"
blueColor = "#8DAEFF"
greenColor = "#5CC98A"
orangeColor = "#F0A35E"
violetColor = "#B69CFF"
redColor = "#FF8A80"
grayColor = "#A3A7B0"
chartCategoricalColors = ["#8DAEFF", "#5CC98A", "#F0A35E", "#B69CFF", "#FF8A80", "#4FC7D0", "#A3A7B0"]

[theme.dark.sidebar]
backgroundColor = "#101216"
secondaryBackgroundColor = "#1A1D23"
```
All values were checked against Streamlit 1.65 `streamlit config show` (2026-10-05), and all ratios
are WCAG contrast against the mode's main background unless noted. Text, links and semantic colors
are all ≥4.5:1 (AA) in both modes.

**Semantic color use in the UI (same meaning in both modes):**
| Meaning | Color | Where |
|---|---|---|
| Citations, timestamps, slide refs | `linkColor` | Notes, Ask, Search |
| Decisions | green | `st.badge` / `:green[...]` |
| Action items | orange | `st.badge`, Runs table |
| Open questions | violet | `st.badge` |
| Errors, memory over limit | red | `st.error`, sidebar |
| Stage status / secondary text | gray | `st.caption` |
| Uncited claims (§5.6) | gray text + gray "uncited" badge | Notes, Ask |
| Peak memory per stage (4 models) | chart colors 1–4 | Runs page chart |

Icons use Material Symbols (`:material/...:`), and labels use sentence case.

**Test (`tests/test_theme.py`, no models, runs in CI):** parse `.streamlit/config.toml` with
`tomllib` and assert:
- `[theme.light]` and `[theme.dark]` set the **same keys**, and their `chartCategoricalColors`
  have the same length. This parity check is what keeps the two modes complementary when someone
  edits one later.
- Contrast: text vs. background ≥ 7:1, link and every semantic color vs. background ≥ 4.5:1,
  white vs. `primaryColor` ≥ 4.5:1, `primaryColor` vs. background ≥ 3:1, in both modes.
- No `fonts.googleapis.com` anywhere in the file, and `browser.gatherUsageStats` is `false`
  (offline and no-telemetry rules, §1).

**Manual check (Phase 5):** switch light ↔ dark in the settings menu and look at buttons, forms,
dataframes, code blocks, the Runs memory chart, badges, and a Notes page full of citation links.

---

## 8. Testing

- **Unit tests (no models):** slide text alignment (monotonic DP, threshold, skipped slides),
  two-pass citation marker resolution, Ask BM25 ranking under the cap, citation-safe translation
  splitting, pipeline atomic writes / cancel / interrupted-stage recovery, VAD chunk packing from synthetic speech runs (merge gaps, max length,
  long-run split, gaps recorded), relative silence threshold, silence-cut selection, frame dedupe, segment windowing and token
  budget split, citation regex and index mapping, render anchors, manifest skip logic, storage (link vs. copy, prune keeps notes usable, disk-space refusal).
- **Reduce output parsing (`test_citations.py`, `test_render.py`):** section markers in a
  non-English output (German fixture) land in the right sections, and unknown markers go to Summary;
  out-of-range document indices are dropped; uncited claims are kept with `citations: []`;
  citation time refinement picks the best-matching chunk and falls back to segment `t0` below
  the threshold; action item lines parse owner/due, with the `map.json` fallback; the portable
  `notes.md` has no `#seg-` links and has a Sources entry for every cited segment and slide;
  Map invalid-JSON fallback; Ask history trimming under the cap.
- **Unit tests must not need Metal or model weights:** they run on CI runners without a usable
  GPU (§15). Import `mlx`, `mlx_lm`, `mlx_vlm`, `mlx_audio` and `mlx_speech` lazily inside the model wrappers
  (`vad.py`, `asr.py`, `vision.py`, `mapper.py`, `reduce.py`, `models.py`), never at module top level in code
  that unit tests import.
- **Fixtures generated on the fly** (no binaries committed):
  - Audio: `say -o <tmp_path>/x.aiff "…"` → ffmpeg → wav (macOS TTS), two speakers via `-v`
    voices, ~2 min total, with 20 s of non-speech spliced in (ffmpeg `anoisesrc` noise +
    `sine` tones as "music") to exercise VAD gaps.
  - Slides: PIL draws 3 images with a title, bullets and a small table with known numbers.
- **Smoke test** (marked `@pytest.mark.models`, opt in): full run on the fixtures. Checks that
  the transcript contains known words, the slide JSON contains the table numbers, `notes.md`
  has at least one citation per section, and every citation resolves to an existing segment or slide.
- Lint, format and type-check with `ruff` and `ty` (commands and config in §6 Development tooling).

---

## 9. Implementation phases (acceptance criteria)

**Phase 0: checks before building (do first; record findings in `NOTES.md` and results in the §20
decisions table)**
- [x] `.gitignore` from §6 created and its `git check-ignore` check passes. This is the very first step,
      before any other file is created.
- [x] `CLAUDE.md` from §21 created and committed (right after `.gitignore`).
- [x] Claude Code hooks from §14 are in place, pass their pipe tests, and show in `/hooks`
      (restart the session after creating them). Do this before any other code.
- [x] **Minimal uv project:** `uv init --package --name cohr` with Python 3.13, then `uv add` the
      runtime dependencies (§6) and `uv add --dev` / `--group eval` the tool groups. Commit
      `pyproject.toml`, `.python-version` and `uv.lock`. The checks below need this; Phase 1
      builds the modules on top of it.
- [x] `uv sync` on Python 3.13 resolves every dependency (`mlx-lm`, `mlx-vlm` git, `mlx-audio`,
      `mlx-speech`, `streamlit`, …). If one doesn't support 3.13, record it and ask the user.
- [x] Each model loads and runs alone; record load time and peak memory.
- [x] **Pin revisions:** record the commit SHA of all six model repos (Transcribe, Micro Vision,
      Tiny Aya, R7B, Silero VAD, Qwen3 judge) as `revision=` in `config.py`, and pin `mlx-vlm` to
      the working git commit in `pyproject.toml`.
- [x] **Transcribe runtime comparison (§5.2):** run `mlx-audio` and `mlx-speech` on the same
      weights and the same fixture chunks. Record: loads the §2 repo or not, transcript accuracy
      (word error rate against the fixture script text), seconds per audio minute, peak memory.
      Pick `mlx-audio` unless `mlx-speech` is clearly better; record the decision and drop the loser.
- [x] Transcribe (chosen runtime): maximum input length, whether language auto-detection works,
      and whether the 4-bit `beshkenadze` repo works.
- [x] **Made-up text check:** transcribe 30 s of music, 30 s of room noise and 30 s of silence
      without VAD. Record any text produced; this is the evidence for the VAD default.
- [x] **VAD (§5.1a):** `mlx-community/silero-vad-v6` loads via `mlx_audio.vad` and its license on
      the Hub card is MIT. On a fixture with speech + music + silence: speech is kept, the
      non-speech parts land in `gaps`, and no chunk exceeds `max_chunk_s`. Compare with the
      silence fallback on the same file.
- [x] Micro Vision through mlx-vlm main: a slide image produces valid JSON with the prompt in §5.3.
- [x] R7B `rag` template: exact kwargs; citations appear in the output in the expected format.
      Record R7B's 23 supported languages (model card) for the target-language list (§5.6).
- [x] Tiny Aya: the JSON-only prompt follows the schema reliably in 3 languages.
- [x] All pipeline models loaded together: peak memory is under the GPU limit with a 32K R7B prompt.
- [x] Record seconds per audio minute for each stage, to check the §9 speed target.
- [x] License audit: record every runtime dependency's license in `NOTES.md` (from `uv pip show`
      / package metadata, or the repo when metadata is missing), including `mlx-speech` (MIT,
      checked 2026-10-05), `mlx-audio` (MIT) and the `mlx-vlm` git commit. Also confirm each model repo's license on its Hub card. Already resolved (2026-10-05):
      FLEURS and AMI are CC BY 4.0 (Hub cards; AMI annotations' `LICENCE.txt`), the judge model
      is Apache-2.0, and `lingua-language-detector` is Apache-2.0 (PyPI classifier + repo
      `LICENSE.txt`; the PyPI license field is just empty).
      Pass: no AGPL/GPL Python packages imported in-process (subprocess tools like ffmpeg are fine)
      and no dependency forbids Apache-2.0 distribution. Anything else goes to the user before Phase 1.
- [x] No PyTorch: after `uv sync`, run `uv tree --invert --package torch` (or `grep -n '^name = "torch' uv.lock`).
      If `torch` or `torchvision` is present, record which package pulls it in in `NOTES.md` and
      try to drop it (exclude the optional extra or pin a version). If it's unavoidable, confirm it is
      never imported at runtime (`python -X importtime … 2>&1 | grep torch` during a full run).
- [x] **Exit:** `scripts/spike.py` runs all five pipeline models **chained**, each one's real output
  feeding the next, on the §8 fixtures (2 min audio + 3 slide images):
  VAD → chunks → Transcribe → transcript; Micro Vision → slide JSON; the whole fixture as **one
  segment** (no Fuse logic yet) → Tiny Aya Map JSON → R7B `rag` with one document per segment and
  slide → notes → citation parse (§5.6 regex, or the fallback format below).
  It prints peak memory and seconds per stage, writes everything to a temp dir (never `runs/`),
  and **passes only if the notes contain at least one citation that resolves** to the segment or
  a slide. This is the first end-to-end result, three phases before Phase 4, so stages that don't
  fit together show up now. Throwaway code: Phase 1+ rebuilds each stage properly.

### Phase 0 go/no-go: fallbacks if a check fails

The design rests on a few assumptions that only Phase 0 can test. If one fails, apply the
fallbacks **in order**. **Auto** = stays within the design: apply it, record it in `NOTES.md` and
§20, and continue. **Ask** = changes the design: stop, show the user the finding and the options,
and wait. Any fallback that's applied is written into the affected plan sections in the Phase 0
PR (§21).

| If Phase 0 finds… | Fallbacks, in order | Who decides |
|---|---|---|
| **Transcribe:** neither `mlx-audio` nor `mlx-speech` loads the §2 repo, or WER on the fixture > 15% | (1) The other runtime. (2) The 4-bit `beshkenadze` repo (**Phase 0: unusable**, mlx-audio-swift format; skip). (3) Another MLX conversion of the same model on the Hub. No non-MLX fallback (§1 all-MLX rule) | 1–2 Auto, 3 Ask |
| **R7B citations (D7):** the `rag` template through `mlx-lm` gives no parseable `<co>` citations in 3 tries on the spike | (1) Fix the kwargs or chat-template call (check the tokenizer config). (2) **Prompt-based citations:** documents numbered `[D1]…[Dn]` in a plain prompt, a required `[Dn]` marker after each claim, a new regex in `reduce.py`, the same §5.6 rules (drop invalid, flag uncited). Ask uses the same format | 1 Auto, 2 Ask |
| **Micro Vision:** won't run on `mlx-vlm` main, or gives invalid JSON on 2 of 3 fixture slides after the retry | (1) The 4-bit repo. (2) An older `mlx-vlm` commit that supports it. (3) **No slide reading:** slides stay images, citable as `[slide N]` with title "Slide N" and no text; slide alignment (§5.4) and the slide eval variants are skipped until fixed | 1–2 Auto, 3 Ask |
| **Tiny Aya JSON:** < 90% of outputs valid on the 3-language check | (1) Temperature 0 + one example in the prompt. (2) The 4-bit Global repo (rules out a conversion problem). (3) R7B does Map too (one fewer model resident, slower) | 1–2 Auto, 3 Ask |
| **Memory:** all models together go over the GPU limit with a 32K R7B prompt | (1) Lower the R7B cap to 24K (two-pass reduce starts sooner, §4). (2) `sequential` becomes the default; Ask loads R7B on demand and says so. (3) 4-bit fallbacks (§2) | 1 Auto, 2–3 Ask |
| **Speed (D10):** projected > 30 min per audio hour (1.5× the target) | (1) Fuse window 300 → 600 s (half the Map calls). (2) Skip Map when the R7B input fits without it (short meetings): Reduce reads transcripts + slides directly. (3) Relax the §9 target, with the reason recorded | 1 Auto, 2–3 Ask |
| **VAD:** Silero won't load through `mlx_audio.vad` | The `silence` chunker (§5.1a) becomes the default; the made-up text results decide whether that's acceptable | Ask |
| **Any blocker not listed** (Python 3.13 support, license problem, `torch` unavoidable) | Already covered in the checklist above: record and ask | Ask |

**Phase 1: scaffold + ingest** (done 2026-10-11, NOTES.md): GitHub repo created and configured (§17), then tool config
(ruff, ty, pytest) in the Phase 0 `pyproject.toml`, `schemas.py` with snapshot tests (§19), config (including `seed`, §3), prompts folder, pipeline runner
(atomic writes, cancel, recovery, log), ingest functions; the §6 gate passes; CI workflow (§15) green on a PR.
**Phase 2: perception** (done 2026-10-11, NOTES.md): `vad.py` (chunks.json) and `asr.py` (slices `audio.wav` by offsets), `vision.py`, ModelManager with
pinned revisions and a free-disk check before downloads, `transcript.json` and `slides.json`, plus skip logic.
**Phase 3: fuse + map** (done 2026-10-11, NOTES.md): slide text alignment, `segments.json` and `map.json` (with `open_questions` and the
invalid-JSON fallback, §5.5); token budget enforced.
**Phase 4: reduce + render**, split in three PRs:
- **4a** (done 2026-10-11, NOTES.md): `notes.json` and portable `notes.md` with working citations (fixed section
  markers, citation rules, time refinement, action item owner/due, §5.6); CLI end to end on fixtures.
- **4b-1: evals without the judge.** Planted-facts generator (`evals/gen/`, seeded: script + `truth.json`,
  `say` voices, PIL slides, ffmpeg noise/tones/silence) for the four `quick` variants; exact metrics (fact
  recall, citation accuracy, uncited claims, made-up items, VAD gap text, WER, slides, output language,
  cost); cross-language exact-field scoring (code + unit tests; the cross-language variants run in `full`,
  Phase 6); `cohr eval --set quick` with reports and the baseline comparison (no committed baseline until
  Phase 6); `tests/test_evals.py`. First `quick` numbers recorded; D11 re-check, D12 (a second `quick`
  run with another window), D19. Fix the 4a Map error (owner's due date given to another action item,
  `prompts/map.md`) with the made-up-items count before and after.
- **4b-2: judge.** Qwen3 judge wrapper (§18), support rate, judge recall for cross-language text facts,
  `evals/gen/judge_calibration.json` (30 pairs: Claude drafts labels from 4b-1 outputs, the user reviews
  them), D15.
**Phase 5: Streamlit UI:** "Paper & Ink" theme and adaptive layout (§7) with `test_theme.py`,
`test_app.py` and the window-width check, all four pages, background job, progress, Ask (follow-ups and answer language, §5.7); upload limit and local
path input (§12); Runs page shows each run's size, with Delete and Prune media.
**Phase 6: hardening:** sequential memory policy, inputs over 2 h (two-pass reduce with
marker-based citations), Ask ranking for long meetings, "Other (beta)" target languages, error
messages for missing ffmpeg or models, a cache size note for the README; optional regional Map
models (§5.5): `map_model_by_lang` filled only for languages that pass D18 (may stay empty);
`full` eval set and the
committed `evals/baseline.json` with release gate (§18); bump to `1.0.0` to release v1 (§16).
**Phase 7 (after v1): cross-meeting search:** SQLite + FTS5 index and Search page per §13.

**Definition of done:** a 30–60 min real recording plus slides produces cited notes in a chosen
target language. All citations resolve. The §18 quick set meets its v1 targets. A 60 min
meeting processes end to end in **≤ 20 min** (`resident` policy; initial target, revisited
with the Phase 0 timings). Re-running only Reduce takes under a minute. Peak memory
stays under the limit with the `resident` policy.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `mlx-vlm` main changes or breaks Compass support | Pin the git commit that works in Phase 0 |
| Small models produce malformed JSON | Low temperature, schema in the prompt, one retry, fall back to raw text |
| R7B citation format differs from expected | Phase 0 check; parser isolated in `reduce.py` with tests; prompt-based citations as the fallback (§9 go/no-go) |
| A Phase 0 assumption fails (runtime, vision, memory, speed) | §9 go/no-go table: ordered fallbacks, each marked Auto or Ask |
| R7B ignores section markers or leaves claims uncited | Fixed markers + fallback to Summary; uncited claims flagged, counted and gated (§5.6, §18) |
| Transcribe makes up text from music/noise/silence | VAD default (§5.1a) skips non-speech; Phase 0 made-up text check |
| A runtime or package lacks Python 3.13 support | Phase 0 `uv sync` check; ask the user before pinning older versions |
| Speech recognition mangles names and numbers | Map prompt prefers slide spelling; terms list passed to Reduce |
| Background thread + MLX + Streamlit reruns | One ModelManager lock, `cache_resource`, one job at a time |
| Memory pressure from other apps | `sequential` policy switch; sidebar warning when free memory is low |
| First-run downloads (~15 GB) | Download step in the UI with progress via `huggingface_hub.snapshot_download`; free-disk check first (§12) |
| Hub repo updated under us | Pinned `revision` per model in config (§12) |
| Symlinked source moved or deleted | Store source size + mtime + sha256 in manifest; warn and offer re-link; derived stages still work from `audio.wav` |
| `runs/` grows without limit | Size shown per run; Delete / Prune media actions; `cohr prune` CLI |

---

## 11. Licensing

Two layers with separate licenses.

**1. This project's code: Apache-2.0**
- Covers everything in the repo: `src/cohr`, `app.py`, tests, scripts, CI, hooks.
- Chosen because it matches Transcribe, Micro Vision, Streamlit and `huggingface_hub`, is compatible
  with the MIT-licensed MLX packages, and includes an explicit patent grant.
- Files: `LICENSE` (full Apache-2.0 text), `NOTICE` (below), and in `pyproject.toml`
  `license = "Apache-2.0"` + `license-files = ["LICENSE", "NOTICE"]` (PEP 639). No per-file headers.

**2. Model weights: the model authors' licenses, never redistributed**
| Model | License |
|---|---|
| Cohere Transcribe 2B | Apache-2.0 |
| North Micro Vision Instruct 2.4B | Apache-2.0 |
| Tiny Aya Global 3.35B | **CC-BY-NC-4.0** |
| Command R7B | **CC-BY-NC-4.0** |
| Silero VAD v6 | MIT |
| Qwen3-4B-Instruct (eval judge only, §18; not used by the app) | Apache-2.0 |
| Tiny Aya Earth / Fire / Water (optional Map models, §5.5; off by default) | **CC-BY-NC-4.0** |

- Weights are downloaded at runtime by `snapshot_download` onto the user's machine. They are
  **never committed, never attached to releases (§16), never bundled**. That keeps the project from
  redistributing CC-BY-NC material, so the NC terms don't attach to the code.
- Consequence: **with the default models, the app is for personal and research use only.** Using
  it commercially requires replacing Map and Reduce with commercially licensed models (config only,
  no code change).

**`NOTICE` file (content)**
```
cohr
Copyright 2026 Daryl Lim

Licensed under the Apache License, Version 2.0 (see LICENSE).

This is an independent project. It is not affiliated with, endorsed by, or
sponsored by Cohere. "Cohere", "Command", "Aya" and "North" are names of their
respective owners and are used here only to identify the models this software runs.

This software downloads and runs third-party machine-learning models at runtime.
Those models are NOT part of this software and are governed by their own licenses:
  - Cohere Transcribe 2B, North Micro Vision Instruct 2.4B: Apache-2.0
  - Silero VAD v6: MIT
  - Tiny Aya Global 3.35B, Command R7B: CC-BY-NC-4.0 (non-commercial use only)
With the default configuration, use of this software is therefore limited to
non-commercial purposes. Check each model's license on its Hugging Face page.
```

**README:**
- A line directly under the title: "Independent project. Not affiliated with or endorsed by Cohere."
- A "License" section with the same two-layer explanation.

**Streamlit sidebar footer:** "Unofficial; not affiliated with Cohere. Default models are
CC-BY-NC: non-commercial use only."

**Naming rule:** refer to models by their names to say what runs. Don't use Cohere logos,
brand colors or wording like "official" or "by Cohere". The repo name stays `cohr`; the disclaimer
covers the resemblance.

**Dependency rule:** in-process Python dependencies must be permissive (MIT, BSD, Apache-2.0,
ISC, PSF, LGPL), or MPL-2.0 when transitive and used unmodified (user-approved in Phase 0 for
`certifi` and `tqdm`; MPL's copyleft covers only its own files). AGPL/GPL tools are allowed only as separate programs called via subprocess (e.g.
ffmpeg). That's why PDFs use `pypdfium2`, not PyMuPDF (AGPL-3.0). Phase 0 checks this (§9);
new dependencies added later need the same check.

---

## 12. Storage

Two stores, both on local disk. **v1 has no database**: each run's `manifest.json` and JSON
stage outputs are the source of truth. A rebuildable search index comes later (§13).

**Model weights: Hugging Face cache**
- Location: `HF_HOME` (default `~/.cache/huggingface`). Set via `COHR_HF_HOME` in config so the
  models can live on an external drive. The app passes `cache_dir` explicitly; it doesn't mutate
  the env of other tools.
- Each model in config has a **pinned `revision`** (commit sha, recorded in Phase 0) passed to
  `snapshot_download`. This stops silent weight changes and duplicate copies in the cache.
- Before any download: `shutil.disk_usage(cache_dir).free` must be ≥ the missing weights plus a
  2 GB margin, or the UI stops with a clear message (needed vs. free).
- **Offline at runtime:** `cohr/__init__.py` sets `HF_HUB_DISABLE_TELEMETRY=1` before
  `huggingface_hub` is imported. The ModelManager resolves every model with
  `snapshot_download(..., local_files_only=True)` and loads it from the local path, so no request
  reaches the Hub during normal use (Phase 2: `HF_HUB_OFFLINE` is read once at import, so setting
  it later wouldn't work; the smoke test runs with it set to prove nothing calls out). Only the
  explicit download step (UI button or `cohr models download`) talks to the Hub.
- Sidebar and `cohr models` CLI show which repos are cached and their size on disk.
  Deleting models is left to `hf cache delete` (documented in the README, not reimplemented).

**Runs: `runs/` (gitignored, location set by `COHR_RUNS_DIR`)**
- **Source handling:** `source_mode = "link"` (default) symlinks originals into `source/`;
  `"copy"` copies them. Streamlit uploads have no original path, so they're always written
  (streamed in chunks) into `source/`. The manifest records each source's size, mtime and sha256.
  If a linked original goes missing, the run still works from `audio.wav` / `frames/`; only
  stages that read `source/` (ingest) need it again.
- **One audio copy:** `audio.wav` only (~115 MB/h). Chunks are offsets in `chunks.json`.
- **Frames:** PNG, ≤1600 px long edge (~0.3–1 MB each, ~20–60 per hour).
- **Expected size per hour of meeting:** ~115 MB audio + ~10–50 MB frames + <5 MB JSON/text,
  so about **130–170 MB/h** excluding `source/` copies.
- **Uploads:** `.streamlit/config.toml` sets `server.maxUploadSize = 4096` (MB); same file as the
  §7 theme. The New run page
  also accepts a **local file path**, which is the recommended route for large videos: no upload,
  no in-memory copy, and it's linked instead of copied.
- **Free-disk check before a run:** estimate (source size if copying + 200 MB/h of audio) and
  refuse to start if free space is below estimate + 1 GB.

**Retention**
- No automatic deletion.
- Runs page and CLI:
  - **Delete run**: removes the whole run directory (confirm first).
  - **Prune media**: removes `source/` copies, `audio.wav` and `frames/`, keeping all JSON,
    `notes_raw.txt` and `notes.md`. Notes, Ask and re-Reduce still work, because Reduce/Ask
    only read `segments.json`, `slides.json`, `map.json` and `transcript.json` (citation time
    refinement, §5.6). The audio player and slide thumbnails
    are disabled, and re-running Ingest/Perceive needs the source again. The manifest marks the
    run `pruned`.
  - `cohr runs` lists runs with size; `cohr prune <run>` / `cohr rm <run>`.

**Tests:** fixtures and test runs go under pytest's `tmp_path`, never into `runs/`.

---

## 13. Database: search index (Phase 7, after v1)

**Decision:** v1 is files only. After v1, add a SQLite index that is **disposable and always
rebuildable from `runs/*/manifest.json` + the JSON stage outputs**. It is never the source of
truth, and nothing is written to it that isn't on disk first.

**Why SQLite + FTS5:** it's in the Python stdlib (`sqlite3`, FTS5 compiled in on macOS builds
from uv-managed Python; confirm with `PRAGMA compile_options`), needs no server, and adds
full-text search across meetings, which files alone can't do quickly.

**Location:** `runs/index.db` (gitignored with `runs/`).

**Schema (sketch)**
```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);          -- schema_version
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, title TEXT, created_at TEXT, duration_s REAL,
  source_lang TEXT, target_lang TEXT, status TEXT,             -- complete | partial | pruned
  size_bytes INTEGER, manifest_hash TEXT
);
CREATE TABLE segments (
  run_id TEXT REFERENCES runs ON DELETE CASCADE, seg_id INTEGER,
  t0 REAL, t1 REAL, PRIMARY KEY (run_id, seg_id)
);
CREATE TABLE action_items (
  run_id TEXT REFERENCES runs ON DELETE CASCADE, seg_id INTEGER,
  owner TEXT, task TEXT, due TEXT
);
CREATE VIRTUAL TABLE docs USING fts5(
  run_id UNINDEXED, seg_id UNINDEXED, kind UNINDEXED,          -- transcript | slide | note
  text, tokenize = 'unicode61 remove_diacritics 2'
);
```

**Sync rules**
- `cohr index rebuild` drops and rebuilds everything from disk.
- After any stage finishes, Delete, or Prune media: re-index that run inside one transaction
  (delete its rows, insert fresh). A run is re-indexed only if its `manifest_hash` changed.
- On app start: if `schema_version` differs or the file is missing or corrupt, rebuild silently.
- Deleting `index.db` is always safe.

**UI:** a **Search** page (`app_pages/search.py`, §7) with a query box and filters (date range, language, kind). Results show
the snippet (FTS5 `snippet()`), the run, and a link to `[mm:ss]` in that run's Notes page.
Action items get a cross-meeting table filterable by owner.

**Limits (accepted):**
- FTS5 `unicode61` doesn't segment Chinese or Japanese (no spaces between words), so search in
  those languages is weak. Option later: the `trigram` tokenizer for those docs.
- Keyword search only. Semantic search across meetings would need an embedding model and a
  vector store (sqlite-vec); out of scope until there's a clear need.

**Tests:** `test_index.py`: rebuild from fixture runs matches incremental updates; delete/prune
updates rows; corrupt or old-schema DB triggers a rebuild.

---

## 14. Claude Code hooks (development guardrails)

Hooks make the §6 gate and the project's rules automatic during implementation sessions,
so they don't depend on Claude remembering them. They're **set up first in Phase 0**.

**Files (committed, shared):**
```
.claude/settings.json        hook registration
.claude/hooks/guard-bash.sh  PreToolUse  Bash
.claude/hooks/protect.sh     PreToolUse  Edit|Write|MultiEdit|NotebookEdit
.claude/hooks/ruff-file.sh   PostToolUse Edit|Write|MultiEdit
.claude/hooks/gate.sh        Stop
```
Scripts are `bash` + `jq` (`/usr/bin/jq` is present), `chmod +x`, read the hook JSON on stdin,
and block by printing the reason to stderr and exiting 2 (Claude sees the reason and adjusts).

**`.claude/settings.json`**
```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash",
        "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/guard-bash.sh" }] },
      { "matcher": "Edit|Write|MultiEdit|NotebookEdit",
        "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/protect.sh" }] }
    ],
    "PostToolUse": [
      { "matcher": "Edit|Write|MultiEdit",
        "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/ruff-file.sh", "timeout": 60 }] }
    ],
    "Stop": [
      { "hooks": [{ "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/gate.sh", "timeout": 300 }] }
    ]
  }
}
```

### H1. `guard-bash.sh`: enforce uv, block risky commands, ask before outward actions (PreToolUse, Bash)
Reads `.tool_input.command`. **Block** rules are checked first, then **ask** rules.

**Block** (exit 2) when it matches:

| Pattern | Reason shown to Claude |
|---|---|
| `pip install`, `pip3 install`, `python -m pip` | "Use `uv add` / `uv add --dev`." |
| `sudo` anywhere (covers `sysctl iogpu.wired_limit_mb`) | "Never run sudo; §4 says the GPU limit is documented, never applied. Ask the user." |
| `rm -rf` touching `runs`, `.cache/huggingface` or `HF_HOME` | "Use `cohr rm`/`cohr prune` or `hf cache delete`; ask the user first." |
| `pytest … -m models` / `-m "models"` | "Model smoke test loads ~15 GB; ask the user before running it." |
| `cohr eval` with `--set full` | "Full eval takes ~1 h and loads all models; ask the user first." |
| `git push --force` / `-f` | "Force push is not allowed." |

**Ask** (forces a confirmation prompt, **even in auto mode**) for anything that changes GitHub:

| Pattern | Prompt reason |
|---|---|
| `git push` | "Pushes to GitHub." |
| `gh repo create`, `gh repo edit`, `gh repo delete` | "Changes the GitHub repository." |
| `gh api` with `-X`/`--method` `PUT`, `POST`, `PATCH`, `DELETE` | "Writes via the GitHub API (e.g. branch protection)." |
| `gh release` (create/edit/delete/upload) | "Changes GitHub releases." |
| `gh pr merge`, `gh pr close` | "Merges or closes a pull request." |

Ask works by printing JSON to stdout and exiting 0, not exit 2. Read-only `gh` commands
(`gh repo view`, `gh pr view`, `gh api` GETs, `gh run list`) and `gh pr create` are allowed
without asking. Opening a PR is reviewable and reversible.

Sketch:
```bash
#!/usr/bin/env bash
cmd=$(jq -r '.tool_input.command // ""')
block() { echo "Blocked: $1" >&2; exit 2; }
grep -Eq '(^|[;&|[:space:]])(pip3?|python3? -m pip) install' <<<"$cmd" && block "use uv add / uv add --dev"
grep -Eq '(^|[;&|[:space:]])sudo[[:space:]]' <<<"$cmd" && block "no sudo; ask the user (see PLAN §4)"
grep -Eq 'rm[[:space:]]+-[a-zA-Z]*r[a-zA-Z]*f?.*(runs|\.cache/huggingface|HF_HOME)' <<<"$cmd" && block "use cohr rm/prune or hf cache delete; ask first"
grep -Eq 'pytest.*-m[[:space:]]+"?models' <<<"$cmd" && block "model smoke test loads ~15 GB; ask the user first"
grep -Eq 'cohr[[:space:]]+eval.*--set[[:space:]=]+full' <<<"$cmd" && block "full eval takes ~1 h; ask the user first"
grep -Eq 'git[[:space:]]+push.*(--force|[[:space:]]-f([[:space:]]|$))' <<<"$cmd" && block "no force push"

ask() {
  jq -n --arg r "$1" '{hookSpecificOutput: {hookEventName: "PreToolUse",
    permissionDecision: "ask", permissionDecisionReason: $r}}'
  exit 0
}
grep -Eq '(^|[;&|[:space:]])git[[:space:]]+push' <<<"$cmd" && ask "Pushes to GitHub."
grep -Eq 'gh[[:space:]]+repo[[:space:]]+(create|edit|delete)' <<<"$cmd" && ask "Changes the GitHub repository."
grep -Eq 'gh[[:space:]]+api.*(-X|--method)[[:space:]=]*(PUT|POST|PATCH|DELETE)' <<<"$cmd" && ask "Writes via the GitHub API."
grep -Eq 'gh[[:space:]]+release[[:space:]]+(create|edit|delete|upload)' <<<"$cmd" && ask "Changes GitHub releases."
grep -Eq 'gh[[:space:]]+pr[[:space:]]+(merge|close)' <<<"$cmd" && ask "Merges or closes a pull request."
exit 0
```
This is a guardrail against mistakes, not a security boundary; regexes are best effort.

### H2. `protect.sh`: protect generated and private files (PreToolUse, edits)
Reads `.tool_input.file_path` (or `.tool_input.notebook_path`) and blocks edits to:
- `uv.lock`: "managed by uv; change `pyproject.toml` or run `uv add`".
- `runs/**`: "run outputs are produced by the pipeline, not edited by hand".
- `.env`, `.env.*`: "may hold secrets (HF token)".
- anything under the HF cache directory.

### H3. `ruff-file.sh`: format + lint each edited Python file (PostToolUse)
- Skips non-`.py` files and exits 0 if `pyproject.toml` doesn't exist yet (only at the very start
  of Phase 0, before `uv init`).
- Runs `uv run --frozen ruff format "$f"` then `uv run --frozen ruff check --fix "$f"`.
- If lint errors remain, prints them to stderr and exits 2, so Claude fixes them right away.
- Per-file only, so it stays fast (well under a second).

### H4. `gate.sh`: run the §6 gate before Claude finishes (Stop)
- Exits 0 immediately if `.stop_hook_active` is true (prevents an infinite loop), if there's
  no `pyproject.toml`, or if `git status --porcelain -- '*.py' pyproject.toml` is empty (nothing
  Python changed, e.g. a planning-only turn).
- Otherwise runs, stopping at the first failure:
  `uv run --frozen ruff format --check .` → `uv run --frozen ruff check .` → `uv run --frozen ty check`
  → `uv run --frozen pytest -q -x -m "not models"`.
- On failure: the last ~40 lines of output go to stderr and it exits 2, so Claude keeps working
  until the gate passes. Unit tests must stay fast (no models), which §8 already requires.

### Not included (on purpose)
- No hook runs the model smoke test or downloads models: too slow and memory-heavy.
- No auto-commit hook: commits stay explicit.
- No hook edits `PLAN.md` or `NOTES.md`.

### Verify (part of Phase 0)
Pipe sample JSON into each script and check the exit codes:
```bash
echo '{"tool_input":{"command":"pip install numpy"}}' | .claude/hooks/guard-bash.sh; echo $?   # 2
echo '{"tool_input":{"command":"uv add numpy"}}'      | .claude/hooks/guard-bash.sh; echo $?   # 0
echo '{"tool_input":{"command":"git push origin main"}}' | .claude/hooks/guard-bash.sh          # JSON with "ask"
echo '{"tool_input":{"command":"git push -f origin x"}}' | .claude/hooks/guard-bash.sh; echo $? # 2 (block wins)
echo '{"tool_input":{"command":"gh repo view"}}'       | .claude/hooks/guard-bash.sh            # no output
echo '{"tool_input":{"file_path":"uv.lock"}}'         | .claude/hooks/protect.sh;    echo $?   # 2
echo '{"stop_hook_active":true}'                       | .claude/hooks/gate.sh;       echo $?   # 0
```
Then restart Claude Code (hooks are read at session start) and confirm all four appear in `/hooks`.

---

## 15. GitHub Actions CI

One workflow that runs the same gate as the H4 Stop hook (§14) on every push and PR, so `main`
stays green whether the change came from Claude, you or anyone else.

**Prerequisite:** the GitHub repo exists and is pushed (§17 steps 1–3).

**Runner choice:** `macos-15` (Apple Silicon, arm64). MLX wheels and `mlx-audio`/`mlx-speech`/`mlx-vlm`
target macOS arm64, so `uv sync --locked` needs to resolve the same platform as the M2 Max.
Hosted macOS runners don't expose a usable Metal GPU and have ~7 GB RAM, so **CI never loads
models**: unit tests only (§8). macOS runner minutes are billed at 10× Linux on private repos
and free on public ones.

**`.github/workflows/ci.yml`**
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
permissions:
  contents: read
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
env:
  HF_HUB_OFFLINE: "1"          # no model downloads in CI, ever
  ALLOW_TORCH: "0"             # set to "1" only if Phase 0 records torch as unavoidable (NOTES.md)
jobs:
  check:
    runs-on: macos-15
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6   # use the current major at implementation time
        with:
          enable-cache: true
      - name: Install ffmpeg
        run: command -v ffmpeg || brew install ffmpeg
      - name: Sync (lockfile must be current; all groups so ty sees eval imports)
        run: uv sync --locked --all-groups
      - name: No PyTorch (§1 all-MLX rule)
        run: |
          if [ "$ALLOW_TORCH" != "1" ] && grep -q '^name = "torch' uv.lock; then
            uv tree --invert --package torch
            echo "::error::torch is in uv.lock; see PLAN §1 / §9 Phase 0"; exit 1
          fi
      - name: Format
        run: uv run ruff format --check .
      - name: Lint
        run: uv run ruff check --output-format=github .
      - name: Type check
        run: uv run ty check
      - name: Unit tests
        run: uv run pytest -q -m "not models"
```

**Notes**
- `uv sync --locked` fails if `uv.lock` doesn't match `pyproject.toml`, which catches
  dependency changes made without `uv add`.
- `ruff --output-format=github` shows lint errors as inline annotations on the PR diff.
- `say` (fixture audio) exists on macOS runners; ffmpeg is installed if missing.
- `mlx-vlm` comes from git (pinned commit, §10), so `uv sync` clones it; the uv cache keeps that fast.
- Pin action versions to full commit SHAs once the workflow is working (supply-chain hygiene).

**Repository settings:** branch protection requiring `check` is set up in §17 step 5.

**Not included (on purpose):** model smoke tests (no GPU, too little RAM, ~15 GB downloads),
publishing to PyPI, Linux or Intel matrix (the app is Apple Silicon only), scheduled runs.

**Optional later:** `.github/dependabot.yml` with the `github-actions` ecosystem (weekly) to
keep the action versions current.

---

## 16. Automatic GitHub releases on version bump

**Rule:** bumping `[project].version` in `pyproject.toml` and merging to `main` creates tag
`v<version>` and a GitHub Release, **only after the CI `check` job passes** on that commit.
Pushes that don't change the version do nothing (the tag already exists).

**Single source of version:** `pyproject.toml` only. Code reads it with
`importlib.metadata.version("cohr")`; there's no hard-coded `__version__`. Start at `0.1.0`
(set by `uv init`), so the first push to `main` with CI green releases `v0.1.0`.

**Bumping (local, by you or by Claude when asked):**
```
uv version --bump patch     # 0.1.0 → 0.1.1   (fixes)
uv version --bump minor     # 0.1.1 → 0.2.0   (new phase / feature)
uv version --bump major     # → 1.0.0         (definition of done, §9)
```
`uv version` also updates `uv.lock`, so `uv sync --locked` in CI keeps passing. Commit both files.
Claude doesn't bump the version unless asked.

**Job added to `.github/workflows/ci.yml`** (same workflow, so `needs: check` gates it):
```yaml
  release:
    needs: check
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest          # building sdist/wheel needs no MLX, and Linux minutes are cheap
    timeout-minutes: 10
    permissions:
      contents: write               # this job only; the workflow default stays read
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0            # all tags, for the downgrade check
      - uses: astral-sh/setup-uv@v6
      - name: Read version
        id: v
        run: |
          V=$(python3 -c 'import tomllib;print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')
          echo "version=$V" >> "$GITHUB_OUTPUT"
          if git rev-parse -q --verify "refs/tags/v$V" >/dev/null; then
            echo "exists=true" >> "$GITHUB_OUTPUT"; echo "v$V already released; skipping."
          else
            echo "exists=false" >> "$GITHUB_OUTPUT"
          fi
      - name: Refuse downgrade
        if: steps.v.outputs.exists == 'false'
        run: |
          LATEST=$(git tag -l 'v*' | sed 's/^v//' | sort -V | tail -1)
          if [ -n "$LATEST" ] && [ "$(printf '%s\n' "$LATEST" "${{ steps.v.outputs.version }}" | sort -V | tail -1)" != "${{ steps.v.outputs.version }}" ]; then
            echo "::error::version ${{ steps.v.outputs.version }} is lower than latest tag v$LATEST"; exit 1
          fi
      - name: Build
        if: steps.v.outputs.exists == 'false'
        run: uv build                # dist/*.whl + dist/*.tar.gz
      - name: Create release
        if: steps.v.outputs.exists == 'false'
        env:
          GH_TOKEN: ${{ github.token }}
          V: ${{ steps.v.outputs.version }}
        run: |
          PRE=""; case "$V" in *a*|*b*|*rc*|*dev*) PRE="--prerelease";; esac
          gh release create "v$V" dist/* --target "$GITHUB_SHA" --title "v$V" --generate-notes $PRE
```

**Behavior**
- **Idempotent:** re-runs and later pushes skip when the tag exists.
- **Gated:** no release unless format, lint, type check and unit tests pass on that exact commit.
- **Pre-releases:** versions like `0.3.0rc1` or `0.3.0b2` are marked as pre-releases.
- **Notes:** GitHub generates them from merged PRs and commits since the previous tag.
  Optional: `.github/release.yml` to group notes by PR label (feature / fix / docs).
- **Assets:** the wheel and sdist are attached for convenience. They aren't published to PyPI:
  `mlx-vlm` is a git dependency (PyPI rejects those) and two models are CC-BY-NC (§11).
- Tags created with `GITHUB_TOKEN` don't trigger other workflows, so there's no loop.

**Repository settings:** covered by §17. With branch protection on `main`, nothing extra is
needed; the job only creates a tag and release, it never pushes commits. If tag protection rules
exist for `v*`, allow GitHub Actions to create them.

---

## 17. GitHub repository setup

One-time setup, done at the **start of Phase 1** (step 5 after the first CI run). Every step is
outward-facing, so **Claude shows each command and runs it only after the user confirms**. The
§14 H1 ask rules enforce this with a confirmation prompt, even in auto mode.
`gh` is installed and logged in as `darylalim` (checked 2026-10-05).

**Visibility: public (decided 2026-10-05).** Reasons: GitHub Actions macOS minutes are free for
public repos and billed at 10× Linux for private ones (§15). Branch protection on private repos
needs a paid plan (step 5). The repo holds no weights, recordings or secrets (§6 .gitignore, §11,
§14 H2). The name `darylalim/cohr` was free on 2026-10-05.

**1. Create the repo and push**
```bash
gh repo create darylalim/cohr --public --source . --remote origin   # no --push
git push -u origin main                          # first push sets the default branch: main
git push -u origin plan/tier-a-meeting-notes     # then open a PR from it into main
```
Don't use `--push` on `gh repo create`: it pushes only the checked-out branch, and the first branch
pushed becomes the repo's default branch. Confirm with
`gh repo view --json defaultBranchRef --jq .defaultBranchRef.name` → `main`.

**2. Description and topics**
```bash
gh repo edit darylalim/cohr \
  --description "Local, offline meeting notes for Apple Silicon: transcribe recordings and slides with Cohere's open models on MLX; every claim cites its timestamp or slide. Unofficial, not affiliated with Cohere. Non-commercial with default models." \
  --add-topic meeting-notes,transcription,speech-to-text,summarization,citations \
  --add-topic mlx,apple-silicon,local-llm,offline,macos \
  --add-topic cohere,command-r,aya,rag,multilingual,vision-language-model,streamlit
```
17 topics (limit 20). If trimming, drop `macos` and `summarization` first. Avoid `whisper`,
`ollama` (not used) and broad ones like `ai`, `llm`, `python`.

**3. Merge settings** (squash merges keep `main` linear and make the generated release notes in
§16 one line per PR)
```bash
gh repo edit darylalim/cohr \
  --enable-squash-merge --enable-merge-commit=false --enable-rebase-merge=false \
  --delete-branch-on-merge --enable-wiki=false --enable-projects=false
```

**4. Actions permissions:** keep the default workflow token permission at **read**
(Settings → Actions → General). The `release` job asks for `contents: write` itself (§16).

**5. Branch protection on `main`** (after CI has run at least once, so the `check` status exists)
```bash
gh api -X PUT repos/darylalim/cohr/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": { "strict": true, "contexts": ["check"] },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```
- Requires the `check` job (§15) to pass and the branch to be up to date before merging.
- No required reviews: solo project. `enforce_admins: false` lets you bypass in an emergency.
- Works on the free plan because the repo is public.

**6. Verify**
```bash
gh repo view darylalim/cohr --json description,repositoryTopics,visibility,deleteBranchOnMerge
gh api repos/darylalim/cohr/branches/main/protection --jq '.required_status_checks.contexts'   # ["check"]
```
Then open a trivial PR and confirm that CI runs and that merging is blocked until `check` passes.

---

## 18. Quality evaluation

Unit tests (§8) check the mechanics, and the smoke test checks that citations *resolve*. This
section measures whether the notes are **right**: did they capture what mattered, do citations point
to the right moment, and is everything stated supported?

### Core design: planted-facts meetings
Synthetic meetings where every fact and the time it's spoken are known, so the main metrics are
exact checks with no human labels and no LLM judge.

**Generator (`evals/gen/`, deterministic from a seed):**
1. Write a meeting script with N decisions, M action items (owner + due date), K numbers, and
   filler talk. Facts come from small templates and word lists, so `truth.json` is exact.
2. Speak it with macOS `say` using 2–3 voices; in non-English variants, voices of that language
   (`say -v '?'` lists them; Phase 0 records which of the 14 languages have voices).
3. Draw matching slides with PIL: titles, bullets, a table containing some of the numbers.
4. Splice in non-speech (ffmpeg `anoisesrc` noise, `sine` tones as music, silence).
5. Save `truth.json`:
   `{facts: [{id, kind: decision|action|number, text, owner?, due?, value?, t0, t1, slide?}],
     speech_spans, gap_spans, slides: [{slide_id, title, cells}]}`.

**Variants (each a separate item):**
| Variant | What it tests |
|---|---|
| `clean-en` | Baseline |
| `noisy-en` | VAD and transcription with background noise and music gaps |
| `pdf-slides-en` | Audio + a PDF deck (no video): slide **text alignment** (§5.4); metric = share of slides placed in the window where they're discussed |
| `slide-conflict-en` | A number is *misspoken* in speech but correct on the slide (e.g. "forty" vs `14`). Correct output uses the slide value (§5.5 rule). |
| `xling-{es,ja}` → target `en` | Source language ≠ target language |
| `en` → target `{de,ar}` | Output in a non-English target, including right-to-left text |
| `long-en` (~2.5 h, built by repeating sections with new facts) | Two-pass reduce (§4) |

### Metrics
| Layer | Metric | Scoring |
|---|---|---|
| VAD (§5.1a) | Speech-frame F1 vs. `speech_spans`; **text produced per minute of gap** | Exact |
| Transcribe | WER (CER for zh/ja/ko) on synthetic items and **FLEURS** test clips | `jiwer` |
| Slides | Title exact match, table-cell accuracy, number recall | Exact |
| Notes: **fact recall** | Share of planted facts found in the right section (decisions → Decisions, …) | Normalized match: numbers exact, owners exact, text via token overlap ≥ 0.6. **Cross-language** (notes language ≠ script language): see below |
| Notes: **citation accuracy** | Share of matched facts whose citation's refined jump time `t` (§5.6) is within ±15 s of the fact's `t0` (D19), or that cite its slide. A time-range overlap test would pass almost any citation, since segments span up to 5 min | Exact |
| Notes: **uncited claims** | Claims with `citations: []` (§5.6) | Exact (count) |
| Notes: **made-up items** | Decisions/action items in the notes that match no planted fact | Exact (count) |
| Notes: **support rate** | Share of claims whose cited segments support them | LLM judge (secondary) |
| Output language | Notes are in the target language | `lingua-language-detector` |
| Ask | Planted-fact questions: answer contains the fact *and* cites the right segment | Exact |
| Realism (`full` only) | Coverage of the **AMI** reference decisions/actions | LLM judge |
| Cost | Seconds per audio hour per stage; peak memory per stage | Measured |

**Cross-language fact matching** (`xling-*`, `en → de/ar`, D18): token overlap against the
script's text can't work when the notes are in another language. For these items:
- Numbers, owners and due dates are matched exactly (names and digits survive translation;
  numbers are normalized to digits first).
- A fact with no such field (a pure-text decision) counts as found if the judge says a claim in the
  right section covers it (prompt: fact text + section claims → `{"covered": bool}`).
- The report shows the exact-field recall and the judge recall separately. Only exact-field recall
  counts toward the gate until D15 passes.

### Judge (secondary metrics only)
- **Not a model the pipeline uses**, to avoid self-preference: `mlx-community/Qwen3-4B-Instruct-2507-4bit`
  (Apache-2.0, ~2.3 GB, MLX via `mlx-lm`). It's loaded only during eval, after the pipeline models
  are released (sequential policy, §4).
- Prompt: claim + cited segment text → `{"supported": "yes|partial|no", "reason": str}`, JSON only,
  temperature 0.
- **Calibration:** 30 hand-labeled claim/segment pairs (`evals/gen/judge_calibration.json`). Record
  the judge's agreement; if it's below 80%, judge metrics are reported but excluded from the gate.
- Optional `--judge claude` for the public and synthetic sets only (needs an API key). User
  recordings are never sent to any judge.

### Eval sets
| Set | Contents | Time | When |
|---|---|---|---|
| `quick` | `clean-en`, `noisy-en`, `pdf-slides-en`, `slide-conflict-en` | ~5 min | After prompt or model changes; before every commit to `reduce.py`/`mapper.py` prompts |
| `full` | All variants + 14-language FLEURS sample (20 clips each) + 5 AMI meetings | ~1 h | Before every version bump (§16) |

External data is downloaded at eval time into `evals/data/` (gitignored). Nothing from it is
committed or redistributed.

| Data | Source | License | What we use |
|---|---|---|---|
| FLEURS | HF dataset `google/fleurs`, `snapshot_download(allow_patterns=["data/<lang>/test.tsv", "data/<lang>/audio/test.tar.gz"])` for the 14 languages | CC BY 4.0 | 20 test clips per language + reference transcripts |
| AMI audio + transcripts | HF dataset `edinburghcstr/ami` (parquet, `ihm` config) | CC BY 4.0 | 5 meetings |
| AMI reference summaries | Official annotations: `https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip` (22.9 MB; `LICENCE.txt` inside) | CC BY 4.0 | `abstractive/<meeting>.abssumm.xml` (142 meetings) |

**AMI join and scoring:**
- Join audio and summaries on meeting ID (e.g. `ES2004a`); use only meetings present in both.
  The 5 meetings are fixed in `evals/sets/full.yaml`.
- Each `abssumm.xml` has `<abstract>`, `<decisions>`, `<actions>` and `<problems>`, each a list of
  `<sentence>` elements. Map them to our sections: `decisions` → Decisions, `actions` → Action
  items, `problems` → Open questions; `abstract` → Summary (judged for coverage only).
- Metric: per section, the share of reference sentences the judge says are covered by our notes.
- Parse with stdlib `xml.etree` (files are ISO-8859-1 encoded). Use the official zip, **not**
  third-party Hub re-uploads of AMI summaries or QMSum mirrors: their licensing and processing are
  unclear.

### Reports and the release gate
- `cohr eval --set quick|full` writes `evals/reports/<YYYYmmdd-HHMMSS>.json` + `.md`: metrics per
  item and per layer, the model repos + revisions, prompt hashes, and the git sha.
- **Attribution (CC BY requirement):** every report that includes external data ends with a fixed
  footer: "Evaluation data: AMI Meeting Corpus (University of Edinburgh et al., CC BY 4.0);
  FLEURS (Google, CC BY 4.0). Data was not modified or redistributed." The same line goes in
  `evals/README.md`. Synthetic-only `quick` reports don't need it.
- It compares with the committed `evals/baseline.json` and **fails (exit 1)** if:
  fact recall drops > 5 points, citation accuracy drops > 3 points, made-up items or uncited
  claims increase, VAD gap text rate increases, or any stage gets > 25% slower.
- `cohr eval --update-baseline` rewrites the baseline. Only commit it with the reason in the commit
  message (e.g. "new reduce prompt: recall +4, accuracy −1").
- **Release gate:** before `uv version --bump` (§16), run `full` and attach the report summary to
  the PR. This is manual: CI has no GPU (§15).

**v1 targets (`quick` set):** fact recall ≥ 85%, citation accuracy ≥ 90% (±15 s), made-up decisions/action
items = 0, uncited claims = 0, VAD gap text = 0 words, notes in the target language 100%, processing ≤ 20 min per
audio hour (§9), slide alignment ≥ 80% (`pdf-slides-en`).

### Human review (your own recordings; never committed)
After a real meeting, five questions in about five minutes, logged to `evals/reports/human.jsonl`
(gitignored): anything important missing? anything wrong? 5 random citations: right moment?
readable? right language? Comparing these with the automatic metrics shows whether the planted-facts
scores track real quality.

### Tests (no models, CI)
`tests/test_evals.py`: fact matching (number normalization, owner match, token overlap,
cross-language exact-field matching), citation jump-time tolerance, baseline comparison thresholds, and generator determinism (same seed → same
`truth.json`).

---

## 19. Data contracts and run versioning

Every stage reads the previous stage's files, and runs are kept forever (§12). So the file
formats need **one definition** and a **version**, or a format change in a later release breaks
every older run.

### One definition: `src/cohr/schemas.py`
Pydantic models for every run file. Stages read and write **only** through them
(`Model.model_validate_json` / `model_dump_json`); no stage touches raw dicts.

| File | Model | Key fields |
|---|---|---|
| `manifest.json` | `Manifest` | `app_version`, `run_id`, `created_at`, `inputs[]` (kind, path, stored, mode, size, mtime, sha256), `config`, `models{stage: {repo, revision}}`, `stages{name: {status, format_version, config_hash, input_hash, started_at, heartbeat_at, finished_at, peak_mem_bytes, error?}}`, `pruned` |
| `chunks.json` | `ChunksFile` | `chunker`, `chunks[{chunk_id, t0, t1}]`, `gaps[{t0, t1}]` |
| `frames/frames.json` | `FramesFile` | `frames[{frame_id, path, pts?, phash, source, page?}]` |
| `transcript.json` | `Transcript` | `language`, `runtime`, `items[{chunk_id, t0, t1, text}]` |
| `slides.json` | `SlidesFile` | `slides[{slide_id, frame_id, t?, t_source, align_score?, title, text_md, tables_md[], chart_notes, visual_notes, raw?}]` |
| `segments.json` | `SegmentsFile` | `segments[{seg_id, t0, t1, chunk_ids[], transcript_text, slide_ids[]}]`, `unaligned_slide_ids[]` |
| `map.json` | `MapFile` | `target_language`, `notes[{seg_id, summary, key_points[], decisions[], action_items[{owner?, task, due?}], open_questions[], terms[], raw: bool}]` (v2) |
| `notes.json` | `NotesDoc` | `sections{summary, key_points, decisions, action_items, open_questions}` as lists of `{text, citations[{kind: seg|slide, id, t?}]}` (`t` = refined jump time, seg only; `citations: []` = uncited, §5.6); action items also have `owner?`, `due?`; `passes: 1|2`, `language`, `raw` (R7B reply) (v2) |

`notes.json` is new: the **parsed** reduce output. `notes.md` is rendered from it, and Notes, Ask,
Search and the evals read `notes.json`, never by re-parsing Markdown.
All times are seconds as floats; all ids are ints; enums are `Literal[...]`.

### Versioning
- Each stage has a `FORMAT_VERSION: int` constant next to its model. It's stored per stage in
  the manifest and **included in that stage's config hash** (§3). Bumping it automatically
  invalidates that stage and every later stage through the existing skip logic. No separate
  migration code is needed.
- `app_version` (from `importlib.metadata`, §16) is recorded for display and debugging only.
- **Opening an older run:** validate each file with its model.
  - Formats match: normal.
  - Older format, media present: the Runs and Notes pages show "Made with an older version" and a
    **Re-process from <first outdated stage>** button. Earlier stages are reused.
  - Older format, run **pruned** (no media), and an outdated stage needs media (Ingest, VAD,
    Perceive): **read-only legacy mode**. Show `notes.md` as stored (Markdown always renders), with
    Ask and re-Reduce disabled and a short explanation.
  - A file that fails validation for any other reason is treated as corrupt: the stage is marked
    `failed` with the validation error in `log.txt`, and offered for re-run.

### Guard against silent format changes
- `tests/schemas/<file>.json` holds the committed `model_json_schema()` of every model.
- `tests/test_schemas.py` fails if a model's JSON Schema differs from its snapshot **while its
  `FORMAT_VERSION` is unchanged**. To change a format: edit the model, bump `FORMAT_VERSION`,
  regenerate snapshots with `uv run python -m cohr.schemas --write-snapshots`, and commit all
  three together.
- Also tested: round-trip (dump → load) of fixture files, and that an older-version fixture run
  triggers "re-process from the right stage".

---

## 20. Open decisions

Everything the plan leaves to measurement, in one place. The implementation session fills in the
**Result** column (and the details in `NOTES.md`) as each is decided, then updates the affected
section.

| # | Decision | Default until decided | How it's decided | When | Result |
|---|---|---|---|---|---|
| D1 | Transcribe runtime: `mlx-audio` vs `mlx-speech` (§5.2) | `mlx-audio` | Same-weights comparison: loads, WER, speed, memory | Phase 0 || **mlx-speech**: WER 0.11, 1.96 s/audio min, 5.5 GB; mlx-audio can't load the int8 checkpoint (gibberish) |
| D2 | Transcribe max input length → `max_chunk_s` (§5.1a) | 30 s | Longest clip transcribed without truncation | Phase 0 || 30 s kept; no truncation up to 60 s (mlx-speech chunks long input internally) |
| D3 | Source-language auto-detection (§5.2) | User picks the language | Try `language` omitted on 3 languages | Phase 0 || No auto-detection in either runtime: user picks the language |
| D4 | 4-bit fallbacks work with chosen runtimes (§2) | 8-bit only | Load + one run each | Phase 0 || Not usable: the 4-bit Transcribe repo is mlx-audio-swift format; not needed (8.6 GB headroom) |
| D5 | VAD parameters (§5.1a) | threshold 0.5, 250/100/30 ms, merge < 0.5 s | Speech/gap F1 on the VAD fixture | Phase 0, re-check on eval || Defaults kept: speech F1 0.979, 0 s detected in the noise/music gap |
| D6 | Model revisions + `mlx-vlm` commit (§12) | — | Record SHAs that pass Phase 0 | Phase 0 || 7 repo SHAs + mlx-vlm `952d4f6` (NOTES.md 2026-10-10) |
| D7 | R7B `rag` kwargs, citation format, 23 languages (§5.6) | `<co>…</co: 0:[i]>` | Run the template | Phase 0 || Works as planned: `rag` + `documents` + `enable_citations=True`; `<co>…</co: 0:[i,j]>`; 23 languages (NOTES.md) |
| D8 | Which of the 14 languages have macOS `say` voices (§18) | en, es, ja, de | `say -v '?'` | Phase 0 || All 14 have a voice; ar, el, pl, vi only one each |
| D9 | `torch` in the lockfile (§1, §15) | Must be absent | `uv tree --invert --package torch` | Phase 0 || Absent from `uv.lock`; never imported |
| D10 | Speed target realistic (§9) | ≤ 20 min per audio hour | Phase 0 per-stage timings | Phase 0 || Realistic: ≈ 9–10 min per audio hour projected |
| D11 | Slide alignment threshold (§5.4) | 0.15 | Maximize `pdf-slides-en` accuracy | Phase 3, re-check Phase 4b-1 | **0.15 provisional**: on fixture text, discussed slides score 0.23–0.56, an undiscussed slide 0.125; 0.25 drops one. Final on `pdf-slides-en` (Phase 4b-1) |
| D12 | Fuse window length (§5.4) | 300 s | Fact recall vs. speed on `quick` | Phase 4b-1 | |
| D13 | Notes column `width=880` shrinks below 880 px (§7) | 880 | Resize check | Phase 5 | |
| D14 | Anchors scroll the Sources panel (§7) | Anchors | Click check; fallback = drawer buttons | Phase 5 | |
| D15 | Judge agreement ≥ 80% (§18) | Judge metrics excluded from gate | 30 labeled pairs | Phase 4b-2 | |
| D16 | Eval gate thresholds (§18) | recall −5, accuracy −3, speed +25% | Run-to-run noise on 3 repeated `quick` runs: threshold ≥ 2× noise | Phase 6 | |
| D17 | FTS5 available in uv's Python (§13) | Assumed | `PRAGMA compile_options` | Phase 7 | |
| D18 | Regional Tiny Aya variant per target language (§5.5) | Global for every language (`map_model_by_lang` empty) | Planted-facts `en → target <lang>` for 2 languages per variant's region, Global vs variant, scored with cross-language matching (§18): adopt if exact-field fact recall improves by ≥ 2× the D16 noise and output-language accuracy is no worse | Phase 6 | |
| D19 | Citation refinement: min chunk score and the eval's jump tolerance (§5.6, §18) | score ≥ 0.1, ±15 s | Share of planted facts whose refined `t` lands within tolerance on `quick`; check that 15 s isn't hiding misses (look at the error distribution) | Phase 4b-1 | |

Decided already (kept for reference): Python 3.13 (§6), public repo (§17), Apache-2.0 (§11),
VAD on by default (§5.1a), files-only v1 (§12), "Paper & Ink" theme (§7).

---

## 21. Handoff to implementation sessions

Implementation happens in new Claude Code sessions. A fresh session loads `CLAUDE.md`
automatically but not this plan, so the rules that apply to every change live in `CLAUDE.md`, and
the plan stays the full design reference.

### Workflow
- **One phase per session, one PR per phase** (branch `phase-N-<slug>`). Large phases may take
  several sessions; each ends with a pushed branch and a short entry in `NOTES.md`.
- At the start of a session: read `CLAUDE.md`, then the plan sections for the current phase, then
  the latest `NOTES.md` entry.
- At the end: tick the phase's checklist items in this plan, fill in any §20 results, update
  `NOTES.md`, make sure the gate passes (the H4 hook enforces it), and open the PR (pushing asks
  first, H1).
- If implementation shows the plan is wrong, **update the plan in the same PR** and say so in
  the PR description. The plan stays the source of truth.

### `CLAUDE.md` (content, created in Phase 0)
```markdown
# cohr: cited meeting notes on MLX

Local Streamlit app: recording + slides → multilingual notes where every claim cites
[mm:ss] or [slide N]. Design: PLAN.md (full). Progress and findings: NOTES.md.
Current phase: see the last entry in NOTES.md.

## Commands (always through uv)
uv sync --all-groups · uv run ruff format . · uv run ruff check --fix . · uv run ty check
uv run pytest (no models) · uv run streamlit run app.py · uv run cohr run … · uv run cohr eval --set quick

## Hard rules
- Python 3.13, uv only. Never pip. Add deps with `uv add` (dev: `--dev`, eval: `--group eval`).
- All model inference on MLX (mlx-audio / mlx-speech, mlx-vlm, mlx-lm). No torch (CI fails on it).
- Import mlx, mlx_lm, mlx_vlm, mlx_audio, mlx_speech only inside functions in the model
  wrappers (vad, asr, vision, mapper, reduce, models). Unit tests run in CI without a GPU.
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

## Where things are (PLAN.md)
Architecture §3 · Memory §4 · Stages §5 · Layout & tooling §6 · UI, layout, theme §7 ·
Tests §8 · Phases §9 · Storage §12 · Hooks §14 · CI §15 · Releases §16 · Evals §18 ·
Schemas §19 · Open decisions §20

## Each session
Start: CLAUDE.md → PLAN sections for the phase → last NOTES.md entry.
End: tick PLAN checklist, fill §20 results, add a NOTES.md entry, gate green, open PR.
```
Keep it under ~80 lines; details belong in the plan.

### Kickoff prompt for the Phase 0 session (paste into a new session)
```text
We're starting implementation of cohr. Read PLAN.md fully; it is the source of truth.
First create branch phase-0-checks from plan/tier-a-meeting-notes (main doesn't have the plan yet).
Do Phase 0 (§9) in order:
1. Create .gitignore (§6) and run its git check-ignore check.
2. Create CLAUDE.md from §21 and NOTES.md (empty template: date, phase, findings, decisions).
3. Create the Claude Code hooks (§14), run the pipe tests, then tell me to restart the session.
After the restart, continue with the rest of Phase 0: the uv project, uv sync on 3.13, the model
checks, pinning revisions, and scripts/spike.py (chained end to end). If a check fails, follow
the §9 go/no-go table: apply "Auto" fallbacks, stop and ask me on "Ask" ones. Record every
finding in NOTES.md and every result in the §20 table. Ask me before downloading the models (~15 GB) and before anything that
pushes to GitHub. Stop at the end of Phase 0 and summarize what was decided and what changed in
the plan.
```

### Kickoff prompt for later phases (template)
```text
Continue cohr: Phase <N> (<name>). Read CLAUDE.md, then PLAN.md sections <list>, then the latest
NOTES.md entry. Work on branch phase-<N>-<slug>. Follow the phase's acceptance criteria in §9.
When done: tick the checklist, update §20 and NOTES.md, make sure the gate passes, and open a PR
(ask before pushing).
```

### Kickoff prompt for Phase 4b-1
```text
Continue cohr: Phase 4b-1 (evals without the judge). Read CLAUDE.md, then PLAN.md sections
§9 (the 4b-1 bullet), §18 (all of it except the Judge subsection, AMI and FLEURS), §5.4, §5.5, §5.6
(citation refinement), §19, §20 (D11, D12, D19) and §8. Then read the two latest NOTES.md entries:
the 4a entry has the Map due-date error, and the 4b planning entry has the split.
Work on branch phase-4b-1-evals. Follow the 4b-1 acceptance criteria in §9:
1. Planted-facts generator in evals/gen/. The same seed must give the same script and truth.json.
   It uses `say` voices, PIL slides and ffmpeg noise/tones/silence, and builds the four quick
   variants (clean-en, noisy-en, pdf-slides-en, slide-conflict-en). Generated audio and slides go
   only to gitignored paths (evals/data/). Never commit them.
2. Exact metrics: fact recall, citation accuracy (refined t within ±15 s, or the right slide),
   uncited claims, made-up items, VAD gap text, WER, slides, output language, and cost.
   Cross-language exact-field scoring gets code and unit tests only; those variants run in
   Phase 6.
3. `cohr eval --set quick`: writes reports to evals/reports/ and compares against a baseline.
   Don't commit evals/baseline.json; that waits for Phase 6.
4. tests/test_evals.py as listed in §18 "Tests". No models in these tests.
5. Run quick and record the first numbers. Then decide D11 (on pdf-slides-en), D12 (do a second
   quick run with another Fuse window) and D19 (look at the citation error distribution, not just
   the pass rate).
6. Fix the Map due-date error in prompts/map.md. Record the made-up-items count before and after,
   both from quick runs on the same seed.
Running `cohr eval --set quick` is fine. Ask me before anything in `full`, before `pytest -m
models`, and before new model downloads. Eval deps (jiwer, pyyaml, lingua) are already installed.
Check the license of any new dependency (§11).
When done: tick the checklist, update §20 and NOTES.md, make sure the gate passes, and open a PR
(ask before pushing). Stop there; 4b-2 (the judge) is a separate session.
```
