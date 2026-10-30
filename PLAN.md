# Plan: Cited Meeting Notes (Tier A pipeline)

A local Streamlit app that turns a **recording plus slides** (or a screen-recording video) into
**multilingual notes in which every claim cites its timestamp or slide**. It uses four Cohere
models plus a small voice-detection model, all on MLX and all fitting in memory together on an
**M2 Max with 32 GB**.

Status: Phase 0 done 2026-10-10 (all checks passed; findings in `NOTES.md`, results in §20).
Phase 1 done 2026-10-11 (scaffold + ingest; §17 GitHub setup complete, CI green).
Phase 2 done 2026-10-11 (perception: VAD, Transcribe, Vision, ModelManager; smoke test passes offline).
Phase 3 done 2026-10-11 (fuse + map: slide text alignment, segments, Tiny Aya Map).
Phase 4a done 2026-10-11 (reduce + render). Phase 4b-1 done 2026-10-11 (planted-facts evals,
`cohr eval --set quick`, D11/D12/D19; fixes for ingest scene detection, Map due dates, Reduce
markers and citation jump times). Phase 4b-1b done 2026-10-11 (Map number guard, D20).
Phase 4b-2 done 2026-10-11 (judge: support rate, judge recall, D15 passed at 80.0%).
Phase 5a done 2026-10-11 (UI without Ask: theme, job layer, New run / Notes / Runs; D13, D14 pass).
Phase 5b done 2026-10-11 (Ask page and drawer, Notes Beside/Below toggle; D22, D23; browser
checks passed).
Phase 6a done 2026-10-11 (Ask eval, D16 thresholds, first committed `evals/baseline.json`).
Phase 6b done 2026-10-11 (planted open questions, 3 near-misses per item; Ask near-miss 0/12 →
7/12; Map open-question rule, made-up open questions 5 → 0; D24).
Phase 6c done 2026-10-12 (Reduce one fact per item; two-pass reduce and Ask FTS5 ranking for
long meetings; `long` set with `long-en`; VAD per 10-minute block; D17, D25–D27).
Phase 6b-2 done 2026-10-13 (Map fixes; D20 narrowed, D28). Phase 6d-1 done 2026-10-13 (notes
in other languages; D29, D30). Phase 6d-2 done 2026-10-13 (Spanish and Japanese meetings,
numbers read in the meeting's language; D18 Fire not adopted).
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
  both passes** (§5.6): pass 1 is a cited reduce per group of consecutive segments (with their
  aligned slides; a group's prompt fills up to 40% of the cap, D26) with `prompts/reduce_part.md`,
  and its citations are written into the group summary as inline markers `[seg-N]` /
  `[slide-N]` after every claim. Both long-meeting prompts list decisions, action items and open
  questions before key points, and key points are figures only; both passes, like a single
  pass, may write up to `reduce.max_tokens` = 8192 tokens (6c raised two-pass to 4096 when
  2048 stopped notes inside KEY_POINTS; 6e-2 raised every pass to 8192, D32).
  Pass 2 (`prompts/reduce_pass2.md`) cites group summaries and the slides aligned to no segment;
  each pass-2 citation of a group resolves, in order, to the markers inside the cited span
  (R7B copied them in 38 of 108 spans with the first prompt, none with the final one), else the citations of the best-matching
  pass-1 claim (token overlap ≥ 0.5), else all segments of that group (coarse but valid). A span
  matching an uncited pass-1 claim stays uncited. Final links always point to segments and
  slides; `notes.json` keeps the groups and how each citation resolved (`two_pass`, §19).
- Show peak memory (`mx.get_peak_memory()`) in the UI after each stage.

---

## 5. Stage details

### 5.1 Ingest
- Normalize: `ffmpeg -i in -ac 1 -ar 16000 -c:a pcm_s16le audio.wav`.
- Chunking happens in the next step (§5.1a), not here. Ingest uses no models.
- Video frames: `ffmpeg -vf "select='gt(scene,0.02)',showinfo" -fps_mode vfr frames/%03d.png` and
  parse each frame's `pts_time` from the `showinfo` output. Always keep the first frame. Add
  `scale='min(1600,iw)':-2` (for portrait sources, scale the long edge instead) so 4K screen
  frames aren't stored at full size.
- Dedupe: `imagehash.phash(hash_size=16)` (256 bits), dropping a frame if its Hamming distance to
  the last kept frame is ≤ 24. Expect 20–60 frames per hour of talk. Both values come from the
  4b-1 eval: a change between two white text slides scores only 0.04–0.07 on ffmpeg's scene
  metric (0.3 kept one slide per deck), and their 8×8 phashes were 4–8 bits apart (the old
  limit was 6); at 16×16 they are 38–60 apart.
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
- **Number guard** (pure Python, after parsing; `map.number_guard`, default on; Phase 4b-1b, D20): Tiny Aya changes digits even
  when the transcript is right ($42,500 → "$420,500", 336 → "3,360", "22nd" → "2nd"), and neither
  the prompt nor temperature 0 stops it (4b-1). Every number written as digits in a Map string
  (summary, key points, decisions, action tasks and due dates, open questions) is checked against
  the segment's **source numbers**: its transcript plus the text of its attached slides, both
  normalized to digits (`numbers.py`: number words, separators, `k`/`thousand`, ordinals).
  - In the sources: kept.
  - Not in the sources, and **exactly one** source number is one digit edit away (one digit
    inserted, dropped or changed, on the digit string without separators) or a factor of 10, 100
    or 1000 away: **repaired**. The digits are replaced in place, keeping the surrounding text
    and the output's separator style and decimal places ("$420,500" → "$42,500"; "420.500" →
    "42.500"). The segment's `number_repairs` lists each one (`{original, repaired}`), and the
    run log names it.
    Candidates (6b-2): the output number is read only the way the target language writes numbers
    (`numbers.DECIMAL_MARK`; in English "$770,000" is 770000, never 770.0; languages written
    both ways, and numbers no native reading fits, keep every reading), and an ordinal ("2nd")
    is only compared with the ordinals in the sources ("the 22nd", "the twenty-second"), not
    with counts and amounts.
    8g (D45): a plain number of one or two digits is never repaired, only kept or flagged
    (AMI IS1009b: "forty percent … and the rest" → Map's "60%" became the 40 said, one edit
    away; every other short repair was a day). With two or more candidates, those another
    number of the same string states (kept, or repaired to its one candidate) are dropped
    ("$34,000, up from $13,000" with 354,000 and 134,000 said → "$354,000, up from $134,000").
    A string that says one value twice is kept; a miscopied repeat of a value another number
    states would be repaired to the other candidate (not seen in the runs). A digit edit
    doesn't beat a factor: tried in 8g, it repaired pdf-slides' "6,900 candidates" (69 said,
    ×100) to the $8,900 cloud bill at seed 1, so xling-ja f9 ("34%" is a third candidate for
    "$34,000") stays flagged.
  - Otherwise (no candidate, or more than one): **kept and flagged**, never dropped and never
    guessed. The segment's `unverified_numbers` lists them (`MapFile` v3, §19) and the run log
    warns. An invented year such as `2024` in `2024-11-24` lands here.
  - A written number is read every way its separators allow ("42,500" = 42500 or, with a decimal
    comma, 42.5; thousands as `,` `.` NBSP, or a plain space only as "42 500"), and is kept if
    any reading is in the sources. With a scale suffix ("42.5k") or plain-space grouping
    ("42 500", which may also be two numbers) it is only kept or flagged, never repaired.
  - Digits glued to letters ("Q3", "R7B", "3D") are names, not numbers: not checked. Neither
    are numbers inside `terms` and numbers written as words. The sources are read the way
    the meeting's language writes numbers (6d-2, `numbers.source_numbers_in`): Spanish "11.800"
    and "42,500" are 11800 and 42500 (a separator before exactly three digits groups), "354 mil"
    is 354000; Japanese "35万4000" and "四万二千五百" are 354000 and 42500. English number
    words are read; other languages' number words aren't (Transcribe wrote digits in es/ja).
    A Map number is kept or repaired only by the notes language's readings (`native_readings`).
  - The guard does not resolve a speech/slide conflict (§18 `slide-conflict-en`): both values
    are in the sources, so both pass. Map ignores `map.md`'s slide rule, so (6b-2, user's
    choice) a pure-Python step **marks** the conflict instead of picking a side: a slide table
    row whose one value was never said, while the sentences naming its label hold exactly one
    number on no slide, gets the slide's value added after the spoken one in every Map string
    naming the label ("$80,000 (slide: $18,000)"). Recorded in `slide_conflicts` (MapFile v4).
    A speaker may be correcting an outdated slide, so neither value is replaced.
- **Invalid JSON:** same rule as §5.3. Retry once with "Return only valid JSON". If it still fails,
  store the raw text as `summary`, leave the lists empty, set `raw: true` on that segment's entry,
  and log a warning. The retry prompt (`prompts/map_retry.md`) also ends with the target language.
- Due dates are copied as spoken, never computed (Phase 0 saw "end of next week" → "10月27日";
  the 4b-1 eval saw `2024-11-24` for "November 24th"). `map.md` says to take owner and due only
  from the task's own sentence, never to write numeric dates or add a year, and to copy numbers
  digit for digit. One bad segment never fails the stage.
- **Dates and dues (8c, D41):** after the number guard (same `map.number_guard` switch), each
  segment's Map strings are checked against what the segment's transcript and slides say
  (`mapper.guard_dates`). English notes of an English meeting: a date ("November 1", "the 1st
  of November") that the segment doesn't say is repaired when exactly one said date of the same
  month has a day one digit edit away ("November 1" → "November 11"; recorded in
  `number_repairs`), otherwise kept and flagged in `unverified_numbers`. The number guard alone
  kept "November 1" because the segment said 1 elsewhere. Then a due the segment doesn't say is
  dropped (`dues.due_said`, D40's rule: English words close together, other languages digits
  only) and listed in `dropped_dues` (`MapFile` v5): Map copies its prompt's example due ("End
  of next week") onto tasks that have none, and Ask's documents showed it beside the real ones.
- **Owners (8d, D42):** last, on every segment (not behind `map.number_guard`), English notes
  only (`mapper.guard_owners`, `names.py`). A decision saying a person will do something
  ("Tom will update the pricing page by the 2nd of November") becomes that person's action item
  when the person is said in the segment (a `said_names` word, accents folded: "Víctor") and
  no Map task there states it (≥ 0.6); its due only if `dues.due_said` passes. A task with no
  owner gets one from its own text ("Laura will clean up …") or, in an English meeting, from
  the segment sentence it matches best, when that sentence gives it to exactly one person in a
  cue position ("Maria, can you …", "Grace will …", "Action item for Tom …") and names no
  one else; never from a neighbouring task. Weekdays, months and pronouns are no one; AMI's
  "Design will …" isn't a said name ("design" is said too). Changes are logged only (no
  MapFile field: a new field would move every Ask cache key, 8c).
- **Tasks Map never wrote (8e, D43):** after the owners, in an English meeting
  (`map.cue_tasks`, default on). Map drops some tasks at random: which ones changes with the
  seed (quick: Omar's hiring plan written at seed 0, dropped at 2 and 3). Python lists the
  segment's sentences that give one person a task in a cue position (`names.addressee`, as
  above) while no Map task there is that person's (`mapper.uncovered_cues`; owners compared
  in the meeting's spelling, a two-word owner also by its first word). Only those go to one
  extra Tiny Aya call per segment (`prompts/map_tasks.md`, temperature 0, no example lines),
  which says which sentences give a task and writes each in the notes language: the model
  tells "Tom, can you hear me?" from a task, which a verb list would only fit to the
  generator. Python keeps the owner (the sentence's person, never the model's), completes a
  task cut short from the sentence's own words up to its "by …" when the model's words are
  said there (`mapper.whole_task`, English notes of an English meeting: "order laptops" →
  "order laptops for the new starters"; rewording the prompt didn't stop it), and runs the
  added items through Map's own number, date and due checks. Each `{"line": …}` entry of the
  reply is read on its own (Tiny Aya closed one list with `}}}`). The run log names each
  added task and each cue sentence the model turned down. Replies ("Okay, will do."),
  one-letter words (ASR's "the F site") and a capitalised verb after a name (ASR's "Action
  item for Laura Schedule …") are no one. No MapFile field; the prompt moves Map's config
  hash. AMI addresses no one by name: no call, no change.
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
  Decide this in Phase 6 and document it in `NOTES.md`. **6d-2:** `ModelSpec.quantize_bits`
  converts a repo locally at `cohr models download` time; Fire was tried (D18) and not adopted,
  so `map_model_by_lang` is empty.
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
- **Over the cap:** two-pass reduce (§4, 6c). Transcript trimming (25% at a time, down to 300
  characters per segment, then a clear error) is left only for a two-pass group over the cap on
  its own (one very long segment). Map notes and slides are never trimmed.
- **Phase 0:** confirm the exact `apply_chat_template(..., documents=..., chat_template="rag", enable_citations=True)`
  arguments, and the citation format R7B produces (expected: `<co>span</co: 0:[i,j]>`).
- **Fixed section markers:** the prompt asks for five sections under fixed English markers, whatever
  the target language: `## SUMMARY`, `## KEY_POINTS`, `## DECISIONS`, `## ACTION_ITEMS`,
  `## OPEN_QUESTIONS`. The parser splits on these exact markers, so German or Japanese notes
  parse the same way as English ones. The prompt alone doesn't hold them: in de/ar notes R7B
  translated all five (6d-1), so Reduce starts R7B's reply with `## SUMMARY` (a prefill; Ask
  has none, D22), and R7B then writes the other four in English. `render.py` shows each heading in the target language from
  a static table (`LABELS[lang][key]`: the five headings plus "Sources", "owner", "due" and
  "uncited", for all 23 languages; English if the language is missing). Missing or unknown
  markers: log them, and put unmatched text in Summary rather than dropping it. A list item
  under an unknown heading goes to the section of the cited segments' Map item it matches best
  (≥ 0.6; 8h, D46), else Summary: AMI IS1009b's reply wrote segment 0's five sections, then
  "## <co>Remote Control Design</co: …>" over segments 1–6 as one list of Map's key points and
  decisions word for word. Prose under it stays Summary. A claim repeated in its section with
  the same citations is kept once (a looping reply wrote 14 lines twice, TS3003c three
  segments' decisions three times).
- **Claims:** one per list item (continuation lines join the item; `- none` is skipped) and one
  per sentence in Summary (`.!?` + space, or `。！？`). A `<co>` span is never split, so its
  citations stay with the claim that contains it. A claim's citations are its spans' targets,
  first occurrence of each. A segment's refined `t` comes from the span citing it that best
  matches one of its chunks (ties: the first; `reduce.best_span`, 8c): R7B often cites a
  segment through a name alone and then the task (`<co>Tom</co>` … `<co>finish the post
  mortem…</co>`), and the name's first chunk was 197.7 s early on long-en.
- **Action item line format:** one item per line, `- [owner: <name or ?>] [due: <date or ?>] <task>`
  (`?` = unknown). The parser fills `owner`/`due` (§19). If a line doesn't match, match the task
  text against `map.json` action items from the cited segments (token overlap ≥ 0.6) and copy their
  owner and due date; otherwise leave them null. Phase 0 got `Priya: screenshots by 20 October`
  without an example, so `reduce.md` shows two example lines.
- **Tasks filed as decisions (6e-2, D33):** a decision that matches a cited segment's `map.json`
  action item (≥ 0.6) better than any of its Map decisions, or as well when that action item
  names an owner, moves to Action items with its citations; when an action item already
  states it, its citations are added there instead.
- **Owners in the meeting's spelling (6e-2, D34):** an owner in a non-Latin script that the
  transcript doesn't contain as written is replaced by the name said in the meeting with the
  same consonants after transliteration (`names.py`, `anyascii`); kept as written otherwise.
- **Due guard and vague tasks (8b, D40):** on the final notes, after the owners. A due that its
  cited segments and slides don't say is dropped, never replaced (Map copies its prompt's
  example due onto tasks that have none). English notes of an English meeting: the due's words
  within a few words of each other (`dues.said`: "17 November" = "the 17th of November").
  Otherwise only its digits are read, in the meeting's language; a due with no digits is kept,
  and so is an uncited item's. Placeholder owners and dues ("Not specified", "TBD") are unknown.
  Then an English action item with no owner and no due that starts "Continue", "Ensure", "Keep",
  "Monitor" or "Maintain" moves to Key points with its citations (never deleted).
- **Map's tasks the notes lose (8d, D42):** before the owners are restored. An action item
  R7B wrote as `[owner: ?]` takes the owner of the cited segment's Map task it matches best
  (≥ 0.6; a tie between owners stays unknown); when another item already gives that owner the
  same task, its citations merge there (pass 2 wrote one task twice). Then a Map task with an owner said in its
  segment that no action item or decision states (≥ 0.6) is added, citing its segment, in
  segment order: R7B drops some although every prompt keeps every task (pass 2 lost a task
  pass 1 had; a one-pass reply lost one). Both place the segment citation with an
  `[owner: …] task` span, so the owner anchors it in another language (D39); the anchors
  compare names with accents folded. Owner-less Map tasks are never added (AMI writes decided
  designs as tasks).
- **Decisions the notes lose (8f, D44):** after Map's lost tasks, in an English meeting. A
  sentence stating a decision in a cue form ("We decided to …", "We agreed to …", "the decision
  is to …"; not a question, a denial, "whether", or an earlier meeting's) that no decision or
  action item states (≥ 0.6) becomes a decision citing its segment, in segment order. English
  notes: a key point that states only that decision moves to Decisions (R7B filed one there);
  otherwise the sentence's own words are the decision ("Keep the onboarding checklist."), and a
  bundled key point stays. Other notes languages can't be compared with the sentence: a segment
  with more cues than decisions citing it gets its Map decisions the notes don't state, up to
  the difference, skipping Map's tasks and open questions; when no decision cited the segment,
  each is placed at its cue sentence (Map lists them in the order said). Without "to" a cue
  refers back or sums up (AMI: "I think we all agreed on that."); real meetings decide in other
  words too, and those stay R7B's and Map's.
- **Figures the notes lose (8g, D45):** after the decisions, in any notes language. A Map key
  point with a figure (digits, 10 or more, read the way the notes language writes numbers,
  script digits folded per D38) that no claim of the notes states goes to Key points citing its
  segment, in segment order; a notes key point citing that segment whose figures are some of
  the Map key point's is replaced by it, citations kept (R7B kept "$80,000" and dropped
  "(slide: $18,000)"). Either way the segment's slides that show a figure it adds are cited
  too (the judge read the slide value against seg 0 alone). A Map key point holding a number
  the guard flagged is skipped, so the step adds no unverified number. Reduce lost 6 of the eval's numbers this way (pass 1 and pass
  2 of long-en, one-pass ar/hi). Numbers Map never wrote stay missed (8 in 8g's kickoff): a
  sentence net fits evals/gen's "The X is Y. That's compared with Z." and copies AMI's
  fragments.
- **Segments a section skipped (8h, D47):** one-pass notes only, after the decisions. A segment
  that no decision cites gets its Map decisions the notes don't state (≥ 0.6; not Map's tasks
  or open questions of the segment), and one that no open question cites gets its Map open
  questions the notes don't ask, each citing its segment, in segment order; a key point citing
  the segment that states only the decision moves to Decisions instead (D44's move). Items
  with no letter or digit (Map's "[]") are skipped. R7B's one-pass replies over AMI's 7–9
  segments leave whole segments out of a section (IS1009b seg 1's five decisions, TS3003c,
  ES2004b's key points); the eval's 2-segment items never did. Two-pass notes are condensed on
  purpose and long-en's Map wrote a planted open question ("sponsor the local hackathon") as a
  decision, so they're left as written. Key points aren't filled: Summary covers AMI's abstract,
  and the fills read as padding ("The meeting was stuck due to an issue with opening a file").
- **Cut replies (6e-2, D32):** a reply that stopped at `reduce.max_tokens` without
  `<|END_RESPONSE|>` (mlx-lm's finish reason) is counted in `NotesDoc.truncated`, logged and
  shown on the Notes page. It is never retried (temperature 0 would write the same start) or
  completed.
- Citation parsing: regex `<co>(.*?)</co:\s*([^>]+)>` → list of document indices → `citations[]`.
- **Citation rules (every claim cites):**
  - A document index outside the documents sent is dropped and logged. It never becomes a link.
  - A claim left with no valid citation is kept, with `citations: []`. The Notes page shows it
    grayed out with an "uncited" badge (gray, §7), and the eval counts it (§18). Never invent a
    citation.
- **Citation time refinement** (pure Python, after parsing): a segment citation points at a window
  of up to 5 minutes, which is too coarse to play from. For each segment citation, score the cited
  span against every transcript chunk in that segment (the token overlap from §5.4) and store the
  best chunk; then place `t` inside it at the start of its best-matching sentence, by character
  position (assumes an even speaking rate). If no chunk scores ≥ 0.1 (D19), which is what
  happens when the notes and the meeting are in different languages (Hindi notes of an English
  meeting, English notes of a Japanese one), the span's **anchors** pick the chunk (8a, D39):
  its numbers, read in the notes' language with other scripts' digits folded and in each chunk
  in the meeting's language (`numbers.source_numbers_in`: "35万4000" = 354000), and an action
  item's `[owner: …]`, in the meeting's spelling for the segment (D34) and said as a whole word
  or, in Japanese, as katakana with the same consonants (トム = Tom). The chunk with the most
  anchors wins (ties: word overlap, then earliest), and `t` goes to the start of its sentence
  with the most anchors. Names elsewhere in a claim aren't anchors (Arabic تم "was done" has
  Tom's consonants). No anchor in any chunk: the segment's `t0`, never a guess. With the
  chunk's `t0` alone, `t` was a median 12.5 s early (chunks run up to 30 s) and only 61% of
  citations landed within ±15 s; with the sentence offset, 98% (D19, 4b-1).
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
  **Built in 6d-1 (D30):** the 44 languages of Tiny Aya's card beyond R7B's 23
  (`config.BETA_LANGUAGES`, listed after R7B's on the New run page as "Swahili (beta)"). For
  these, Map and Reduce write English (`Config.notes_language`), and a **translate** stage
  (between reduce and render, Tiny Aya, loaded after R7B) writes `notes_translated.json`
  (`TranslatedNotes`): each section's claims go out as numbered lines (`<3> text`), the number is
  the only marker, and each translated line goes back on its own claim, so citations never pass
  through the model. A line must keep every number's value (`translate.same_numbers`; Tiny Aya
  wrote "$4,500" for "$42,500"); a bad reply is retried one claim at a time, then the claim stays
  English (`untranslated`). Headings stay English (no LABELS entry); Ask falls back to English.
  Render, the Notes page and the eval show the translated notes when present.

### 5.7 Ask
- Same documents as Reduce, the user's question, and the `rag` template with citations. Answer
  rendered with the same citation links, using the same citation rules and time refinement as
  §5.6. No re-processing needed.
- **Follow-ups:** the last 3 question/answer turns are sent as history (answers with
  citation markup removed, to save tokens), so "and who owns that?" works. History counts toward
  the ~32K cap and is dropped oldest-first if needed.
- **Prompt shape (D22, Phase 5b):** one user turn through Reduce's own `prompt()`: the rules,
  the history (`prompts/ask_history.md`, `ask_turn.md`), the question, and "Answer in
  <language>." as the **last line** (as in Map, §5.5). R7B's native multi-turn `rag` format puts
  the documents after the first turn only, and R7B then called the document tool again on every
  follow-up instead of answering. Reduce's prompt, code and config hash are unchanged (tested).
- **Answer language (D23):** the language of the question, detected with `lingua-language-detector`
  (a runtime dependency) over R7B's 23 languages. Sure means: at least 2 words (or any CJK
  text), top confidence ≥ 0.2, and a lead of ≥ 0.05 over the runner-up. Otherwise, use the run's
  target language. The documents stay as they are; R7B answers across languages.
- **Known limits (5b manual check; 6b measured near-miss, §18):** a question in another language than the history can get
  "no information" or a repeat of an earlier answer; a near-miss question ("marketing budget
  next year") can get a related cited fact ("budget for next quarter") instead of "not
  stated". An Ask eval on the planted facts would measure both (Phase 6).
- **Map's values (8c, D41):** the segment documents carry map.json as Map's guards left it, so
  Ask sees repaired dates and no copied dues (a-f82's "November 1" was "11 November"). A task
  Map missed is only in the transcript; under the 12-document cap its segment may not be sent
  (long-en a-f15, ASR "Oma" for Omar).
- **Long meetings (6c, D17, D27):** if the documents don't fit even without history, rank them
  for the question plus the last turn (a follow-up has few terms of its own) with an in-memory
  SQLite FTS5 table (BM25, terms OR'ed, stdlib; `trigram` windows for zh/ja/ko text), add the
  best ones that fit with the history, at most 12 (filling the 32K cap took ~100 s per answer
  on long-en and R7B cited document after document), and send them in meeting order, so citation indices map
  as before. This replaces transcript trimming in Ask. The Ask page then says "Answer based on
  the N most relevant segments (and M slides)". The ranking is pure Python and unit tested.

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
  judge.py           eval-only Qwen3 judge: support / coverage questions, JSON retry, answer
                     cache (§18, Phase 4b-2)
  render.py          notes.md rendering with anchors
  pipeline.py        stage runner: manifest, hashing, skip logic, progress callbacks, cancel
                     flag, interrupted-stage recovery, atomic writes, run log
  jobs.py            UI job layer, no Streamlit: pipeline in a background thread, progress
                     queue, cancel, one job at a time, recover_all on app start (Phase 5a)
  ui.py              Streamlit glue shared by app.py and app_pages/: cached ModelManager and
                     JobRunner, sidebar, job panel, per-segment audio clips (Phase 5a)
  prompts/           vision.md  map.md  reduce.md  reduce_pass2.md  ask.md  ask_history.md
                     ask_turn.md  translate.md  judge.md
                     judge_coverage.md  judge_retry.md
                     (plain files; each hash feeds its stage's config hash, §3)
  cli.py             `cohr run <audio> [--slides …] [--video …] --lang en --target en`, `cohr eval`
  numbers.py         number normalization (words, separators, ordinals → digits) shared by
                     the Map number guard and the evals (Phase 4b-1b)
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
  search.py                  layout="wide"      (Phase 7, §13)
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
    lines on very wide screens. (5a: it shrinks, D13.)
  - Right, **sources**: `st.container(height=…, border=True)`, scrolling independently, one
    block per segment with its anchor (`st.subheader(anchor="seg-N")`, slides `slide-N`), time
    range, an `st.audio` player and slide images (`width="stretch"`, `alt=` slide title).
    Unaligned slides come after. The player gets that segment's clip of `audio.wav` (cached WAV
    bytes, `start_time` = earliest cited moment − t0): `st.audio` with a path reads and hashes
    the whole file on every call (5a).
  - At ≤ 640 px the columns stack: notes first, then sources. Citation anchors still jump to the
    right block, because they're anchors, not layout.
  - **Phase 5 check:** in-page anchors may scroll only the page, not the Sources panel's own
    scroll area. If a citation click doesn't bring its block into view, use the Ask pattern
    instead: citations become small buttons that open the right-side drawer with that segment.
    (5a: anchors scroll the panel, D14.)
  - **Medium windows (5a finding):** at ~800 px with the sidebar open, the columns get
    ~209 / 128 px because they only stack at ≤ 640 px viewport width. Closing the sidebar gives
    422 / 271 px. **Fix (D21, Phase 5b):** a "Sources" `st.segmented_control` in the header row,
    **Beside** (default, the two columns above) or **Below** (notes in the same 880 px
    container, then the Sources blocks under them without the fixed-height scroll box, so
    anchors scroll the page). The choice lives in `st.session_state` for the session. Python
    can't see the width, so the user picks; nothing is measured.

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
fileWatcherType = "none"                 # its sys.modules walk imports transformers' lazy torch modules (5a)

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
  non-English output (German fixture) land in the right sections, and unknown markers go to Summary
  (list items under one go to their Map item's section, D46; a repeated claim is kept once);
  out-of-range document indices are dropped; uncited claims are kept with `citations: []`;
  citation time refinement picks the best-matching chunk and falls back to segment `t0` below
  the threshold; action item lines parse owner/due, with the `map.json` fallback; the portable
  `notes.md` has no `#seg-` links and has a Sources entry for every cited segment and slide;
  Map invalid-JSON fallback; Ask history trimming under the cap.
- **Unit tests must not need Metal or model weights:** they run on CI runners without a usable
  GPU (§15). Import `mlx`, `mlx_lm`, `mlx_vlm`, `mlx_audio` and `mlx_speech` lazily inside the model wrappers
  (`vad.py`, `asr.py`, `vision.py`, `mapper.py`, `reduce.py`, `judge.py`, `models.py`), never at module top level in code
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
- **4b-1: evals without the judge** (done 2026-10-11, NOTES.md). Planted-facts generator (`evals/gen/`, seeded: script + `truth.json`,
  `say` voices, PIL slides, ffmpeg noise/tones/silence) for the four `quick` variants; exact metrics (fact
  recall, citation accuracy, uncited claims, made-up items, VAD gap text, WER, slides, output language,
  cost); cross-language exact-field scoring (code + unit tests; the cross-language variants run in `full`,
  Phase 6); `cohr eval --set quick` with reports and the baseline comparison (no committed baseline until
  Phase 6); `tests/test_evals.py`. First `quick` numbers recorded; D11 re-check, D12 (a second `quick`
  run with another window), D19. Fix the 4a Map error (owner's due date given to another action item,
  `prompts/map.md`) with the made-up-items count before and after.
- **4b-1b: Map number guard** (done 2026-10-11, NOTES.md; before 4b-2, so the judge's calibration pairs come from cleaner
  output). Move `digits`/`numbers_in` from `evals.py` into `numbers.py` (no behaviour change; the
  eval tests stay green). Guard in `mapper.py` per §5.5, `MapFile` v3 (`unverified_numbers` per
  segment, snapshot regenerated). Unit tests from the 4b-1 cases: 42500/420500, 57000/570000,
  336/3360, 22/2 repaired; a value two edits away flagged; two equally close candidates flagged;
  separator and `$`/`%` style kept; numbers in the sources untouched; ISO year flagged; a JSON
  fallback (`raw: true`) segment is not touched. Eval: per item, `map_numbers_repaired` and
  `map_numbers_unverified` (from `map.json`), and `unsupported_numbers` (numbers in `notes.json`
  claims found in no transcript or slide, to see whether Reduce also changes digits). `quick`
  before/after on the same seeds: number recall, fact recall, made-up items; no other metric
  worse (D20). Record a wrong repair if the eval shows one.
- **4b-2: judge** (done 2026-10-11, NOTES.md). Qwen3 judge wrapper (§18), support rate, judge recall for cross-language text facts,
  `evals/gen/judge_calibration.json` (30 pairs: Claude drafts labels from 4b-1b outputs, the user reviews
  them before the judge runs on them), D15. `cohr eval --calibrate-judge`, `--no-judge`; answers cached
  per item in `evals/data/judge/`. Kickoff prompt in §21.
**Phase 5: Streamlit UI**, split in two PRs:
- **5a: UI without Ask** (done 2026-10-11, NOTES.md). "Paper & Ink" theme (`.streamlit/config.toml` exactly as §7) with
  `test_theme.py`; a Streamlit-free job layer (`src/cohr/jobs.py`: pipeline in a background thread,
  progress queue, cancel, one job at a time, `recover_interrupted` on app start) with unit tests;
  `pipeline.run(force=…)` so Runs/Notes can re-run a stage on demand ("Re-reduce"); `app.py`
  (navigation, sidebar run picker, model/memory status with the GPU limit warning, §11 footer);
  pages New run (upload or local path, §12 limits, Advanced options, per-stage `st.status`,
  Cancel), Notes (in-app render, Sources panel with audio and slides, download `notes.md`,
  Re-reduce, uncited caption) and Runs (table, Resume, re-run stage, Prune/Delete with
  confirmation, peak-memory chart); `test_app.py` (AppTest, fake ModelManager, fixture run);
  the window-width and light/dark checks; D13, D14. Kickoff prompt in §21.
- **5b: Ask** (done 2026-10-11, NOTES.md). `src/cohr/ask.py` + `prompts/ask.md` (§5.7: same `rag` documents as Reduce, the
  last 3 turns as history with citation markup removed, dropped oldest-first under the ~32K cap,
  answer language from `lingua` with the run's target language as fallback, same citation
  parsing and time refinement); the Ask page (chat, source pills, right-side drawer) and Notes'
  "Open in Ask" button; tests for history trimming, language fallback and citations in answers.
  Notes' Sources Beside/Below toggle for medium windows (D21, §7).
  Long-meeting ranking (FTS5) stays in Phase 6. Kickoff prompt in §21.
**Phase 6: hardening**, split in five PRs. Measurement comes first, so every later change is
compared with a committed baseline:
- **6a: baseline + Ask eval** (done 2026-10-11, NOTES.md). Measurement only: no prompt or pipeline behavior changes. The
  Ask eval (§18 Ask row) on `quick` uses template questions from `truth.json`: number and
  action questions, a follow-up, German questions alone and after English history, and one
  near-miss per item. Answers are cached like the judge's. D16 comes from 3 `quick` runs on the
  same media with pipeline seeds 1–3 (`-c seed=N`; `--seed` changes the generated meetings).
  Then commit `evals/baseline.json`, with `compare()` using the D16 thresholds. Kickoff prompt
  in §21.
- **6b: quality fixes** (done 2026-10-11, NOTES.md), each checked against the baseline, which is updated with the reason:
  Reduce turning filler into open questions (4b-2), Reduce dropping facts Map has right
  (4b-1b: 4650, $57,000), and Ask's near-miss and cross-language limits (§5.7, 5b). 6a
  measured near-miss at 0 of 4 on every seed and found Ask using the misspoken value on
  `slide-conflict-en`. Measurement goes first: planted open questions in the generator and
  three near-miss questions per item, re-baselined. Kickoff prompt in §21.
- **6c: long meetings** (done 2026-10-12, NOTES.md). First, from 6b: Reduce bundling several facts into one claim (one
  citation for 3–4 facts) and the citation-accuracy gate (noise 7.1 points on the 6b media, gate
  3; D25), since citation accuracy is how two-pass is judged. Then inputs over 2 h: two-pass
  reduce with marker-based citations (§4, D26), checked first on `quick` with a lowered cap, then
  on the `long-en` item (own `long` set); Ask ranking with FTS5 (§5.7, D17, D27) and "Answer
  based on the N most relevant segments". Kickoff prompt in §21.
  Results: §20 D17, D25–D27; `evals/baseline-long.json` (report-only). Carry-overs: Map's
  bundled key points (Reduce copies them; key-point citations land on the bundle, 6b-2); pass 2
  of two-pass dropping about a third of pass 1's decisions on `long-en`; near-miss under the cap
  (4/12 vs 7/12); `long-en` German-after-history (0/1).
- **6b-2: Map fixes** (done 2026-10-13, NOTES.md; after 6c, from 6b and 6c): measurement first (each planted fact's path
  transcript → Map → notes, and whether Map replies hit `map.max_tokens`); then Map's bundled
  key points (6c: Reduce copies them, so key-point citations land on the bundle), Map dropping
  facts on long, dense segments (~280 s, 7 topics: ~7 key points kept), the due-date misread
  ("22nd" → "2nd"), and the slide-conflict value (Map repeats the misspoken number although the
  slide is in its input). Then `long-en` (Map changes change it): whether two-pass pass 2 still
  drops about a third of pass 1's decisions, re-baselined report-only. Map prompt changes move
  every metric: one re-baseline per kept change. Not in 6b-2: Ask near-miss under the cap
  (4/12 vs 7/12) and `long-en` German-after-history (one question), left for 6e's long-meeting
  check. Kickoff prompt in §21.
  Results: one fact per Map key point (citation 100% on seeds 1–3); dense-segment drops not
  fixed (Tiny Aya coverage; D12 re-checked, 300 s kept); number guard reads separators by
  target language and repairs ordinals only to days said (D20); speech/slide conflicts shown
  with both values (D28, MapFile v4); `long-en` re-baselined (recall 70.0%, decisions 75%, Ask
  94.4%; pass 2 keeps 15 of 18 decisions). Carry-overs (6e): Map files long-en tasks as
  decisions (actions 40%; a Reduce rule was ignored); `reduce.md`'s "Update the release notes"
  example overlaps a generator task; Ask doesn't see slide conflicts.
- **6d: languages**, split in two (user, 2026-10-13): the output side first, then the input
  side and the Map models.
  - **6d-1: notes in other languages.** The `en → {de,ar}` items in a new `lang` set
    (report-only baseline `evals/baseline-lang.json`), measured first; scoring that reads numbers
    the way the notes language writes them; the " (slide: …)" note (6b-2) in the target
    language; Arabic (right-to-left) in the Notes page and `notes.md`; "Other (beta)" target
    languages with translation that keeps citations (§5.6). Kickoff prompt in §21.
    Results: R7B translated the five section markers in de/ar (everything fell into Summary,
    recall 38.5%): Reduce now starts the reply with `## SUMMARY`; the eval reads numbers the
    notes language's way; the "(slide: …)" note is localized and found in German notes; Arabic
    lines are RTL isolates (D29); "Other (beta)" is its own translate stage (D30). `lang` seed 1:
    recall 84.6% (conflict-de 76.9, clean-ar 84.6, clean-sw 92.3), citation 100%, made-up 0,
    language 3/3, slide_value_used true; baseline `evals/baseline-lang.json` (report-only).
    `quick` unchanged (86.8%). Carry-overs: ar owners in Arabic script (eval misses them);
    translation keeps ~4 money claims English and once invented the summary.
  - **6d-2: other source languages and regional Map models.** The `xling-{es,ja}` items (the
    generator in Spanish and Japanese: templates, number words, `say` voices; CER for ja);
    optional regional Map models (§5.5, D18) scored on 6d-1's `en → X` items:
    `map_model_by_lang` filled only for languages that pass D18 (may stay empty). Gated model
    downloads and the HF token need the user's approval. Kickoff prompt in §21 (written after
    6d-1, from its findings).
    Done 2026-10-13 (NOTES.md). Results: the generator speaks Spanish and Japanese (per-language
    tables; English scripts unchanged; Helvetica/Hiragino on the slides). Transcribe writes
    digits (es "11.800", "354 mil"; ja "35万4000", once "四万二千五百"); the guard and the eval
    read the sources in the meeting's language and keep only the notes language's readings
    (D20): `xling-es` recall 69.2 → 92.3%, `xling-ja` 76.9% (its misses are an ASR drop and
    Map). D18: Fire not adopted (hi −2.6, bn +2.5 vs a 5.8 bar); `map_model_by_lang` empty.
    Found: `pipeline.run` didn't switch a shared ModelManager to the run's models (eval
    model overrides were ignored; fixed). `lang` seed 1 (7 items): recall 81.3%, citation
    96.0%, language 7/7; baseline rewritten. `quick` unchanged (86.8%).
    Carry-overs (6e): Transcribe drops the first sentence of 3 of 18 Japanese chunks (cut
    0.4 s earlier, it's kept: chunk padding or overlap); R7B writes owners in Devanagari (hi)
    and Arabic script (ar), which the eval's owner match misses (clean-hi made-up 5), and once
    wrote Tuesday for Monday in Hindi; Map drops facts in dense Japanese segments and misreads
    万 forms ("35万4000" → "$34,000", now flagged); no slide-conflict check for Japanese.
- **6e: full set + v1**, split in two (user, 2026-10-13): `quick` already meets the v1
  targets, so the carry-overs don't block v1; the first `full` run ranks them.
  - **6e-1: full set + robustness, measured.** FLEURS and AMI in `full` (§18) with the CC BY
    footer; the speed check on real meetings (AMI, §9 definition of done); error messages for
    missing ffmpeg or models (CLI and UI); an end-to-end check of the sequential memory policy;
    README brought up to date (status, app, `cohr models download`, model cache size and
    `hf cache rm`, not `prune`). The first `full` run is a measurement: no fixes,
    `baseline-full.json` written. Kickoff prompt in §21.
    Done 2026-10-14 (NOTES.md). Results: `full` = quick + lang (`include`, `gate: [quick]`,
    D31) + FLEURS (streamed, 205 MB kept) + 5 AMI meetings (browser download: the mirror
    returned 403 to scripted requests). Gate items meet every v1 target (recall 89.7%,
    citations 98.4%, 14.7 min/audio h with Vision; AMI 5.5 min/h, peak 17.3 GB). FLEURS WER
    6.6%, CER 7.9% (ar 16.1, zh 12.5). AMI coverage 44.9% (decisions 34.8, actions 18.2),
    support 80.2%. Found: Reduce's 2048-token cap cuts 4 of 5 AMI notes silently (4096:
    52.8%); interrupted model downloads passed as cached (fixed); Ask eval failed on es/ja
    meetings (now skipped). `sequential` = `resident` byte for byte, peak 5.9 vs 16.8 GB.
    Carry-overs ranked in NOTES.md (6e-1 "Carry-overs ranked").
  - **6e-2: carry-over fixes + v1.** The carry-overs from 6b-2, 6c, 6d-1 and 6d-2 that the
    `full` run ranks highest, then the `full` release gate and the bump to `1.0.0` (§16, only
    when asked). Kickoff prompt in §21, from the 6e-1 ranking (carry-overs 1–4; 5–7 after
    v1 unless step 0 argues otherwise).
    Done 2026-10-14 (NOTES.md), the 1.0.0 bump still to ask. Results: Reduce writes up to
    8192 tokens and flags a cut reply (D32; AMI coverage 44.9 → 52.8%, decisions 52.2,
    actions 36.4%, 0 cut); decisions that are Map's tasks move to Action items (D33; AMI
    ES2004b 15 → 9 decisions); owners back in the meeting's spelling (D34; clean-hi recall
    54 → 92%); numbers in no source counted on AMI (0); 万 normalization tried, not kept
    (D35); `cohr eval --rerun STAGE`. Release gate: `full` **pass** (recall 88.7%, made-up
    items 1, action recall 94.5%, Ask 98.3%); `long` report-only on regenerated long-en
    audio (recall 67.5%, made-up items 12, Ask 77.8%). Post-v1 list in NOTES.md.
Kickoff prompts for 6b–6e are written after the previous sub-phase, using its findings.
**Phase 7 (after v1): cross-meeting search:** SQLite + FTS5 index and Search page per §13.
Kickoff prompt in §21.
Done 2026-10-27 (NOTES.md), the 1.1.0 bump still to ask. `src/cohr/index.py` (synced on
read, D36), `src/cohr/fts.py` (Ask's term code, shared), `cohr index rebuild`, `cohr search`,
the Search page (results with a source drawer and Open notes at the moment; action items of
every meeting by owner). Two-table tokenizing (D37), transcript hits at the matched sentence,
digits folded across scripts (D38). Search check on the 12 planted-fact eval items
(`scripts/search_check.py`, report-only): within ±15 s in the fact's top 5, numbers
119/126 transcript · 100/126 notes, owners 67/75 · 53/75, fact words 252/288 · 109/288.
**Phase 8a (after v1): cross-language citation times** (post-v1 item 2): a claim whose words
share no token with the transcript (Hindi notes of an English meeting, English notes of a
Japanese one) cites its segment's start, not its moment. lang baseline: citation accuracy
clean-hi 58.3%, xling-ja 70%, every other item 100%. Kickoff prompt in §21.
Done 2026-10-28 (NOTES.md). Anchors in `reduce.refine_time` when word overlap fails (D39):
numbers and the action item's owner; Reduce and Ask pass the notes' and the meeting's
languages. lang seed 1: citation accuracy 89.9 → 98.7% (clean-hi 58.3 → 100%, xling-ja 70 →
90%), recall and made-up items unchanged; quick identical (no citation fell back there).
**Phase 8b (after v1): made-up action items on long meetings** (post-v1 items 1 and 6): long-en
lists 12 made-up decisions and action items (`baseline-long.json`), 11 of them action items. 9 of
those 11 have no owner, and 6 of the 9 carry Map's due "end of next week", which their cited
segment doesn't say (AMI: 18 owner-less dues, none said in a cited segment). Kickoff prompt in §21.
Done 2026-10-28 (NOTES.md). On the final notes (D40): `reduce.guard_dues` drops a due its
cited text doesn't say (`dues.py`, shared with the evals), and `reduce.move_vague_tasks` moves
owner-less, due-less "Continue …"/"Ensure …" tasks to Key points. long-en made-up items 12 → 7
(recall 67.5%, unchanged; support 75.7 → 79.4%); AMI coverage unchanged, support up in 4 of 5;
quick and lang seed 1 identical.
**Phase 8c (after v1): Ask on long meetings** (post-v1 item 7): long-en Ask accuracy 77.8%
(`baseline-long.json`; 94.4% on the audio before 6e-2), the same 4 misses before and after 8b.
Ranking isn't the cause: the answer's segment ranks in the top 3 for all four. Three trace to
Map's values in Ask's documents (a copied due, a misread date, a missed task), one to a
one-word cited span placing the citation. Kickoff prompt in §21.
Done 2026-10-28 (NOTES.md, D41). A segment cited through several spans is placed by the best
match (`reduce.best_span`); Map's dates are repaired or flagged and its unsaid dues dropped on
map.json (`mapper.guard_dates`, MapFile v5). long-en Ask 77.8 → 88.9% (citation 93.3 → 100%),
made-up items 7 → 4, recall unchanged; quick citation accuracy 98.4 → 100%; lang identical
but one judged claim.
**Phase 8d (after v1): missed action items** (post-v1 item 3, widened): the latest quick, lang
and long reports miss 17 planted action items. Map wrote 2 only as a decision (item 3: long-en
"Tom will update the pricing page", xling-es "Victor will write the notes"), 3 without their
owner (clean-sw, clean-bn, conflict-de), and 5 as tasks the final notes lost or changed; 7 it
never wrote. Kickoff prompt in §21.
Done 2026-10-29 (NOTES.md, D42). Map: a person's "X will …" decision becomes their task and an
owner-less task takes the one person its own text or sentence gives it to
(`mapper.guard_owners`). Reduce: `[owner: ?]` items take their Map task's owner
(`reduce.fill_owners`) and owned Map tasks the notes lost are added (`reduce.add_lost_tasks`),
cited with the owner as anchor. 9 of the 17 found; lang seed 1 action recall 80.0 → 94.3%,
long-en 55 → 70% (made-up 4 → 6: two owner-less pass-1 tasks pass 2 now keeps); quick identical; AMI replay unchanged.
**Phase 8e (after v1): action items Map never wrote** (8d's "Open"): 8 planted tasks are in no
map.json (long-en f14, f15, f19, f71, f86; noisy-en f7; conflict-de f5, f6 with no owner).
Segments aren't uneven (long-en: 261–299 s, 726–864 words each; seg 4 wrote 0 of its 3 tasks,
seg 6 all 3). Each is a sentence `names.addressee` already reads ("Maria, can you …", "Kenji
will …", "Action item for Grace …"); a scan of those against each segment's Map owners reaches
7 of the 8. AMI addresses no one by name, so it can't test a rule like that. Kickoff prompt in §21.
Done 2026-10-29 (NOTES.md, D43). The sentences Map's owners don't cover go to one Tiny Aya call
per segment (`mapper.uncovered_cues`, `prompts/map_tasks.md`); Python sets the owner and
checks the values. Map's misses change with the seed; quick action recall 100% at seeds 0, 2
and 3 (was 95/80/75%); lang seed 1 94.3 → 97.1%; long-en 70 → 90%, Ask 88.9 → 94.4%, made-up
6 → 7 (f15's ASR name). conflict-de f6 open: the model turns Grace's sentence down in German.
**Phase 8f (after v1): missed decisions**: decisions are now the weakest kind in every set
(quick 75% at seed 0, 81/81/88% at seeds 1–3; long-en 70%; lang 21/28 judge-covered), and no
phase has looked at them. Of quick seed 0's 4 misses and long-en's 6, 7 are in no map.json,
1 is a Map key point and 1 a Map decision that the notes lost; long-en f55 is ASR's "PostGIS"
for Postgres. noisy-en f3 and slide-conflict-en f2 are missed at every seed 0–3. Kickoff
prompt in §21.
Done 2026-10-29 (NOTES.md, D44). 6 of the 18 misses were Reduce's (pass 1 dropped long-en
f49; R7B filed clean-sw/bn's as a key point; clean-ar's notes kept no seg-1 decision). One
Python step after Reduce (`reduce.add_said_decisions`): cue sentences no decision states
become decisions (English notes), or a segment's Map decisions up to its cue count (other
notes languages). quick decisions 100% at seeds 0–3 (was 75/81/81/88%); long-en 70 → 95%,
fact recall 78.8 → 85.0%; lang judge-covered 21 → 24/28; made-up items unchanged; AMI 0 changes.
**Phase 8g (after v1): missed numbers** (post-v1 item 4, widened): numbers are long-en's weakest
kind (31/40, 77.5%), and the latest reports miss 17 number facts in all. Map never wrote 8,
Reduce lost 6 that are in Map's key points, and xling-ja f9 is the one wrong digit left: Map's
"$34,000" is one edit from both 354,000 and 134,000, so the guard flags it instead of
repairing. 8c already fixed "1 November". Kickoff prompt in §21.
Done 2026-10-30 (NOTES.md, D45). The guard never repairs a one- or two-digit number (AMI
IS1009b's "60%", the only wrong repair of 41) and drops a candidate the same claim states;
f9 stays flagged ("34%" is a third candidate; a digit-edit tiebreak repaired a wrong value at
seed 1 and was dropped). After Reduce (`reduce.add_lost_numbers`), a Map key point with a
figure the notes lost is added or replaces the claim that kept part of it. Numbers: long-en
77.5 → 85.0%, quick 93.8/93.8/93.8/96.9 → 96.9/93.8/96.9/100% at seeds 0–3, lang 91.1 →
94.6%; made-up items, Ask and citation accuracy unchanged. Map's 8 never-written numbers left.
**Phase 8h (after v1): AMI coverage** (8g's "Open"): 8g's run, AMI's first Map run since Oct
14, fell 52.8 → 44.9% (decisions 52.2 → 30.4%). Most of it is one reply: on IS1009b R7B wrote
all five sections for segment 0 only, then segments 1–6 as one flat list under
"## <co>Remote Control Design</co: …>", 14 of its lines twice, and §5.6's fallback filed it
all as Summary (34 summary claims; decisions 75 → 0%). Reduce's prompt is unchanged since
Oct 13; its input lost 44 tokens (8c's dropped dues, now `[due: ?]`). The other four meetings
kept the shape and moved by one or two claims each. Kickoff prompt in §21.
Done 2026-10-30 (NOTES.md, D46, D47). Step 0 scored both sides from the judge cache: 3 of the
7 lost sentences were judge noise or reply drift, and the shared cause was R7B's one-pass
replies over 7–9 segments leaving whole segments out of a section (IS1009b, TS3003c's loop,
ES2004b), not the dump alone. The parser files list items under an unknown heading by their
Map item and keeps a repeated claim once (D46); after Reduce, a segment no decision or open
question cites gets its Map ones (one-pass notes, D47). AMI coverage 44.9 → 50.6%,
decisions 30.4 → 47.8%, problems 16.7 → 33.3%, support 80.3 → 80.1%; quick (seeds 0–3), lang
and long-en notes identical.

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

**Location:** `<runs_dir>/index.db` (gitignored with `runs/`). Plain SQL with
`meta.schema_version` (`index.SCHEMA_VERSION`), not a §19 run file: it isn't a run file, and it
reads the run files only through `schemas.load`. Only `runs_dir()` is indexed, so eval runs
under `evals/data/runs/` never are.

**What's indexed (as built, Phase 7)**
| kind | from | one row per | a hit opens at |
|---|---|---|---|
| `transcript` | `transcript.json` items, segment via `segments.chunk_ids` | chunk (≤ 30 s) | the chunk's sentence best matching the query (D19's character-position offset), not the chunk start |
| `slide` | `slides.json`: title, `text_md`, `tables_md`, `chart_notes` (not Vision's `visual_notes`) | slide | its aligned `t` / `#slide-N` |
| `note` | `render.shown_notes` (an "Other (beta)" run is searched in the language people read), citations stored as JSON | claim | its first seg citation's refined `t`, else its first slide |

Action items also go in `action_items` (owner, due, task, citations); a note row of an action
item holds "task · owner · due", so a name finds that person's tasks. A run with no notes yet
is indexed by the files it has; an unreadable file is skipped and the rest indexed; a run whose
manifest can't be read is left out and retried on the next sync.

**Schema (as built)**
```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);   -- schema_version
CREATE TABLE runs (run_id TEXT PRIMARY KEY, key TEXT NOT NULL, created_at TEXT NOT NULL,
  source_lang TEXT, notes_lang TEXT, status TEXT NOT NULL,        -- done | partial
  pruned INTEGER NOT NULL);
CREATE TABLE items (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
  lang TEXT, section TEXT, seg_id INTEGER, slide_id INTEGER, t REAL, t1 REAL,
  citations TEXT, text TEXT NOT NULL);                            -- digits folded (D38)
CREATE TABLE action_items (run_id TEXT NOT NULL, owner TEXT, due TEXT, text TEXT NOT NULL,
  lang TEXT, seg_id INTEGER, slide_id INTEGER, t REAL, citations TEXT NOT NULL);
-- Two FTS5 tables over items (external content, text stored once; triggers keep them in step):
CREATE VIRTUAL TABLE fts_words USING fts5(text, content='items', content_rowid='id',
  tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE fts_tri USING fts5(text, content='items', content_rowid='id',
  tokenize='trigram');
```

**Sync (D36: on read, not hooks)**
- `index.open_index` (Search page, `cohr search`) lists the runs in `runs_dir` and re-indexes
  each run whose key changed, one transaction per run (`BEGIN IMMEDIATE`: delete its rows,
  insert fresh), and drops runs no longer on disk. The key is the mtime and size of
  `manifest.json` and every indexed file, not a hash of the manifest (which changes on every
  heartbeat and misses a hand-edited notes file). So stage finishes, Delete, Prune media, CLI
  runs and hand edits all reach the index with no code in `pipeline`, `jobs` or `storage`.
- A missing, corrupt or other-`schema_version` file is rebuilt into a temp file and swapped
  in (`atomic.atomic_path`). `cohr index rebuild` does the same on request.
- Deleting `index.db` is always safe.

**Queries (D37)**
- Terms come from `fts.query_terms` (Ask's code, moved to `src/cohr/fts.py`): words without
  English stopwords, or 3-character windows when the query has Chinese, Japanese or Korean.
- A CJK query searches every row through `fts_tri`. Any other query searches rows in other
  languages through `fts_words`, and ja/zh/ko rows through `fts_tri` with its 3+ character terms:
  unicode61 reads "チケットは390件" as one token, so "390" would never match it.
- Every term first (AND); if no row has them all, any term (OR), and the page says so.
  BM25 order, `snippet()` with control-character marks the page turns into bold after escaping.
- Digits are folded to ASCII in the stored text and in the query (D38).

**UI:** the **Search** page (`app_pages/search.py`, `layout="wide"`, between Ask and Runs): a
query box, filters (kind as pills, language, run date range), results with run, section or
kind, time and language, the snippet, **Listen** (Ask's right-side drawer: the segment's
audio from the hit's time, its slides) and **Open notes** (sets the current run and a
`ui.Focus`; Notes shows "From Search: [mm:ss](#seg-N)", because anchors only scroll within one
page, and starts that segment's player at the hit). An **Action items** tab lists every run's
tasks (run, date, owner, due, task, time), filterable by owner; selecting one offers the same
two buttons.

**Limits (accepted):**
- Keyword search only, in the language of the text: an English query doesn't find Arabic
  notes (it finds the English transcript of the same meeting). Semantic search across meetings
  would need an embedding model and a vector store (sqlite-vec); out of scope until there's a
  clear need.
- CJK terms under 3 characters can't match (D17), and Latin terms under 3 characters inside
  CJK runs ("Q3予算") neither.
- Spoken number forms aren't read: "354 mil", "35万4000" don't match "354,000" (search check:
  3 of 126 facts). Candidate: a second FTS column with `numbers.source_numbers_in` readings.

**Tests:** `test_index.py` (rebuild equals incremental updates; Delete and Prune update rows;
missing, garbage, old-schema and missing-table files rebuild; partial runs; unchanged runs
not re-indexed; AND then OR with a case where the fallback must not fire; trigram for CJK and
Latin-in-CJK; translated notes; digits across scripts; filters; sentence jump), `test_fts.py`,
`test_cli.py` (index rebuild, search), `test_app.py` (Search page: results, no matches,
drawer, Open notes banner, action items by owner).

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
As built (6e-2, after 1.0.0 merged without it): actions pinned by commit SHA like `check`,
the version passed to the shell through `env`, and `dist/*.whl dist/*.tar.gz` (uv build also
writes `dist/.gitignore`).

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
| `slide-conflict-en` | A number is *misspoken* in speech but correct on the slide (e.g. "forty" vs `14`). Correct output uses the slide value (§5.5 rule); since 6b-2 the notes show both ("$80,000 (slide: $18,000)"), which counts as the slide value used. |
| `xling-{es,ja}` → target `en` | Source language ≠ target language |
| `en` → target `{de,ar}` | Output in a non-English target, including right-to-left text |
| `long-en` (~2.5 h, audio + PDF deck, own `long` set; 6c) | Two-pass reduce (§4) and Ask ranking (§5.7). Realistic density (user, 6c): the topic loop once per region (5 regions × 8 topics, 40 slides), 36 discussion lines per topic, 88 facts (40 numbers, 20 decisions, 20 actions, 8 open questions). The region is in the spoken metric, the fact label and the slide title. Ask asks a fixed sample: 12 number, 6 action, the follow-up, 2 German, 3 near-miss |

### Metrics
| Layer | Metric | Scoring |
|---|---|---|
| VAD (§5.1a) | Speech-frame F1 vs. `speech_spans`; **text produced per minute of gap** | Exact |
| Transcribe | WER (CER for zh/ja/ko) on synthetic items and **FLEURS** test clips | `jiwer` |
| Slides | Title exact match, table-cell accuracy, number recall | Exact |
| Notes: **fact recall** | Share of planted facts found in the right section (decisions → Decisions, …) | Normalized match: numbers exact, owners exact, text via token overlap ≥ 0.6. **Cross-language** (notes language ≠ script language): see below |
| Notes: **citation accuracy** | Share of matched facts whose citation's refined jump time `t` (§5.6) is within ±15 s of the fact's `t0` (D19), or that cite its slide. A time-range overlap test would pass almost any citation, since segments span up to 5 min | Exact |
| Notes: **uncited claims** | Claims with `citations: []` (§5.6) | Exact (count) |
| Notes: **made-up items** | Decisions/action items in the notes that match no planted fact. An action item with the right task but another owner or due date counts (it states something not said); a real task with no owner stated doesn't (incomplete, not made up) | Exact (count) |
| Notes: **support rate** | Share of claims whose cited segments support them | LLM judge (secondary) |
| Output language | Notes are in the target language | `lingua-language-detector` |
| Ask | Planted-fact questions from templates (`evals.ask_questions`): answer contains the fact *and* cites its segment (±15 s) or slide; plus follow-up, German alone / after history and near-miss rates (rule in NOTES.md, 6a; three near-misses per item from 6b, one per `NEAR_MISS` template) | Exact |
| Notes: **open questions** | Planted open questions (6b, 1–2 per item) found under Open questions (token overlap ≥ 0.6): open-question recall, kept out of fact recall; **made-up open questions** = claims there matching no planted one | Exact |
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
  temperature 0. The claim is prefixed with its section ("Decision: …"; action items carry owner
  and due), since a decision or an open question claims something was agreed or asked. A cited
  segment is its **transcript only** (not the Map notes Reduce also saw, so a Map error can't
  support itself); a cited slide is its slide text. Support means "the cited text says it", not
  "it's true": a misspoken number cited to the transcript is supported (fact recall and the
  slide-conflict check cover truth).
- Invalid JSON: one retry with `judge_retry.md`, then "unjudged" (in the support-rate
  denominator, never counted as supported). Uncited claims aren't judged.
- **Support rate** = "yes" / cited claims; partial, no and unjudged are reported as counts.
  **Judge recall** = judge_only facts the coverage question finds / judge_only facts.
- Answers are cached per item in `evals/data/judge/<item>.json`, keyed by the question's text,
  the hash of every judge prompt and the judge revision: re-scoring with nothing stale loads no
  model. The judge loads on the first cache miss, after `release()` of the pipeline models (§4),
  and its cost is its own report row, outside `stages` and the gate.
- **Calibration:** 30 hand-labeled claim/segment pairs (`evals/gen/judge_calibration.json`). Record
  the judge's agreement; if it's below 80%, judge metrics are reported but excluded from the gate.
  `cohr eval --calibrate-judge` writes `evals/reports/<ts>-judge-calibration.{json,md}`. Don't
  tune the judge prompts on these 30 pairs: they're the only held-out measure of the judge.
- Optional `--judge claude` for the public and synthetic sets only (needs an API key). User
  recordings are never sent to any judge.

### Eval sets
| Set | Contents | Time | When |
|---|---|---|---|
| `quick` | `clean-en`, `noisy-en`, `pdf-slides-en`, `slide-conflict-en` | ~5 min | After prompt or model changes; before every commit to `reduce.py`/`mapper.py` prompts |
| `lang` | `conflict-de`, `clean-ar`, `clean-hi` (R7B in de/ar/hi), `clean-sw`, `clean-bn` ("Other (beta)", translated), `xling-es`, `xling-ja` (Spanish and Japanese meetings, English notes; 6d-2) | ~20 min | After Reduce, Map, number-reading or translate changes (6d-1, 6d-2); report-only baseline `evals/baseline-lang.json`, `-c seed=1 --no-ask` |
| `long` | `long-en` | ~1 h (generation, ~40 min pipeline, Ask) | After two-pass Reduce or Ask ranking changes; report-only baseline `evals/baseline-long.json` |
| `full` | The `quick` and `lang` items + 14-language FLEURS sample (20 clips each, Transcribe only) + 5 AMI meetings (3.05 h of audio, English notes); `long-en` stays in `long` (6e-1, user) | ~1 h (estimate: synthetic ~20 min, FLEURS ~4, AMI ~26 + judge ~9) | Before every version bump (§16), followed by `long` |

External data is downloaded at eval time into `evals/data/` (gitignored). Nothing from it is
committed or redistributed.

| Data | Source | License | What we use |
|---|---|---|---|
| FLEURS | HF dataset `google/fleurs` at revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`: `data/<lang>/test.tsv` and `data/<lang>/audio/test.tar.gz` **streamed** (6e-1, user): the tars are 159–658 MB each (5.9 GB for 14 languages), so the sample is 20 clips drawn by seed from the first 100 tar members, one per sentence (the TSV repeats a sentence per speaker); ~0.9 GB read once, ~10 MB kept | CC BY 4.0 | 20 test clips per language + reference transcripts |
| AMI audio | Official mirror: `https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<id>/audio/<id>.Mix-Headset.wav` (16 kHz mix of the headset mics, whole meeting; 6e-1, user). Not `edinburghcstr/ami`: it holds one row per utterance per speaker (uppercase, no punctuation), so a meeting would have to be rebuilt without crosstalk or room noise | CC BY 4.0 | 5 meetings, ~352 MB |
| AMI reference summaries | Official annotations: `https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip` (22.9 MB; `LICENCE.txt` inside) | CC BY 4.0 | `abstractive/<meeting>.abssumm.xml` (142 meetings) |

**AMI join and scoring:**
- Join audio and summaries on meeting ID (e.g. `ES2004a`); use only meetings present in both.
  The 5 meetings are fixed in `evals/sets/full.yaml` (6e-1): ES2004b, ES2004d, IS1009b,
  IS1009c, TS3003c (AMI test split, three scenario series, 30–43 min each; kickoff "a"
  meetings skipped, ES2004c swapped for ES2004d: 1 decision, 0 actions). Reference sentences:
  abstract 43, decisions 23, actions 11, problems 12, so one sentence moves a section's
  coverage by 4–9 points.
- Each `abssumm.xml` has `<abstract>`, `<decisions>`, `<actions>` and `<problems>`, each a list of
  `<sentence>` elements. Map them to our sections: `decisions` → Decisions, `actions` → Action
  items, `problems` → Open questions; `abstract` → Summary **and Key points** (6e-1: our Summary
  is one claim, AMI abstracts are 5–11 sentences; `evals.AMI_HOME`).
- Metric: per section, the share of reference sentences the judge says are covered by our notes.
- Noise (8h): the judge reads a sentence against the whole list of claims in its sections, so
  a change elsewhere in the list can flip it. ES2004d's problem sentence flipped when two open
  questions lost their "?", and IS1009b's "plastic cases" sentence when its claim moved from
  Summary to Key points. Read a change of one sentence per meeting as noise until the
  sentence's own claim is shown to have moved; the cached answers (`evals/data/judge/`) score
  stored notes offline.
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
  fact recall drops > 5.9 points, citation accuracy drops > 6.9 points (D25), Ask accuracy drops > 5
  points, Ask citation accuracy drops > 3 points, near-miss drops > 16.7 points (2 of 12),
  made-up items or made-up open questions rise by > 2, uncited claims or the VAD gap text
  rate increase, or any stage gets > 25% slower (D16, D24; tables `MAX_DROP` / `MAX_RISE` in
  `evals.py`). Report-only: the judge metrics and the one-question-per-item Ask rates
  (follow-up, German).
- `cohr eval --update-baseline` rewrites the set's baseline (`evals/baseline.json` for `quick`,
  `evals/baseline-<set>.json` otherwise). Only commit it with the reason in the commit message
  (e.g. "new reduce prompt: recall +4, accuracy −1"). Only `quick` and `full` fail the gate;
  `long` compares with its own baseline, report-only (one run, no noise measured).
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
| `map.json` | `MapFile` | `target_language`, `notes[{seg_id, summary, key_points[], decisions[], action_items[{owner?, task, due?}], open_questions[], terms[], raw: bool, number_repairs[{original, repaired}], unverified_numbers[]}]` (v3: number guard, D20) |
| `notes.json` | `NotesDoc` | `sections{summary, key_points, decisions, action_items, open_questions}` as lists of `{text, citations[{kind: seg|slide, id, t?}]}` (`t` = refined jump time, seg only; `citations: []` = uncited, §5.6); action items also have `owner?`, `due?`; `passes: 1|2`, `language`, `raw` (R7B reply; pass 2's when two-pass), `two_pass?{groups[{seg_ids, slide_ids, raw}], resolved{marker, claim, group, uncited}}` (v3: `t` placed at the cited sentence, D19; v4: `two_pass`, 6c) |

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
| D11 | Slide alignment threshold (§5.4) | 0.15 | Maximize `pdf-slides-en` accuracy | Phase 3, re-check Phase 4b-1 | **0.15**: on `pdf-slides-en` all 8 slides align correctly at every threshold 0.05–0.35 (best scores 0.35–0.50); 0.15 stays between those and the 0.125 of an undiscussed slide (Phase 3 fixture) |
| D12 | Fuse window length (§5.4) | 300 s | Fact recall vs. speed on `quick` | Phase 4b-1 | **300 s**: vs 120 s, recall 82.3% vs 79.4%, made-up items 3 vs 6, Map 123 vs 180 s/audio h (one seed per item; re-check against D16 noise). Re-checked in 6b-2 (seeds 1–3): 180 s recall 89.7 / 85.3 / 83.8% vs 300 s 83.8 / 85.3 / 86.8% (mean +1.0, inside the 5.9 noise), 11 vs 8 segments on `quick`; **300 s kept** |
| D13 | Notes column `width=880` shrinks below 880 px (§7) | 880 | Resize check | Phase 5 | **Shrinks, 880 kept**: 377 px at a 420 px window, 422 at 800 (sidebar closed), 559 at 1512, capped at 880 at 2560. But at 800 px with the sidebar open the columns get 209 / 128 px (they only stack at ≤ 640) |
| D14 | Anchors scroll the Sources panel (§7) | Anchors | Click check; fallback = drawer buttons | Phase 5 | **Anchors work**: clicking `[5:41](#seg-1)` scrolled the Sources container 0 → 2358 px and brought its block to the top; no drawer needed |
| D15 | Judge agreement ≥ 80% (§18) | Judge metrics excluded from gate | 30 labeled pairs | Phase 4b-2 | **Pass, no margin**: 3-way 80.0% (24/30), yes vs not-yes 96.7% (29/30). All 12 "yes" right; 4 of 8 "partial" judged "no" (an added detail read as a contradiction), 1 "partial" judged "yes", 1 "no" judged "partial". Support rate counts only "yes", so the partial/no confusion barely moves it. Judge metrics stay out of the gate until D16 (Phase 6) |
| D16 | Eval gate thresholds (§18) | recall −5, accuracy −3, speed +25% | Run-to-run noise on 3 `quick` runs, same media, pipeline seeds 1–3 (`-c seed=N`): threshold = max(default, 2× noise), Ask metrics included | Phase 6a | **Recall −5.9 (noise 2.9), made-up +2 (noise 1), rest at defaults** (citation −3, uncited and gap text no rise, stages +25%: noisiest Reduce 10.5%); Ask accuracy −5 and Ask citation −3 (noise 0, defaults borrowed from recall/citation). Report-only (user, 6a): follow-up and German after history (noise 25 points: one question per item), German alone and near-miss (25-point steps), support rate (noise 9.1, D15 no margin). Open questions not counted as made up: 13 / 11 / 4 over the seeds. Baseline = seed 1 |
| D17 | FTS5 available in uv's Python (§13) | Assumed | `PRAGMA compile_options` | Phase 6c (Ask ranking needs it; was Phase 7) | **Available**: uv's Python 3.13.12 has SQLite 3.50.4 with `ENABLE_FTS5`; the `trigram` tokenizer works on Japanese. Trigram can't match terms under 3 characters, so CJK questions are split into 3-character windows |
| D18 | Regional Tiny Aya variant per target language (§5.5) | Global for every language (`map_model_by_lang` empty) | Planted-facts `en → target <lang>` for 2 languages per variant's region, Global vs variant, scored with cross-language matching (§18): adopt if exact-field fact recall improves by ≥ 2× the D16 noise and output-language accuracy is no worse | Phase 6d-2 || **Not adopted (Fire tried, user's choice)**: `mlx-community/tiny-aya-fire` converted to 8-bit locally (`ModelSpec.quantize_bits`), Global vs Fire on `clean-hi` and `clean-bn`, seeds 1–3: hi 59.0 → 56.4% (−2.6), bn 87.2 → 89.7% (+2.5), both under the 5.8-point bar (2 × 2.9); output language 3/3 either way. `map_model_by_lang` stays empty; Earth and Water not tried |
| D19 | Citation refinement: min chunk score and the eval's jump tolerance (§5.6, §18) | score ≥ 0.1, ±15 s | Share of planted facts whose refined `t` lands within tolerance on `quick`; check that 15 s isn't hiding misses (look at the error distribution) | Phase 4b-1 | **0.1 and ±15 s, plus a sentence offset inside the chunk**: chunk `t0` alone was a median 12.5 s early (61% within 15 s); with the offset 98% (53 of 54 errors ≤ 5 s, so 15 s hides nothing). Min score 0.0 = 0.1 > 0.2 (94.6%) > 0.3 (91.1%) |
| D20 | Map number guard: repair rule (§5.5) | Repair when exactly one source number is one digit edit or a factor of 10/100/1000 away; else flag | `quick` before/after on the same seeds: number recall up, no wrong repair, no other metric worse; if a wrong repair shows up, narrow the rule (e.g. factor-of-10 only) | Phase 4b-1b | **Default rule kept**: 8 repairs on `quick`, all to the transcript's value (0 wrong); number recall 81.2% → 87.5%, fact recall 82.3% → 85.3%, made-up 3 → 3, citation accuracy 98.2% → 100%; numbers in no source 6 → 1 (a flagged `3,360` with two candidates). Short numbers are weak evidence (any one-digit number is one edit from the next): watch on `full`. **6b-2 narrowed the candidates** (user chose the separator rule): a wrong repair ("$770,000" → $740 via the decimal-comma reading) is now flagged, and "November 2nd" is repaired to the only day said (22nd); `quick` seed 1: recall 83.8 → 85.3%, made-up 1 → 0, wrong owner/due 1 → 0, unsupported numbers 2 → 1 **6d-2: sources and keeping read by language**: the sources are read the way the meeting's language writes numbers (`source_numbers_in`: es "11.800" = 11800 and "mil", ja 万 and kanji numerals), and a number is kept only by the notes language's readings, as repairs already were (English "$34,000" was kept as the 34 of "34%"). xling-es recall 69.2 → 92.3% (4 right repairs), xling-ja "$13,000" → $134,000 and "$34,000" flagged; quick unchanged (10 repairs) |
| D21 | Notes layout in medium windows (§7) | Two columns; close the sidebar | Options within the no-CSS rule (NOTES.md 2026-10-11 5a finding) | Before 5b | **Beside/Below toggle** in the Notes header, default Beside, per session. Rejected: Sources always in the drawer (undoes D14, loses side-by-side on wide screens), a "close the sidebar" hint only, `initial_sidebar_state="collapsed"` on Notes (localStorage overrides it). Built in 5b |
| D22 | Ask history format (§5.7) | R7B's multi-turn `rag` messages | Manual check on clean-en: follow-ups cited and answered | Phase 5b | **Folded into one user turn** (Reduce's `prompt()`), language line last. Native multi-turn: 4 of 6 answers were a new tool call, no answer. Native + `<|START_RESPONSE|>` prefill: answered, but follow-ups lost their citations. Folded: every English answer cited, jump times inside the truth spans |
| D23 | Ask answer-language detection (§5.7) | lingua, fallback = target language | lingua confidences on sample questions in 9 languages | Phase 5b | **≥ 2 words (or CJK), confidence ≥ 0.2, lead ≥ 0.05**: "Priya?" read as Indonesian at 0.56 and "Wat is het budget?" tied nl/en at 0.25; full sentences scored 0.25–1.0 with a clear lead |
| D24 | 6b gate additions (§18) | Report-only (6a) | Same rule as D16 on the 6b media and final config: 3 `quick` runs, pipeline seeds 1–3 | Phase 6b | **Near-miss −16.7 points (noise 8.3: 7, 7, 6 of 12), made-up open questions +2 (noise 1: 0, 1, 0)**. Also measured: citation accuracy noise 7.1 (92.9 / 96.6 / 100%), above its D16 gate of 3; left at 3 (baseline seed 1 is the low point), revisit in 6c |
| D25 | Citation-accuracy gate after the bundling fix (§18) | −3 (D16) | Reduce bundling fix first (one fact per claim), then D16's rule on 3 `quick` runs (seeds 1–3): max(3, 2 × noise) | Phase 6c | **−6.9 points (noise 3.4: 96.7 / 96.6 / 100%)** after `reduce.md`'s one-fact-per-item rule (recall 82.3 → 89.7%, citation 92.9 → 96.7%). Fact-recall noise on the same runs rose to 5.9 (89.7 / 85.3 / 83.8%); its gate stays −5.9 (user: seed 1 is the high point). Key points stay bundled: Reduce copies Map's bundled key points word for word (6b-2) |
| D26 | Two-pass reduce grouping (§4) | Groups of consecutive segments filling ~75% of `max_input_tokens` | `quick` with a lowered cap forcing 2–3 groups vs single pass on the same media: citation accuracy within D25, no uncited rise, every pass-2 citation resolves; then `long-en` | Phase 6c | **40% (`GROUP_FILL` 0.4), with `reduce_part.md` (decisions/actions before figures-only key points) and 4096 output tokens.** `quick` at cap 4000 (2 groups per item) vs single pass: recall 89.7 = 89.7%, citation 100 vs 96.7%, uncited 0, made-up 3 vs 2, 91 of 91 pass-2 citations by claim match. `long-en` (2.2 h, 32K cap): at 75% (2 groups of ~23K) pass 1 ran out of output inside KEY_POINTS (recall 11.3 → 37.5% with the prompts and 4096); 0.4 (4–5 groups of ~12K) recall 53.8% on the same media, 67.5% after fact-free filler (citation 98.1%, 5 group fallbacks of 139). Pass 2 still loses about a third of pass 1's decisions (carry-over, §9 6c) |
| D27 | Ask ranking under the cap (§5.7) | FTS5 BM25 over segment + slide documents, question terms OR'ed, top-ranked that fit | Ask accuracy on `quick` with a lowered cap vs the full-document baseline, and on `long-en`'s question sample | Phase 6c | **FTS5 BM25 over the question plus the last turn, at most the top 12 documents that fit, sent in meeting order.** `quick` at cap 4000 vs no cap / vs the old transcript trimming: Ask accuracy 96.2 vs 98.1 / 98.1%, citation 98.0 vs 100 / 100%, follow-up 3/4 vs 3/4 / 1/4, near-miss 4/12 vs 7/12 / 5/12 (any way of fitting the cap costs near-misses; ranking is within one question of trimming and scales). `long-en`: filling the 32K cap (~60 documents) gave 61% at ~100 s per answer and citation loops; top 12 gives 83.3%, citation 93.8%, ~32 s |
| D28 | Speech/slide conflict (§5.5, §18 `slide-conflict-en`) | Prompt: use the slide's value | Map ignores the prompt rule on every seed; a guard that swaps in the slide value would be wrong when a speaker corrects an outdated slide | Phase 6b-2 | **Show both** (user): a slide table row whose one value was never said, while the sentences naming its label hold exactly one number on no slide, gets "(slide: $18,000)" after the spoken value in Map's strings; `slide_conflicts` in MapFile v4. `quick` seed 1: slide_value_used false → true, recall 85.3 → 86.8%; dry run on 36 real segments found only the planted conflict |
| D29 | Right-to-left notes on the Notes page and in `notes.md` (§5.6, §7) | Plain Markdown, no direction marks | Browser check of `clean-ar` (6d-1, Claude in Chrome) within the no-CSS rule | Phase 6d-1 | **Unicode isolates per line**: Arabic-only lines read right (one RTL run, citation at the end), but a line with Latin names or dates ("(المسؤول: Tom; الموعد: Monday)") was reordered, citation on the far side. `render.isolate` wraps each claim line, the summary paragraph and Ask answers in RLI … PDI for ar/fa/he/ur (`notes.md` format v2): reading order right, citation links still jump to Sources. Bullets and alignment stay on the left (that needs `dir`/CSS, not allowed) |
| D30 | Where "Other (beta)" translation runs (§5.6) | — | Proposal before coding (6d-1 kickoff) | Phase 6d-1 | **Own `translate` stage** (user): between reduce and render, new `notes_translated.json` (TranslatedNotes v1), NotesDoc unchanged; a translate-prompt change re-runs only translate + render, and the English notes stay for the eval's citations-identical check. Rejected: inside reduce (two models in one stage, a prompt change re-runs R7B). `clean-sw` seeds 1–3: Swahili 3/3, citations identical 3/3, exact recall 92.3% = the English notes' on each seed, 4 claims kept English per run |
| D31 | What `full` gates on (§18) | — | Proposal before code (6e-1 kickoff) | Phase 6e-1 | **Decided 2026-10-14 (user, after the first full run)**: the `quick` items' metrics at the D16/D24/D25 thresholds and the speed/memory rows gate; FLEURS WER/CER, AMI coverage and the `lang` items are report-only until their run-to-run noise is measured (D16's rule). Built in 6e-1 step 3: `gate: [quick]` in full.yaml, the report's `gate_summary`/`gate_stages`. The gate never compared memory (quick neither: no threshold exists), so only the speed rows gate |
| D32 | Reduce output cap (§4, §5.6) | 2048 (two-pass 4096) | Token counts of existing replies, AMI run (6e-2 step 0) | Phase 6e-2 | **8192 for every pass, a cut reply flagged, no retry** (user): `mlx_lm.stream_generate` keeps `finish_reason`; a reply that stops at the cap without `<|END_RESPONSE|>` counts in `NotesDoc.truncated` (v5), the run log and a Notes page warning, and its claim is kept uncited. AMI coverage 44.9 → 52.8%, 0 cut; quick notes byte-identical; Reduce 139 s per audio hour. Not retried: at temperature 0 a retry writes the same start again |
| D33 | Tasks Map wrote as tasks but Reduce filed as decisions (§5.6) | Prompt rule | Rule on `map.json`, replay on AMI and quick | Phase 6e-2 | **Pure-Python move** (user): a decision whose cited segment has a Map action item scoring ≥ 0.6 that beats every Map decision there (or ties with them and names an owner) moves to Action items, merged into a matching one. AMI ES2004b 15 → 9 decisions; full action recall 85 → 95%; long-en +3 tasks found, +4 made-up (pass 2 sees the moved items). Map's own task-as-decision writing (Victor-style) after v1 |
| D34 | Owners written in the notes' script (§5.6) | As written | User's choice (6e-2 step 0): fix the notes | Phase 6e-2 | **anyascii** (user): a one-word non-Latin owner absent from the cited text becomes the Latin name said there with the same consonants, the closest spelling winning and a tie keeping the owner. clean-hi recall 54 → 92%; lang seed 1 81.3 → 86.8% |
| D35 | 万 numbers in ja notes (§5.5) | Read as written | User's choice (6e-2 step 0: non-ja notes only), xling-ja seeds 1–3 | Phase 6e-2 | **Not kept**: grouping 35万4000 and telling Map the source language left 354,000 missed on every seed (Map wrote 3,400,000). Patch kept out of the tree; Map digit miscopying for ja after v1 |
| D36 | Who keeps the search index in step with the runs (§13) | Hooks after each stage, Delete and Prune, keyed by `manifest_hash` | Proposal before code (Phase 7 step 0) | Phase 7 | **Sync on read** (user): opening the index re-indexes each run whose key (mtime + size of the manifest and every indexed file) changed and drops deleted runs; no hooks in pipeline, jobs or storage. A manifest hash changes on every heartbeat and misses a hand-edited notes file |
| D37 | Tokenizer for zh/ja/ko text in the index (§13) | unicode61 for everything | Proposal (step 0), then `test_index.py` and the search check | Phase 7 | **Two FTS5 tables (unicode61, trigram) over the same rows**; a CJK query uses trigram for all rows, any other query trigram for ja/zh/ko rows only. Found while testing: unicode61 kept "チケットは390件" as one token, so "390" missed it |
| D38 | Digits in other scripts (§13) | As written | Search check (Phase 7 step 4) | Phase 7 | **Folded to ASCII** in the index and the query: Tiny Aya writes Bengali notes with Bengali digits ("$৪২,৫০০"); clean-bn numbers found in the notes 2/8 → 8/8. Snippets show ASCII digits |
| D39 | Jump time when the notes and the meeting share no words (§5.6) | Segment start | Offline replay of `refine_time` on the stored lang runs (8a step 0), then lang seed 1 and quick | Phase 8a | **Language-neutral anchors** when no chunk scores ≥ 0.1: the span's numbers (read in each text's language) and its `[owner: …]` (meeting's spelling for the segment, whole word or same-consonant katakana); most anchors wins, ties by word overlap then earliest; segment start when none. Citation accuracy clean-hi 58.3 → 100%, xling-ja 70 → 90%, quick identical. Names elsewhere in claims not used (تم = Tom, مللي = Maria in the replay) |
| D40 | Dues and owner-less tasks the meeting didn't give (§5.6) | As Map and Reduce wrote them | Step 0 measurement on the stored long-en and AMI runs, offline replay, then quick, lang seed 1, long-en and AMI | Phase 8b | **Due guard + vague-task move on the final notes** (user): a due its cited segments and slides don't say is dropped, never replaced (English: words within a window, `dues.said`; other languages: digits only, in the meeting's language; uncited kept); then an English action item with no owner and no due starting Continue/Ensure/Keep/Monitor/Maintain moves to Key points. long-en made-up items 12 → 7, recall unchanged; AMI 34 unsaid dues dropped and 4 placeholders cleared (1 of 39 left), coverage unchanged; quick and lang identical. Rejected: "no owner and no due" alone (9/9 on long-en but would move 25 of AMI's 47 tasks) and an ask-cue check on the source (no signal). Map's example due left as is (post-v1). Eval: an empty due field reads a due written in the task text, so clearing a field can't turn a wrong date into a match |
| D41 | Ask on long meetings: citation placement and Map's values in its documents (§5.5–5.7) | First span places a segment citation; Map's dates and dues as written | Step 0 on the cached Ask replies and stored notes (offline replay), then quick, lang seed 1 and long-en | Phase 8c | **Best-matching span places the citation; Map's dates repaired or flagged, unsaid dues dropped on map.json** (user: on Map's items, one value for every reader). Date repair only from exactly one said date of the same month a digit edit away, English notes of an English meeting; dues by D40's rule (`dues.due_said`). long-en Ask 77.8 → 88.9%, citation 93.3 → 100%, made-up items 7 → 4; quick citation 98.4 → 100%; lang identical but one judged claim. map.md's example kept (the check drops every copy). Rejected: Ask snippet-only fix (leaves map.json wrong), an Ask prompt rule (a-f61 left open) |
| D42 | Missed action items: tasks Map wrote as decisions or without owners, tasks the notes lost (§5.5–5.6) | As Map and Reduce wrote them | Step 0 on the stored quick, lang and long runs (pass-1 replies are stored in notes.json), offline replay incl. AMI, then lang seed 1, quick and long-en | Phase 8d | **Python on map.json and the final notes** (user): Map's "X will …" decision → X's task when X is said in the segment; an owner-less task's owner from its own text or its best sentence's one cued person (English notes; English meeting for the sentence); an `[owner: ?]` notes item takes its Map task's owner (merged into a duplicate); an owned Map task no notes item states is added, cited with `[owner: …]` as anchor. Logged, no MapFile field (user: Ask keys move only where map.json changes). lang seed 1 action recall 80.0 → 94.3%; long-en action recall 55 → 70%, fact recall 67.5 → 71.3%, made-up 4 → 6 (owner-less pass-1 tasks pass 2 kept, not the rules: replayed without them, still 6); quick identical; AMI replay 0 changes. Rejected: a map.md prompt rule (Python reaches every Map-reachable case but conflict-de f6, a German task of an English meeting) and names anywhere in a sentence (`said_names` keeps "Okay", "Monday", "Action") |
| D43 | Action items Map never wrote (§5.5) | As Map wrote them | Step 0 scan of the stored runs and quick at pipeline seeds 2 and 3, then quick seeds 0/2/3, lang seed 1 and long-en | Phase 8e | **Python finds the sentences, one Tiny Aya call decides** (user, over a Python-only net and a map.md rule): in an English meeting, a sentence giving one person a task in a cue position while no Map task of the segment is theirs goes to `prompts/map_tasks.md` (temperature 0, no examples, notes language); Python keeps the owner, completes a task cut short from its sentence (English notes), and runs Map's number, date and due checks. Map's misses change with the seed (quick action recall 95/80/75% at seeds 0/2/3), every one a flagged sentence, 0 false flags (AMI: no sentence names anyone). quick 100% at all three seeds; lang seed 1 94.3 → 97.1%; long-en action recall 70 → 90%, fact recall 71.3 → 78.8%, Ask 88.9 → 94.4%, made-up 6 → 7 (f15, said "Oma"). Rejected: a verb list for must-not-fire (would fit the generator), a prompt rewording for f86's short task (Tiny Aya kept it short and added the name in German) |
| D44 | Missed decisions: Map never wrote them, or Reduce lost Map's (§5.6) | As Map and Reduce wrote them | Step 0 scan of every stored run (AMI included), lang map.json read by hand, pass-1 replies; offline replay of the stored notes.json; then quick seeds 0–3, lang seed 1 with the judge, long-en | Phase 8f | **Python after Reduce** (`reduce.add_said_decisions`, user, over a D43-style call, a map.md change and re-adding every lost Map decision): in an English meeting a cue sentence ("We decided/agreed to …", "the decision is to …"; not asked, denied, "whether" or an earlier meeting's) no notes decision or action item states becomes a decision in its own words, or its key point moves to Decisions (English notes); other notes languages get the segment's unstated Map decisions up to its cue count, placed at the cues when no decision cited the segment. No prompt or MapFile change (Ask keys and REDUCE_HASH stay). quick decisions 75/81/81/88 → 100% at seeds 0–3, fact recall 91.2 → 97.1% (seed 0); long-en 70 → 95%, fact recall 78.8 → 85.0%; lang judge-covered 21 → 24/28; made-up items and Ask unchanged; AMI: 1 cue, already a decision, 0 changes. Rejected: re-adding every Map decision the notes lack (8 on AMI, several not decisions, plus long-en's "Leave Bitcoin acceptance decision open"), a map.md change (the net catches the misses at every seed), a model call for the 3 Map never wrote in de/ar notes (Tiny Aya misjudged cue sentences in German in 8e) |
| D45 | Missed numbers: one wrong digit, a wrong repair, figures Reduce lost (§5.5–5.6) | As Map and Reduce wrote them; D20's repair rule | Step 0 trace of the 17 misses (pass-1 replies read), every stored repair counted, AMI's said numbers in no Map item or notes; offline replay of the stored notes.json; then quick seeds 0–3, quick with Ask and the judge, lang seed 1 (`--rerun map`), long-en and AMI | Phase 8g | **Guard bound + Python after Reduce** (user, over a Map-side net and a map.md change): a plain number of one or two digits is never repaired (AMI IS1009b "60%" → 40, the only wrong one of 41 stored repairs; the other short ones are days), and a candidate another number of the claim states is dropped. A digit edit preferred over a factor would have fixed xling-ja f9 but repaired pdf-slides' "6,900 candidates" (69 said) to the $8,900 cloud bill at seed 1: dropped, f9 stays flagged. `reduce.add_lost_numbers`: a Map key point with a figure (digits, ≥ 10, notes language, D38 folding) no claim states is added citing its segment, or replaces the key point citing it that kept some of its figures, and cites the segment's slides that show an added figure; flagged numbers skipped. No prompt or MapFile change. long-en numbers 77.5 → 85.0%, fact recall 85.0 → 88.8%; quick numbers 93.8 → 96.9% (seed 0), 100% at seed 3, seed 1 unchanged; lang 91.1 → 94.6%, unsupported 1 → 1; made-up items, Ask and citation accuracy unchanged; quick judge 95.1 → 93.8% (it reads "(slide: $18,000)" as contradicting $80,000). AMI: 6 key points added, all figures said; IS1009b "60%" kept and flagged (unsupported 0 → 1: "the rest" has no digit). Rejected: a sentence net for numbers Map never wrote (copies evals/gen's "That's compared with" sentences and AMI fragments; can't reach xling-ja), a map.md change (misses move with the seed) |
| D46 | Reduce replies with an unknown heading or repeated lines (§5.6) | Unknown-heading text to Summary; repeats kept | Step 0: both AMI states scored from the judge cache, raw replies read, Map items matched to the dumped lines; offline replay of every stored notes.json (main's code reproduced each); then quick seeds 0–3, quick with Ask and the judge, lang seed 1 and AMI (`--rerun reduce`) | Phase 8h | **Parser** (user): a list item under an unknown heading goes to the section of the cited segments' Map item it matches best (≥ 0.6, key points first on a tie), else Summary; prose under it stays Summary. A claim repeated in its section with the same citations is kept once. IS1009b's dump: 20 key points, 2 decisions, 32 lines → 22 (Map's items word for word, every match 0.86–1.00); TS3003c: 10 repeated decisions dropped. Fires on no other stored run. No NotesDoc, prompt or REDUCE_HASH change. Rejected: continuing the section before the heading (8g's candidate: the dump holds key points and decisions both, so all would have become open questions) |
| D47 | Segments a section skipped (§5.6) | As Reduce wrote them | Same runs as D46; long-en's lost-item candidates checked against its planted facts | Phase 8h | **Python after Reduce, one-pass notes only** (user, over also filling key points and the `[due: ?]` input change): `reduce.add_skipped_segments`, after D44: a segment no decision cites gets its Map decisions the notes don't state (`_lost_map_decisions`: not the segment's tasks or open questions), one no open question cites gets its Map open questions the notes don't ask; each cites its segment, in segment order; a key point stating only the decision moves to Decisions; items with no letter or digit (Map's "[]") skipped. AMI coverage 44.9 → 50.6% (40 → 45 of 89: IS1009b teletext, whole-house and budget, TS3003c's battery once its repeats went, ES2004b Video Plus, ES2004d's noise sentence back; IS1009b's plastic-cases sentence flipped away), decisions 30.4 → 47.8%, problems 16.7 → 33.3%, support 80.3 → 80.1%; quick seeds 0–3, lang seed 1 and long-en notes identical (it never fired). Rejected: key points too (abstract held at 58–60%; the fills read as padding), two-pass notes (long-en's Map wrote the planted question "sponsor the local hackathon" as a decision), restoring the old input (the Oct 28 reply also dropped a segment; it would move every set's Reduce input) |

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
  wrappers (vad, asr, vision, mapper, reduce, judge, models). Unit tests run in CI without a GPU.
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

### Kickoff prompt for Phase 4b-1b (Map number guard)
```text
Continue cohr: Phase 4b-1b (Map number guard). Read CLAUDE.md, then PLAN.md sections §9 (the
4b-1b bullet), §5.5 (Number guard), §19 (MapFile), §20 (D20) and §18 Metrics. Then read the
latest NOTES.md entry (4b-1: "Open: Map changes digits" has the observed cases).
Work on branch phase-4b-1b-number-guard. Follow the 4b-1b acceptance criteria in §9:
1. Move digits/numbers_in from src/cohr/evals.py into src/cohr/numbers.py. No behaviour change:
   tests/test_evals.py must pass unchanged (only its imports may move).
2. The guard in mapper.py: pure function, applied to each parsed segment before saving. Repair
   only when exactly one source number qualifies; otherwise keep the text and list the number
   in unverified_numbers. Never drop a claim, never guess. Skip raw (JSON fallback) segments.
3. MapFile v3 with unverified_numbers; regenerate the schema snapshots.
4. Unit tests from the §9 list (no models).
5. Eval: add map_numbers_repaired, map_numbers_unverified and unsupported_numbers per item and
   in the summary.
6. Record the "before" first: run `cohr eval --set quick` on main's code with the eval metrics
   from step 5 in place but the guard switched off (`-c map.number_guard=false`), then
   with the guard on, same seeds. Compare number recall, fact recall and made-up items, and
   check every repair in map.json against truth.json: a wrong repair is a finding, not noise.
   Decide D20 from that.
Running `cohr eval --set quick` is fine. Ask me before anything in `full`, before `pytest -m
models`, and before new model downloads. No new dependencies are expected; if one is needed,
check its license (§11) and ask.
When done: tick the checklist, update §20 and NOTES.md, make sure the gate passes, and open a PR
(ask before pushing). Stop there; 4b-2 (the judge) is a separate session.
```

### Kickoff prompt for Phase 4b-2 (judge)
```text
Continue cohr: Phase 4b-2 (judge). Read CLAUDE.md, then PLAN.md sections §9 (the 4b-2 bullet),
§18 (Judge, Metrics, Reports and the release gate), §4 (memory policy), §20 (D15) and §6
(prompts, layout). Then read the latest NOTES.md entry (4b-1b) for the current quick numbers.
Work on branch phase-4b-2-judge. Follow the 4b-2 acceptance criteria in §9:
1. Judge wrapper in a new src/cohr/judge.py (Qwen3 judge, mlx-lm, the "judge" ModelSpec already
   pinned in config.py and already in the HF cache). Import mlx/mlx_lm only inside functions, and
   add judge.py to the model-wrapper lists in CLAUDE.md and §8. Prompts in src/cohr/prompts/:
   judge.md (claim + cited text → {"supported": "yes|partial|no", "reason": str}) and one for
   cross-language coverage (fact + section claims → {"covered": bool}). JSON only, temperature 0,
   one retry on invalid JSON, then count the pair as unjudged (never as supported).
2. Calibration BEFORE running the judge on it: draft 30 claim/cited-text pairs from the current
   quick runs (evals/data/runs, 4b-1b outputs) into evals/gen/judge_calibration.json with your
   labels and a one-line reason each. Real outputs are mostly supported, so mix in negatives:
   claims paired with the wrong segment, and the known errors (slide-conflict $80,000, a flagged
   Map number). Aim for roughly 12 yes / 8 partial / 10 no. Synthetic text only. Then STOP and ask
   me to review the labels; don't show the judge's answers before I've approved them.
3. D15: agreement = share of the 30 pairs where the judge's 3-way label equals mine; also report
   yes vs not-yes. Pass at ≥ 80% on the 3-way label. If only the binary passes, report both and
   ask me.
4. Eval: support rate (share of cited claims the judge marks "yes" against their cited segments
   and slides; "partial" and unjudged counted apart) per item and in the summary, and judge
   recall for cross-language text facts (the judge_only facts). Cache judge answers per item
   under evals/data (key: claim, cited text, prompt hash, judge revision) so re-scoring with
   nothing stale still loads no model. Load the judge only after the pipeline models are
   released (§4) and report its cost as its own row. Judge metrics stay out of compare() (the
   gate) in this phase whatever D15 shows; their thresholds come with D16 in Phase 6.
5. Unit tests without models: a fake judge, JSON parsing and the retry, agreement, support
   rate, judge recall, the cache (second scoring makes no judge call).
6. Run `cohr eval --set quick` and record support rate, the judge's cost and peak memory. The
   cross-language items run in `full` (Phase 6): cover judge recall with unit tests only.
Running `cohr eval --set quick` is fine. Ask me before anything in `full`, before `pytest -m
models`, and before new model downloads. Don't add --judge claude (needs an API key and a new
dependency). No new dependencies are expected; if one is needed, check its license (§11) and ask.
When done: tick the checklist, update §20 and NOTES.md, make sure the gate passes, and open a PR
(ask before pushing). Stop there; Phase 5 (UI) is a separate session.
```

### Kickoff prompt for Phase 5a (UI without Ask)
```text
Continue cohr: Phase 5a (Streamlit UI without Ask). Read CLAUDE.md, then PLAN.md sections §9
(the 5a bullet), §7 (all of it: UI, layout, theme, tests), §12 (sources, upload limit, local
path, prune/delete, disk checks), §4 (memory policy, GPU limit), §5.6 (in-app render mode,
anchors, refined t), §11 (footer disclaimer) and §20 (D13, D14). Then the latest NOTES.md entry.
Load the developing-with-streamlit skill before writing any UI code. Work on branch
phase-5a-ui. Follow the 5a acceptance criteria in §9:
1. Theme first: .streamlit/config.toml exactly as in §7 (server address localhost, upload limit,
   gatherUsageStats=false, [theme], [theme.light], [theme.dark] and their sidebars) and
   tests/test_theme.py (light/dark key parity, contrast ratios, no Google Fonts, no telemetry).
2. Job layer without Streamlit: src/cohr/jobs.py runs pipeline.run in a background thread with
   a progress queue, a cancel Event and one job at a time (the ModelManager lock); the UI only
   reads its state. On app start, pipeline.recover_interrupted for every run. Add
   pipeline.run(force=[stage names]) so a stage can re-run on demand (later stages follow
   through the input-hash chain); "Re-reduce" uses it. Unit tests with fake stages, no models.
3. app.py: per-page set_page_config layouts as in §7, st.navigation with New run, Notes and
   Runs (Ask is 5b), the sidebar run picker (the current run in session_state), the model and
   memory status line with the GPU limit warning, and the §11 footer. ModelManager through
   st.cache_resource. Poll a running job with st.fragment(run_every=…), not full-page reruns.
4. Pages per §7: New run (st.form; upload or local path; languages side by side; Advanced
   expander with window length, memory policy, source mode; a clear message when disk space or
   models are missing; one st.status per stage; Cancel). Notes (render.py in app mode; the
   Sources panel with one block per segment: anchor, time range, st.audio(start_time,
   end_time), slide images with alt text; unaligned slides last; download the portable
   notes.md; Re-reduce; uncited caption; semantic badge colors). Runs (st.dataframe with
   single-row selection, size, date, status; Resume for interrupted runs; re-run stage; Prune
   and Delete confirmed in st.dialog; the peak-memory-per-stage bar chart).
5. tests/test_app.py with AppTest, a fake ModelManager and a fixture run dir (no models): every
   page renders without exceptions, Notes has one Sources block per segment, Runs lists the
   run, New run refuses a missing local path.
6. Manual checks, then record them: start the app (uv run streamlit run app.py) and do one real
   run through the UI on evals/data/gen/clean-en (progress, Cancel, then Resume). Ask me before
   using Claude in Chrome; with it, take screenshots at ~420, 800, 1280, 1512 and 2560 px in
   light and dark. D13: does the 880 px notes container shrink below 880 px (else
   width="stretch")? D14: does a citation anchor scroll the Sources panel (else the drawer
   fallback from §7)? Record both in §20.
Rules from CLAUDE.md apply: theme only via config.toml, no custom CSS or JS components, pages in
app_pages/, never use_container_width, no telemetry. No new dependencies are expected; if one is
needed, check its license (§11) and ask. Ask me before the opt-in model tests (the `models`
pytest marker), before anything in `cohr eval --set full`, and before new model downloads.
When done: tick the checklist, update §20 and NOTES.md, make sure the gate passes, and open a PR
(ask before pushing). Stop there; Phase 5b (Ask) is a separate session.
```

### Kickoff prompt for Phase 5b (Ask)
```text
Continue cohr: Phase 5b (Ask). Read CLAUDE.md, then PLAN.md sections §9 (the 5b bullet),
§5.7 (Ask), §5.6 (rag documents, citation parsing, time refinement, uncited rule), §7 (Ask
page, drawer, job lifecycle: Ask waits while a job holds the models), §4 (32K cap) and §11.
Then the latest NOTES.md entry (Phase 5a). Load the developing-with-streamlit skill before
writing UI code. Work on branch phase-5b-ask. Follow the 5b acceptance criteria in §9:
1. src/cohr/ask.py, Streamlit-free, reusing reduce.py rather than copying it: the same rag
   documents (reduce.documents / fit_prompt), the same citation parsing and dropped-index rule,
   and the same time refinement (segment_refiner). CommandR7B gets a chat variant of prompt()
   that takes the message list (history + question); prompt() keeps its current output, so
   Reduce's prompt and config hash don't change. History = the last 3 question/answer turns,
   answers with citation markup removed, dropped oldest-first when the prompt exceeds the run's
   reduce.max_input_tokens; if the documents alone don't fit, trim transcripts like Reduce
   (FTS5 ranking is Phase 6). Answer language: lingua on the question (minimum relative
   confidence chosen and recorded in NOTES.md), else the run's target language. Uncited answer
   sentences are kept and flagged, never given citations. Return a small dataclass (answer
   claims with citations, language, prompt tokens, turns dropped) that the page renders.
2. prompts/ask.md (no prompt text in code): answer only from the documents, in {language},
   short prose, say so when the documents don't answer it.
3. Ask is not a pipeline stage and writes no run file: history lives in st.session_state per
   run, so no schema change. It runs through ModelManager.use("reduce") under the same lock.
   While a job runs (JobRunner.busy()), the chat input is disabled with a caption saying why,
   instead of blocking the script thread on the lock; record this reading of §7 in NOTES.md.
   Apply the run's config to the manager first (set_config), as jobs do.
4. app_pages/ask.py per §7 (layout "centered"): st.chat_message history, st.chat_input, the
   answer rendered with render's app-mode citation markup, uncited sentences gray with the
   badge, st.pills of the cited sources under each answer, and a right-side
   st.dialog(position="right") drawer for the selected source: time range, st.audio from the
   cited t (ui.audio_clip), slide images. Build the source block once in ui.py and use it in
   both the drawer and Notes' Sources panel. Add "Open in Ask" to the Notes header and Ask to
   st.navigation. "Answer based on the N most relevant segments" is Phase 6 (with ranking).
   Also on Notes (D21, §7): a "Sources" st.segmented_control in the header row, Beside
   (default, current columns) or Below (notes in the 880 px container, then the source blocks
   without the fixed-height box), kept in st.session_state.
5. Tests without models: tests/test_ask.py (history trimming oldest-first under a fake token
   counter, citation markup stripped from history, language fallback when lingua is unsure,
   citations in answers parsed and refined, dropped indices, uncited sentences kept) with a fake
   R7B; tests/test_app.py additions (Ask renders, pills for cited sources, input disabled while
   a job runs, Open in Ask switches page with the run, Notes in Below mode renders every
   source block). Reduce's config hash must be unchanged (test it).
6. Manual checks, then record them: on the clean-en run, ask 3 questions whose answers are in
   evals/gen truth.json (one numeric), one follow-up that needs history ("who owns that?"),
   one German question (answer in German), and one the meeting doesn't answer. Check that the
   citations jump to the right moment. Also press play in the Sources panel and the drawer in a
   normal browser window (5a couldn't verify audio). At ~800 px with the sidebar open, switch
   Notes to Below and check that citations still jump to their block. Ask me before using
   Claude in Chrome.
Rules from CLAUDE.md apply. Ask me before the opt-in model tests (the `models` pytest
marker), anything in `cohr eval --set full`, new model downloads or new dependencies.
When done: tick the checklist, update §20 if anything was decided, add a NOTES.md entry, make
sure the gate passes, and open a PR (ask before pushing). Stop there; Phase 6 is separate.
```

### Kickoff prompt for Phase 6a (baseline + Ask eval)
```text
Continue cohr: Phase 6a (baseline + Ask eval). Read CLAUDE.md, then PLAN.md §9 (the 6a
bullet), §18 (metrics, Ask row, reports and the release gate), §20 (D16), §5.7 (Ask, known
limits) and the latest NOTES.md entry (Phase 5b). Work on branch phase-6a-baseline.
This phase measures; it changes no prompt and no pipeline behavior.
1. Ask eval (src/cohr/evals.py, no new dependency): per quick item, questions built from
   truth.json with templates (no LLM): one per number fact ("What is the <label>?"), one
   per action ("What does <owner> need to do?"), one follow-up after an action ("Who owns
   that, and by when?"), two German questions (one alone, one after English history), and
   one near-miss per item (a number's label with another period or team). Score: correct =
   the answer states the fact (numbers and owners exact, task via token overlap ≥ 0.6, as in
   fact recall) and cites its segment within ±15 s (D19) or its slide; near-miss = correct
   when no claim states a planted value as the answer (define the exact rule and record it
   in NOTES.md before the first run). Report Ask accuracy, Ask citation accuracy, follow-up,
   cross-language (alone / after history) and near-miss rates. Questions go through ask.ask
   with the run's config; cache answers per item like the judge (keyed by question, ask
   prompt hashes and the R7B revision), so re-scoring loads no model.
2. D16: run quick 3 times with the same media and pipeline seeds 1, 2, 3 (-c seed=N;
   --seed changes the generated meetings, not the noise). Per metric, noise = max − min;
   threshold = max(PLAN default, 2 × noise), Ask metrics included. Record in §20 and NOTES.
   Ask me before deciding whether a noisy metric stays report-only.
3. Commit evals/baseline.json from the seed-1 run with the reason in the commit message,
   and make compare() use the D16 thresholds. Count open questions in made-up items only
   if D16 shows it's measurable (the 4b-2 finding is fixed in 6b, not here).
4. Tests without models: question templates (deterministic, every kind covered), Ask scoring
   (right fact + right citation, wrong owner, near-miss pass and fail, German), the cache key,
   and compare() with D16 thresholds.
Rules from CLAUDE.md apply. Ask me before `cohr eval --set full`, the opt-in model tests,
new model downloads or new dependencies. quick runs are fine (~5 min each plus Ask).
When done: tick the checklist, fill D16 in §20, add a NOTES.md entry, make sure the gate
passes, and open a PR (ask before pushing). Stop there; 6b is separate.
```

### Kickoff prompt for Phase 6b (quality fixes)
```text
Continue cohr: Phase 6b (quality fixes). Read CLAUDE.md, then PLAN.md §9 (the 6b bullet),
§5.5–5.7 (Map, Reduce, Ask), §18 (gate, Ask row), §20 (D16) and the latest NOTES.md entry
(Phase 6a). Work on branch phase-6b-quality.
Every eval in this phase runs with `-c seed=1` (the baseline's seed) and is compared with
evals/baseline.json. While iterating, use --item and --no-judge; run full quick (with Ask and
the judge, ~15 min) before each baseline update. Commit each step separately; a baseline
update goes in its own commit with the reason and the before/after numbers.
1. Measurement first (no prompt changes yet):
   a. Planted open questions: add an `open_question` fact kind to evals/gen (1–2 per item,
      spoken as a clearly unresolved question, e.g. "One thing we haven't settled: <q>?";
      the filler "seasonality" lines stay). Score open-question recall (Open questions
      section, token overlap ≥ 0.6) and made-up open questions (claims in that section
      matching no planted one). Regenerated media change every metric, so re-baseline right
      after with the reason "generator: planted open questions", before any fix.
   b. Near-miss: ask all three NEAR_MISS templates per item (12 per set instead of 4).
   Ask me whether made-up open questions and near-miss enter the gate; D16's noise for them
   was measured before this change.
2. Ask fixes (ask.md / ask.py only; Reduce's hash must not move: test_ask pins it):
   near-miss (R7B answers with a related number despite the ask.md rule; try e.g. a worked
   example in the prompt or asking it to name the period/team first). Pass: near-miss up, no
   Ask metric down beyond its D16 threshold; check German-after-history by eye (report-only).
3. Reduce: filler → open questions (13 / 11 / 4 over 6a's seeds). Fix in reduce.md (e.g.
   "only questions someone asked and left open"). Pass: made-up open questions down,
   open-question recall and every gated metric within D16. This moves Reduce's config hash:
   update the test_ask pin on purpose, with the reason in the commit.
4. Find the cause before changing anything, record it in NOTES.md, and fix only if it's clear:
   a. Reduce dropping facts Map has right (4b-1b: 4650 signups on clean-en, the $57,000
      budget): in map.json? in the prompt? cut by the fit? left out by R7B?
   b. The slide-conflict value: on slide-conflict-en both the notes and Ask give the
      misspoken $80,000, not the slide's $18,000 (baseline: slide_value_used false), despite
      the slide rule in reduce.md and ask.md. Which value does map.json carry, and does the
      slide document reach R7B with the right number?
   Pass: recall_number up / slide_value_used true, nothing gated worse.
If a fix doesn't pass, revert it and record what was tried (as in 5b), and move on.
Rules from CLAUDE.md apply. Ask me before `cohr eval --set full`, the opt-in model tests, new
model downloads or new dependencies. quick runs are fine.
When done: tick the checklist, record decisions in §20 (new D numbers), add a NOTES.md entry,
make sure the gate passes, and open a PR (ask before pushing). Stop there; 6c is separate.
```

### Kickoff prompt for Phase 6c (long meetings)
```text
Continue cohr: Phase 6c (long meetings). Read CLAUDE.md, then PLAN.md §4 (two-pass reduce),
§5.6–5.7 (Reduce, Ask, long meetings), §9 (the 6c bullet), §18 (gate, long-en), §20 (D16, D17,
D24–D27) and the latest NOTES.md entry (Phase 6b). Work on branch phase-6c-long.
Quick evals run with `-c seed=1` against evals/baseline.json (iterate with --item, --no-judge,
--no-ask; full quick with Ask + judge, ~12 min, before each baseline update). Commit each step
separately; a baseline update goes in its own commit with the reason and before/after numbers.
A lowered cap (`-c reduce.max_input_tokens=N`) forces two-pass Reduce and Ask ranking on the
5-minute quick items: use it as the fast loop, and long-en only to confirm.
0. From 6b, before any long-meeting work (citation accuracy is how two-pass is judged):
   a. Reduce bundling: on the 6b media Reduce writes run-on key points with 3–4 facts and one
      citation ("Support Queue: 390 open tickets … New logo … Laura to send contract …"), so
      the earlier facts land 19–35 s off. Fix in reduce.md (e.g. one fact per item). Pass:
      citation accuracy up or equal, fact recall and every gated metric within their gate.
      This moves Reduce's hash: update the test_ask pin on purpose, with the reason.
   b. D25: run seeds 2 and 3 on the result (seed 1 is the baseline run) and show me the
      citation-accuracy noise with a threshold by D16's rule; I decide before it goes into
      MAX_DROP.
1. Two-pass reduce (§4, D26; reduce.py + prompts/reduce_pass2.md):
   - Over `max_input_tokens`: pass 1 = a normal cited reduce per group of consecutive segments
     (with the slides aligned to them); its citations are written into the group summary as
     inline markers [seg-N] / [slide-N]. Pass 2 cites group summaries; each pass-2 citation
     resolves to the markers inside the cited span, falling back to every segment of that group
     when the span has none (coarse but valid, never invented). Time refinement runs after
     resolution, against the resolved segments (§5.6). NotesDoc.passes = 2; anything new on
     disk goes through schemas.py (FORMAT_VERSION bump + snapshots if NotesDoc changes).
   - Under the cap nothing changes: same notes for a single-pass run (the pin may move only by
     the prompt list; say so in the commit).
   - Unit tests (no models): grouping under the cap, marker writing and resolution (markers in
     span, none in span → group fallback, out-of-range dropped and logged), citations survive
     both passes (CLAUDE.md rule), uncited claims stay uncited.
   - Check on quick with a cap forcing 2–3 groups vs single pass: citation accuracy within D25,
     no uncited rise, every citation resolves; record the fact-recall cost and pick the group
     size (D26).
2. Ask ranking (§5.7, D17, D27; ask.py, pure ranking function):
   - D17 first: `PRAGMA compile_options` in uv's Python shows FTS5 (stdlib sqlite3, no new
     dependency). If it's missing, stop and ask me.
   - Over the cap: rank segment and slide documents for the question with an in-memory FTS5
     table (BM25; `trigram` tokenizer for zh/ja/ko questions), send the top-ranked ones that fit
     with the history, in meeting order, citation indices mapped as today. The Answer records how
     many were sent; the Ask page says "Answer based on the N most relevant segments".
   - Unit tests: ranking order, fit under the cap with history, index mapping, CJK tokenizer.
   - Check on quick with a lowered cap vs the full-document baseline (Ask accuracy, citation,
     near-miss within their gates).
3. long-en (§18): a `long` set (evals/sets/long.yaml), one item, ~2.5 h, audio + PDF deck
   (a 2.5 h screen recording is large; vision is the slowest stage). Sections repeat the quick
   structure with new facts: every fact stays distinguishable (decision and task texts unique
   after `content()`, numbers distinct and > 31; widen word lists or ranges as needed), and
   quick scripts stay identical (test: same truth for the quick specs). Any evals/gen change
   regenerates quick media: if quick metrics move, re-baseline with the reason. The Ask eval
   on long-en asks a fixed, deterministic sample (e.g. 12 number, 6 action, the follow-up,
   2 German, 3 near-miss), not every fact. One long run is ~40 min of pipeline plus generation
   and Ask: tell me before each, at most three. Commit its report as evals/baseline-long.json,
   report-only (one run: no noise measured).
If a fix doesn't pass, revert it and record what was tried, and move on.
Rules from CLAUDE.md apply. Ask me before `cohr eval --set full`, the opt-in model tests, new
model downloads or new dependencies. quick runs are fine.
When done: tick the checklist, fill D17 and D25–D27 in §20, add a NOTES.md entry, make sure the
gate passes, and open a PR (ask before pushing). Stop there; 6b-2 (Map fixes) is separate.
```

### Kickoff prompt for Phase 6b-2 (Map fixes)
```text
Continue cohr: Phase 6b-2 (Map fixes). Read CLAUDE.md, then PLAN.md §5.5 (Map), §9 (the 6b-2
bullet), §18 (gate, long-en), §20 (D12, D16, D20, D24–D26) and the latest NOTES.md entries
(Phase 6b and 6c). Work on branch phase-6b-2-map.
Quick evals run with `-c seed=1` against evals/baseline.json (iterate with --item, --no-judge,
--no-ask; full quick with Ask + judge, ~12 min, before each baseline update). Map runs at
temperature 0.3 with the pipeline seed, and a Map change re-runs Map and Reduce and re-asks
every Ask question. Commit each step separately; every kept Map change gets its own baseline
commit with the reason and before/after numbers (map.md moves every metric).
0. Measure first, no fixes. For each quick item (seed 1), trace every planted fact:
   transcript (ASR right?) → Map segment notes (which list; bundled with what) → notes.json
   (section, citation). Count Map key points per segment and facts per key point, and check
   whether any Map reply hit map.max_tokens (1024) or fell back to raw. Put the table in
   NOTES.md; it decides the order of steps 1–4.
1. Bundled key points (6c: "Support Queue: 390 open tickets … Laura to send contract to legal
   by Friday." is one Map key point, and Reduce copies it word for word, so the citation lands
   on the bundle): in map.md, one fact per key point (a figure with what it measures), and
   decisions and tasks only in their own lists. Pass: citation accuracy up or equal, fact
   recall and every gated metric within their gate.
2. Dense segments (6b: the ~280 s first segment with 7 topics keeps ~7 key points and drops
   decisions and actions): follow step 0. If replies are cut at max_tokens, raise
   map.max_tokens (a config change: say so in the commit). If not, try the prompt first (list
   every decision and task said in this part). A smaller fuse.window_s was worse in D12; a
   second Map pass per topic is a last resort: ask me first. Pass: recall up, no gated metric
   outside its gate.
3. Due-date misread ("by the 22nd of November" → "November 2nd"): prompt first. If it stays,
   extend the number guard (§5.5, D20) to dues: a due whose day number is in no source is
   repaired when exactly one spoken day is one digit edit away, else flagged (pure Python,
   unit tested; a MapFile change bumps its FORMAT_VERSION and regenerates the snapshots).
4. Slide-conflict value (6b: Map writes the misspoken $80,000 although the slide row
   "Approved budget | $18,000" is in its input): prompt first (when a slide table row names the
   figure being spoken about, use the slide's value). A pure-Python guard that changes values
   (a Map number matching the spoken value while the slide row with the same label has
   another) only after asking me.
5. long-en: Map changes change it. Tell me before each long run, at most two. First, one full
   run: does pass 2 still drop about a third of pass 1's decisions (6c: Map 19/20 → pass 1 15 →
   pass 2 10)? If so, try reduce_pass2.md with Reduce-only reruns (--no-ask, tell me before
   each; they don't count toward the two). Then commit the final long report as
   evals/baseline-long.json (report-only) with the reason.
If a fix doesn't pass, revert it and record what was tried, and move on.
Rules from CLAUDE.md apply. Ask me before `cohr eval --set full`, the opt-in model tests, new
model downloads or new dependencies. quick runs are fine.
When done: tick the checklist, add any new decision to §20, add a NOTES.md entry, make sure
the gate passes, and open a PR (ask before pushing). Stop there; 6d (languages) is separate.
```

### Kickoff prompt for Phase 6d-1 (notes in other languages)
```text
Continue cohr: Phase 6d-1 (notes in other languages). Read CLAUDE.md, then PLAN.md §5.5 (Map,
number guard, slide conflicts), §5.6 (fixed section markers, render labels, "Other (beta)"),
§7 (theme, no custom CSS), §9 (the 6d bullet), §18 (cross-language matching, output language),
§20 (D15, D16, D20, D23, D28) and the latest NOTES.md entries (Phase 6b-2). Work on branch
phase-6d-1-langs.
Notes in a language other than the meeting's have only been checked by hand (Phase 0) and by
Ask's German questions (6a). The cross-language scoring (exact fields + judge recall, 4b-1/4b-2)
has unit tests but has never run on a real item. Commit each step separately.
0. Measure first, no fixes. Add a `lang` set (evals/sets/lang.yaml, report-only baseline
   evals/baseline-lang.json): `conflict-de` (the slide-conflict-en script and media, target
   de, so the "(slide: …)" note shows up in German) and `clean-ar` (the clean-en script and
   media, target ar). Same seed and settings as the quick items, so the generator rebuilds
   the same script and media under the new id; only the target changes. Run it with
   `-c seed=1 --no-ask` (judge on: it scores the cross-language text facts). Record in
   NOTES.md: output language, exact-field and judge recall, citation accuracy, made-up items,
   number-guard repairs/flags, and by hand: do the five English section markers and the
   `[owner: …] [due: …]` lines parse; which digits and separators Map and R7B write (German
   "42.500"? Arabic-Indic digits "٤٢٬٥٠٠"?); how the slide-conflict note reads. Then trace
   each missed fact (scripts/trace_facts.py) as in 6b-2: scoring miss or real miss?
1. Scoring that reads the notes language. evals.py reads numbers with numbers_in (English
   only), so German "42.500" is 42.5 and a correct fact scores as missed. Read the claim's
   numbers the way the target language writes them (numbers.DECIMAL_MARK / native_readings,
   the guard's rule since 6b-2), and Arabic-Indic digits and separators if step 0 shows R7B
   writes them (pure Python, unit tested; the guard and the eval share numbers.py). Pass:
   step 0's scoring misses become matches; quick unchanged.
2. The slide-conflict note in the target language: " (slide: $18,000)" is English inside
   German notes (6b-2). Take the word from one static per-language table (like render's
   LABELS; English for languages not in it), and make sure Reduce and the eval still see both
   values. A change to Map strings re-runs Map: say so in the commit. Pass: the note is in
   German on conflict-de, slide_value_used still true there and on quick.
3. Arabic in the Notes page and notes.md. Check how Streamlit shows right-to-left notes with
   citations and owner/due (no custom CSS: §7, CLAUDE.md). If they read wrong, try what the
   rules allow (Unicode direction marks in render.py's output) and record the result as a new
   §20 decision. Ask me to do the browser check myself, or whether to use Claude in Chrome.
4. "Other (beta)" target languages (§5.6): any language Tiny Aya supports beyond R7B's 23
   (list from the Tiny Aya model card, recorded in config.py with its source). Reduce runs in
   English; Tiny Aya translates the notes section by section. Only the text between citation
   markers is translated, and the markers go back in the same order: pure-Python split/join,
   unit tested, with a claim holding two or more spans, the fixed section markers and the
   `[owner: …] [due: …]` labels kept in English. Prompt in src/cohr/prompts/ (target language
   repeated on its last line, as in map.md). Propose where translation runs (its own stage
   between reduce and render, or inside reduce) and any NotesDoc change before coding: a
   schema change bumps FORMAT_VERSION and regenerates snapshots. The New run page gets the
   option. Eval: one lang item with an Other (beta) target that lingua detects and Tiny Aya
   supports (e.g. Swahili); pass = output language right, every claim's citations identical
   to the English notes', exact-field recall within the D16 noise of the English run.
Not in 6d-1: the xling-{es,ja} items and regional Map models (D18) are 6d-2; Ask in the lang
items (the 6a German questions cover answer language).
Rules from CLAUDE.md apply. Ask me before `cohr eval --set full`, the opt-in model tests, new
model downloads or new dependencies. quick and lang runs are fine; quick must pass its gate at
the end (steps 1–2 touch the guard's numbers code and Map strings).
When done: tick the checklist, add any new decision to §20, add a NOTES.md entry, make sure
the gate passes, and open a PR (ask before pushing). Stop there; 6d-2 is separate.
```

### Kickoff prompt for Phase 6d-2 (other source languages, regional Map models)
```text
Continue cohr: Phase 6d-2 (other source languages and regional Map models). Read CLAUDE.md,
then PLAN.md §5.2 (Transcribe, user-picked language), §5.5 (Map, number guard, regional models),
§9 (the 6d bullets and 6d-1 Results), §18 (variants, cross-language matching, WER/CER),
§20 (D8, D16, D18, D20, D29, D30) and the latest NOTES.md entry (Phase 6d-1). Work on branch
phase-6d-2-xling.
6d-1 made notes in another language work when the meeting is English. Nothing has run on a
meeting in another language yet: the generator writes English only, and the number guard and
the eval read number words in English only (numbers.py), so a Spanish or Japanese transcript
that spells numbers out may give the guard no source numbers. Commit each step separately.
0. Generator in es and ja, then measure, no fixes. Add the script language to evals/gen
   (ItemSpec.language is already there): per-language tables for openings, topic intros,
   discussion and filler, decision/task/open-question lines, weekdays and relative dues, units
   ("por ciento", "パーセント"), and voices (es: Mónica + Paulina + one es_MX/es_ES voice;
   ja: Kyoko + two ja_JP voices, D8). Owners stay the Latin names, as people's names do in
   real meetings; say how the ja voices read them. The English scripts must not change
   (QUICK_SCRIPTS fingerprints) and the planted texts stay distinguishable per language (the
   test that checks it, per language). New items in the `lang` set: `xling-es` and `xling-ja`
   (target en, seed 1, video + slides, slides in the meeting's language). Run them with
   `-c seed=1 --no-ask`, judge on. Record in NOTES.md: WER (es) / CER (ja, jiwer), whether
   Transcribe writes numbers as digits or words, Map's source numbers and guard repairs/flags,
   exact-field and judge recall, citation accuracy, made-up items, output language, and how
   owners come out (ASR "トム" vs notes "Tom"). Trace each miss (scripts/trace_facts.py; its
   text match is English, so read Map by hand as in 6d-1 step 0): ASR, Map, Reduce or scoring?
1. Fix what step 0 shows, in pure Python where it's a value (CLAUDE.md lessons): most likely
   number words for es/ja in numbers.py (shared by the guard and the eval; "cuarenta y dos mil
   quinientos", "4万2500"), unit tested, with a test that English is unchanged. Pass: step 0's
   number misses that are scoring or source-number misses become matches; quick unchanged.
   Prompt changes to Map are judged on seeds 1–3 (CLAUDE.md), on quick too.
2. Regional Map models (§5.5, D18), only after asking me: the CohereLabs originals are gated,
   need the license accepted on the Hub and an HF token (HF_TOKEN or .env, never committed),
   and each is a multi-GB download. Ask me which variants to try first (Fire has an MLX repo;
   Earth and Water need `mlx_lm.convert` to 8-bit, locally) and whether to keep the converted
   weights. Add `map_model_by_lang` to config.py (empty by default; the repo + revision enter
   Map's config hash, and "Other (beta)" translation uses the same model). Score Global vs the
   variant on 6d-1's en → X items for 2 languages of the variant's region (add lang items as
   needed, e.g. en → hi for Fire, en → sw for Earth), seeds 1–3 each, with cross-language
   matching: adopt per language only if exact-field recall improves by ≥ 2× the D16 noise and
   output-language accuracy is no worse. Record D18's result (it may stay empty).
Optional if time allows, after asking: the 6d-1 carry-overs (Arabic owners written in Arabic
script, which the eval misses; "Other (beta)" invented summary).
Not in 6d-2: Ask on the xling items (report-only at most), the full set and v1 (6e).
Rules from CLAUDE.md apply. Ask me before the full eval set, the opt-in model tests, any model
download or HF token use, and new dependencies. quick and lang runs are fine; quick must pass
its gate at the end (step 1 touches numbers.py, which the guard uses).
When done: update evals/baseline-lang.json (say why in the commit), tick the checklist, add
any new decision to §20, add a NOTES.md entry, make sure the gate passes, and open a PR (ask
before pushing). Stop there; 6e is separate.
```

### Kickoff prompt for Phase 6e-1 (full set + robustness, measured)
```text
Continue cohr: Phase 6e-1 (full set and robustness, measured). Read CLAUDE.md, then PLAN.md
§4 (memory policy), §9 (the 6e bullets, the definition of done, and the carry-overs in the
6b-2, 6c, 6d-1 and 6d-2 Results), §11 (licensing), §12 (storage), §18 (eval sets, FLEURS and
AMI, AMI join and scoring, reports and the release gate), §20 (D15, D16) and the latest
NOTES.md entry (Phase 6d-2). Work on branch phase-6e-1-full.
quick meets the v1 targets; 6e-1 builds what v1 still lacks and measures it. No quality
fixes in 6e-1: carry-overs go to 6e-2, ranked by what the first full run shows. Commit each
step separately.
0. Proposal before code (ask me): what `full` contains and how long it runs. §18 says
   "all variants + 20 FLEURS clips × 14 languages + 5 AMI meetings, ~1 h", but long-en alone
   is ~40 min; say whether long-en stays in its own set, and estimate the run time and the
   download sizes (FLEURS test audio per language, AMI ihm parquet). External data is
   fetched at eval time into evals/data/ (gitignored, never committed); ask before any
   download. Check each dataset's license on its card again (CC BY 4.0, §18) and the AMI
   annotations zip's LICENCE.txt.
1. FLEURS (Transcribe only): 20 test clips per language, fixed by a seed in
   evals/sets/full.yaml; WER, or CER for ja/ko/zh (`asr_error`, 6d-2); per-language rows in
   the report. FLEURS clips are single sentences: score Transcribe on the clip, no VAD.
2. AMI: the 5 meetings fixed in full.yaml (in both the audio and the official annotations);
   audio only (no slides), English notes. The judge scores coverage per reference sentence
   (§18 mapping: decisions → Decisions, actions → Action items, problems → Open questions,
   abstract → Summary); parse the XML with stdlib `xml.etree` (ISO-8859-1). Seconds per audio
   hour per stage on these real meetings is the definition of done's speed check (≤ 20 min
   per audio hour, `resident`); peak memory too.
3. The report footer for external data (§18, exact text) on every report that includes it,
   and the same line in evals/README.md. `full` gates against evals/baseline-full.json
   (GATED_SETS has `full` already); the first run writes it.
4. Robustness: missing ffmpeg and missing or partly downloaded models give a clear message in
   the CLI and the UI (both exist in part: ingest.require, ModelsMissing; check every path
   that can hit them, e.g. a model deleted from the cache after a run, HF offline), with
   tests. An end-to-end run of a quick item with `memory_policy=sequential`: notes identical
   in content to `resident` on the same seed, and the peak memory per stage (§4) recorded.
5. README: the status line is from Phase 1. Bring it up to date: what works, the app
   (`uv run streamlit run app.py`), `cohr models download` and the model cache size (~15 GB,
   measured from the cache), deleting models with `hf cache rm model/<repo>`, the eval sets in
   one line each, and the CC-BY-NC note that's there. huggingface_hub 1.33 has no
   `hf cache delete` (its cache commands are list, prune, rm, verify). Warn against
   `hf cache prune`: it removes revisions that no ref points to, and cohr's pinned revisions
   are such revisions (its dry run listed tiny-aya-global-8bit-mlx). Say to run `--dry-run`
   first with either command.
6. The first full run (ask me first: it's slow and downloads data). A measurement: no fixes.
   Record in NOTES.md: FLEURS WER/CER per language, AMI coverage per section and support rate,
   speed and peak memory, and the synthetic items against their quick/lang baselines. Then
   rank the carry-overs (PLAN §9 6b-2 to 6d-2 Results: Japanese chunk-start drops, owners in
   Devanagari/Arabic script, Map's 万 misreads and dense-segment drops, long-en tasks filed as
   decisions, Ask near-miss under the cap, Ask and slide conflicts, the "Other (beta)" limits)
   by what the full run shows, as the input for the 6e-2 kickoff.
Rules from CLAUDE.md apply. Ask me before the full run, any dataset or model download, the
opt-in model tests, and new dependencies (check licenses, §11). quick and lang runs are fine;
quick must pass its gate at the end. Don't bump the version: that's 6e-2, and only when I ask.
When done: write evals/baseline-full.json (say why in the commit), tick the checklist, add
any new decision to §20, add a NOTES.md entry, make sure the gate passes, and open a PR (ask
before pushing). Stop there; 6e-2 is separate.
Step 0 is done (PR #36, 40cb426 on main; NOTES.md "6e-1 · 0. Proposal"): start at step 1
on a new phase-6e-1-full branch from main. Approved downloads: AMI audio (5 meetings, official mirror) and FLEURS (streamed);
the AMI annotations zip is already in evals/data/ami/.
```

### Kickoff prompt for Phase 6e-2 (carry-over fixes + v1)
```text
Continue cohr: Phase 6e-2 (carry-over fixes, then the v1 release gate). Read CLAUDE.md, then
PLAN.md §5.5–5.6 (Map, Reduce), §9 (the 6e bullets and the definition of done), §16
(releases), §18 (eval sets, the release gate, AMI scoring), §20 (D16, D20, D24, D25, D31) and
the latest NOTES.md entry (Phase 6e-1), above all "Carry-overs ranked by the full run".
Work on branch phase-6e-2-v1 from main (6e-1 is PR #39, 80002bd).
Every v1 target is met on the quick items; 6e-2 fixes what the first full run showed on real
meetings, in the ranked order, and measures each fix on every set it can reach (CLAUDE.md:
quick results don't carry over). Commit each step separately; record each in NOTES.md.
0. Proposal before code (ask me): for carry-overs 1–4, the fix you'd make, where (prompt or
   Python, by CLAUDE.md's rule), the sets that measure it, and the runs it costs. Say which
   of 5–7 (Japanese, eval gaps, Ask near-miss) you'd leave for after v1, and why.
1. Reduce's output cap (carry-over 1). Detect a reply that stops at max_tokens (mlx-lm's
   finish reason or the reply's token count; today the notes look complete), then retry with
   a larger budget or flag the notes (a warning in the run log and the UI, never invented
   text). Set the single-pass default from measurement: 4096 was measured on AMI (coverage
   44.9 → 52.8%, longest reply 3470 tokens at 42.8 min), so check the margin for a 60 min
   meeting. Tests for the detection, including one where it must not fire (a reply that ends
   with <|END_RESPONSE|> just under the cap). Changing the default moves REDUCE_HASH.
2. Tasks and considerations filed as decisions (carry-over 2): AMI ES2004b's 15 "decisions"
   ("The project will explore/consider…"), long-en since 6c, quick decision recall 75%.
   6b-2 points to Map (a "tasks aren't decisions" rule in Reduce's long-meeting prompts was
   ignored); trace a few facts first (scripts/trace_facts.py). A Map prompt change is judged
   on the mean of seeds 1–3 (CLAUDE.md), then on long-en and AMI.
3. Owners (carry-over 3): R7B writes owners in Devanagari/Arabic script (clean-hi recall
   53.8%, made-up 5). First find how much is the eval's owner match and how much the notes;
   fix the notes in Python if the names should stay in Latin script, the eval if not. AMI's
   role owners ("The Marketing Expert") are measured, not fixed, unless step 0 argues it.
4. Numbers (carry-over 4): run `unsupported_numbers` on AMI notes (report-only), and the
   Map 万 misreads ("35万4000" → "$34,000") in the number guard (Python, D20's rule).
   Transcribe's own number errors (FLEURS ar, nl) are recorded, not fixed.
5. Release gate (§16, §18): full against evals/baseline-full.json (D31: the quick items gate)
   and long against evals/baseline-long.json. Rewrite each baseline a fix moves on purpose,
   with the reason in the commit (quick, lang, long, full). Attach the full report summary to
   the PR. Don't bump the version: I'll ask for 1.0.0 separately.
Rules from CLAUDE.md apply. Ask me before every full or long run, the opt-in model tests and
new dependencies. quick and lang runs are fine; quick must pass its gate at each step.
When done: tick the checklist, add new decisions to §20, add a NOTES.md entry with the
post-v1 list (what stayed open and why), make sure the gate passes, and open a PR (ask
before pushing).
```

### Kickoff prompt for Phase 7 (cross-meeting search)
```text
Continue cohr: Phase 7 (cross-meeting search, after v1). Read CLAUDE.md, then PLAN.md §13
(search index), §7 (pages, the Search page and layout rules), §12 (storage: runs, Delete,
Prune media), §5.7 and D17/D27 in §20 (Ask's FTS5 ranking and the trigram windows for
zh/ja/ko), §8 and §19, and the latest NOTES.md entry (Phase 6e-2, its post-v1 list).
Work on branch phase-7-search from main (v1.0.0 is released: #42, #43, tag v1.0.0).
Phase 7 runs no models: the index is built from files already on disk, so there are no eval
runs to ask about. The 6e-2 post-v1 list stays open; don't fix it here unless step 0 argues
that search depends on one of its items.
0. Proposal before code (ask me). §13's schema is a sketch from planning, so check it against
   today's files and code first:
   - What goes in the index: segment transcripts, slides, notes claims (`render.shown_notes`,
     so an "Other (beta)" run is searched in the language people read) and action items with
     owner and due. Say whether citations are stored so a hit on a claim opens its [mm:ss].
   - Tokenizer: §13 says unicode61; Ask (D17) found trigram works for ja, with 3-character
     windows (`ask.query_terms`). Propose how zh/ja/ko text is indexed and queried (a second
     FTS5 table, or trigram for those runs), and reuse Ask's term code instead of a copy.
   - Sync: who re-indexes after a stage finishes, Delete and Prune media (pipeline, jobs, the
     storage functions or the pages), keyed by `manifest_hash`. Only runs in `runs_dir()`:
     eval runs under evals/data/runs/ are never indexed by the app.
   - Opening a result: how the Search page selects the run and jumps to the segment in Notes
     (`ui.RUN_KEY` + `st.switch_page`; the #seg-N anchors work within one page only).
   - The CLI (`cohr index rebuild`, and whether `cohr search` is worth adding) and whether
     index.db is a schemas.py model (§19) or plain SQL with its own `schema_version`.
   - A search check without models: query each eval run's planted facts against an index built
     from evals/data/runs/ (every language there), report-only. Say whether it's worth it.
1. The index module (src/cohr/index.py): build, rebuild, re-index one run in one transaction,
   drop and rebuild on a missing, corrupt or old-schema file. test_index.py per §13: rebuild
   from fixture runs equals incremental updates; Delete and Prune update rows; a corrupt or
   old-schema file triggers a rebuild; a partial run (no notes yet) is indexed by what it has.
2. Sync hooks and `cohr index rebuild`, with a test where re-indexing must not happen (an
   unchanged manifest_hash).
3. Search page (app_pages/search.py, layout="wide"): query box, filters (date range,
   language, kind), FTS5 snippet() results with the run and a link that opens the cited
   moment in Notes; the cross-meeting action-item table, filterable by owner. AppTest tests
   in test_app.py; check the page at a narrow and a wide window (§7 widths).
4. If step 0 keeps it: the planted-fact search check, per language.
Rules from CLAUDE.md apply. Ask me before new dependencies (none should be needed: sqlite3 is
stdlib). quick needn't run unless pipeline or schema code changes; if it does, it must pass
its gate. Don't bump the version: I'll ask for 1.1.0 separately.
When done: tick the checklist, add new decisions to §20, update §13 where the build differs
from the sketch, add a NOTES.md entry, make sure the gate passes, and open a PR (ask before
pushing).
```

### Kickoff prompt for Phase 8a (cross-language citation times)
```text
Continue cohr: Phase 8a (cross-language citation times, post-v1 item 2). Read CLAUDE.md, then
PLAN.md §5.5–5.7 (Map, Reduce and its jump times, Ask), §18 (citation accuracy, the lang set),
§19, D19, D34 and D38 in §20, and NOTES.md: the 6e-2 entry (step 3, "New carry-over
(cross-language jump times)", and the post-v1 list) and the Phase 7 entry (clean-hi owners).
Work on branch phase-8a-jump-times from main (v1.1.0 is released: #46, #47).
The problem: `reduce.refine_time` finds a citation's moment by token overlap between the cited
span and each transcript chunk of the segment (`fuse.tokens`, `fuse.overlap`). When the notes
and the transcript are in different languages nothing overlaps, no chunk reaches
`citation_min_score` and the citation gets the segment's start: every clean-hi action item
cites t = 0.002 s (46–246 s early). Baseline-lang citation accuracy: clean-hi 58.3%, xling-ja
70% (English notes, Japanese transcript), every other item 100%. Arabic escapes only because
its task text keeps the Latin names; sw and bn because "Other (beta)" runs Reduce in English
and translates after. Ask uses the same refiner (`reduce.segment_refiner`), so its answers in
another language fall back too (5b, NOTES "German answer, no token overlap").
Correction to the post-v1 candidate ("refine on the claim's English Map item"): Map writes
every value in the notes language (`map.json` of clean-hi is Hindi), so there's no English
Map text to match. Don't build on it without a new reason.
0. Proposal before code (ask me). First measure, without models: replay `refine_time` on the
   stored lang runs (notes.json `raw`, segments, transcript; the D19 approach, "offline on
   stored replies") and list, per item, which citations fall back and what the span shares
   with its true chunk (digits, Latin names, owner and due, nothing). Then propose the fix,
   in Python (CLAUDE.md: values are fixed in Python, not in prompts). Candidates, not choices:
   - Language-neutral anchors when word overlap fails: numbers read in each text's own
     language (`numbers.source_numbers_in` for the transcript, `fts.fold_digits` for notes in
     other scripts; "35万4000" and "354 mil" read as 354000), and names: an action item's
     owner and names in the claim in another script, read through D34's anyascii
     transliteration (लॉरा → Laura; Japanese ビクター doesn't match Victor, post-v1 item 5).
   - Via the item's Map note: the action item's owner/due and Map task in the same segment,
     whose words may share more with the transcript than the R7B claim does.
   - Anything that needs a Map output change (e.g. a chunk id per item) bumps FORMAT_VERSION
     and is a Map prompt change, judged on seeds 1–3: say what it would cost before choosing it.
   Say what happens when no anchor is found: the segment start stays the honest fallback,
   never an invented time (§5.6). Embeddings or a translation model are out of scope (models,
   memory, licensing) unless the measurement shows nothing else works.
1. The fix in `reduce.py` (or a small shared helper if Ask needs the same code). Tests: a
   Hindi claim with a number and an owner lands at its chunk's sentence; an English claim on a
   Japanese transcript lands by its number; and tests where the new anchor must not fire
   (CLAUDE.md: a fallback needs one): an English claim that already overlaps keeps today's
   time exactly, and a number said in two chunks of the segment doesn't win over the chunk the
   words match. No prompt changes, so REDUCE_HASH must not move.
2. Measure on every set it can reach: the offline replay first (before/after per item), then
   lang -c seed=1 with `--rerun reduce` (the stage's config hash doesn't move on a code-only
   change) and quick, which must pass its gate unchanged (its citations are English on
   English and must keep their times). Run xling-es and clean-hi once with Ask and the judge
   on, since the refiner change reaches Ask (CLAUDE.md: lang always runs --no-ask). Long-en
   doesn't need to run unless step 1 changes two-pass code.
3. Rewrite baseline-lang.json if the fix moves it on purpose, with the reason in the commit.
Rules from CLAUDE.md apply. Ask me before every full or long run, the opt-in model tests and
new dependencies (none should be needed). quick and lang runs are fine. The rest of the
post-v1 list stays open; don't fix it here unless step 0 shows it shares this cause. Don't
bump the version: I'll ask separately.
When done: tick the checklist, add new decisions to §20 (the anchor rule as a D-number), update
§5.6's jump-time paragraph, add a NOTES.md entry, make sure the gate passes, and open a PR
(ask before pushing).
```

### Kickoff prompt for Phase 8b (made-up action items on long meetings)
```text
Continue cohr: Phase 8b (made-up action items on long meetings, post-v1 items 1 and 6). Read
CLAUDE.md, then PLAN.md §5.5–5.6 (Map, Reduce, two-pass), §18 (made-up items, the long set,
AMI), §19, D33 and D34 in §20, and NOTES.md: the 6e-2 entry (step 2 and step 5 "long", the
post-v1 list) and the 8b kickoff entry. Work on branch phase-8b-made-up-tasks from main
(v1.2.0 is released: #51, #53).
The problem: on the 6e-2 release gate, long-en listed 12 made-up items (`baseline-long.json`;
6b-2 had 8). Replaying pass 2 with main's code showed the cause: step 2 (D33) gives pass 2
more action items, and pass 2 keeps Map's vague tasks along with the real ones ("Continue
monitoring retention metrics…", "Review and clarify the reporting of cloud costs").
Correction to the post-v1 candidate ("a filter for task-shaped claims with no owner and no
due"), from the stored run evals/data/runs/long-en/20261014-115732-long-en and report
20261014-123008-long.json: 9 of the 11 made-up action items have no owner, but 6 of those 9
*do* have a due, "end of next week", and only 1 of long-en's 7 owner-less dues is said in a
cited segment. AMI is the same (post-v1 item 6): 18 owner-less dues in the 5 meetings, none
said in a cited segment; map.json has "end of next week" on 22 lines, the AMI transcripts say
"next week" once. That phrase is map.md's own example due (line 12), and
evals/gen/script.py plants it too (RELATIVE_DUES), so long-en also rewards copying the example
(CLAUDE.md: example lines must not overlap the script's wording). quick and lang have no
owner-less action items in their stored runs except one in conflict-de, so neither set
tests this code. The other 2 made-up items have owners ("Read marketing emails by Monday",
Omar Crews; "Summarise competitor research by 1 November", said "11 November"): that wrong
date is a separate value error, kept separate here.
0. Proposal before code (ask me). First measure, without models: on the stored long-en and
   AMI runs, list every action item with its owner and due, whether each value is said in a
   cited segment (read in the meeting's language: dates through `numbers`, digits folded),
   and, for long-en, whether `evals.made_up` counts it. Then propose the fix in Python
   (CLAUDE.md: values are fixed in Python). Candidates, not choices:
   - A due guard: a due its cited segments don't say is dropped, like the number guard
     (never replaced by a guessed one). Must not fire on cross-language notes (a Hindi due on
     an English meeting) unless it can read the due in the meeting's language.
   - Then, for the action items left with no owner and no due: a task-shape filter ("continue
     …", "ensure …", "review and clarify …"), moving them to Key points rather than deleting
     them (§5.6: claims are kept; say why if you argue for dropping). Show its must-not-fire
     cases: "Run the user conference online" (long-en, no owner, real) and the AMI tasks D33
     found to be decided designs.
   - Where it runs: on the final notes only (as `restore_owners` does) keeps the stored
     replies replayable and leaves pass 2's input as it is; on pass-1 claims it changes what
     pass 2 writes (CLAUDE.md), so it needs a long-en run to measure. In Map (map.json) it
     would also reach Ask, but changes Map's stage output: say what that costs.
   - Removing "end of next week" from map.md's example is a Map prompt change (Map reshuffles
     at temp 0.3, judged on seeds 1–3, and reaches every set): say whether the Python fix
     makes it unnecessary, and pick an example due that isn't in evals/gen/script.py if not.
1. The fix, with tests: the guard drops an unsaid due and keeps a said one ("by the 17th of
   November" for "17 November", "Friday" for "Friday"); the filter moves a vague owner-less
   task; and must-not-fire tests (CLAUDE.md: a fallback needs one): an owner-less task that's
   real, a due said in the cited segment but in other words, a due in another language. No
   prompt change unless step 0 argues for it (then REDUCE_HASH or Map's hash moves on
   purpose).
2. Measure: replay on the stored runs first (`build_notes` + `restore_owners` on notes.json
   `raw`, then `evals.score_item` on a copy of each run, CLAUDE.md: copy, don't delete);
   long-en made-up items and recall before/after, AMI action items moved or changed. AMI
   coverage needs the judge, so its run is an AMI-only run: ask first. Then quick (the gate
   must pass, unchanged) and lang -c seed=1 --no-ask (conflict-de's owner-less task, and dues
   in other languages). long-en (~40 min) only if the change reaches pass 1 or Map, or to
   confirm the replay: ask first.
3. Rewrite the baselines the fix moves on purpose, with the reason in the commit.
Rules from CLAUDE.md apply. Ask me before every full or long run, AMI runs, the opt-in model
tests and new dependencies (none should be needed). quick and lang runs are fine. The rest of
the post-v1 list stays open (the "1 November" date and Ask on long-en, items 7 and 10,
included) unless step 0 shows it shares this cause. Don't bump the version: I'll ask separately.
When done: tick the checklist, add new decisions to §20 as D-numbers, update §5.6 (owner and
due rules), add a NOTES.md entry and update the post-v1 list, make sure the gate passes, and
open a PR (ask before pushing).
```

### Kickoff prompt for Phase 8c (Ask on long meetings)
```text
Continue cohr: Phase 8c (Ask on long meetings, post-v1 item 7). Read CLAUDE.md, then PLAN.md
§5.5–5.7 (Map, Reduce citations and `refine_time`, Ask), §18 (Ask metrics, the long set),
§19, D19, D27, D39 and D40 in §20, and NOTES.md: the 8b entry ("Open"), the post-v1 list and
the 8c kickoff entry. Work on branch phase-8c-ask-long from main (v1.3.0 is released: #56,
#57).
The problem: long-en Ask accuracy is 77.8% (18 of 24; 94.4% on the audio before 6e-2), with
the same 4 misses in reports 20261014-123008-long.json and 20261028-125536-long.json (run
evals/data/runs/long-en/20261014-115732-long-en; cached replies evals/data/ask/long-en.json).
Correction to the post-v1 candidate ("check with Ask ranking"): replaying `ask.rank` on the
stored run puts the segment holding each answer at rank 0, 0, 0 and 2 (at most 12 are sent,
MAX_RANKED), so the right text reaches R7B every time. The misses, from the kickoff check:
- a-f26 "What does Tom need to do by November 17?": answer right, citation −197.7 s. R7B
  wrote `<co>Tom</co: 0:[5]> needs to <co>finish the incident post mortem…</co: 0:[5]>`;
  `refine_time` places "Tom" at his first chunk in seg 6 (1820.88) and the full span at
  2019.56 (right), and the claim keeps the first. Same `resolve` path as Reduce's notes.
- a-f61 "What does Omar need to do by the end of next week?": Map's Omar line is right (seg
  16, rank 0), but 8 of the 12 top-ranked documents carry a `[due: end of next week]` Map line (9 in
  the meeting; map.md's example, 8b "Open"), and R7B answered Maria's real one, ignoring the
  owner the question names.
- a-f82 "What does Grace need to do by November 11?": Map wrote "November 1" in both its
  decision and its action line (the transcript says "11 November"), and R7B answered another
  Grace task ("book the venue … by November 14th"). The 8b "1 November" date.
- a-f15 "What does Omar need to do by Wednesday?": the transcript says "Oma will organize the
  team lunch by Wednesday" (ASR dropped the r); Map has no lunch item, and R7B answered
  Omar's Map line "Read marketing emails by Monday" instead of the transcript.
So 3 of 4 trace to Map's values (Ask's documents put Map's lines before the transcript) and 1
to citation placement. Read the run files yourself before building on this (CLAUDE.md).
0. Proposal before code (ask me). First, without models: for each miss, the cached raw reply,
   its cited spans and where each places, and the Map lines and transcript text the sent
   documents hold for the question's owner and date. Then count, over all 24 cached replies
   and the stored notes.json `raw` of quick, lang and long-en, how often a claim's citations
   to one segment place differently by span and which one wins (name-only spans). Then
   propose fixes in Python first (CLAUDE.md: values are fixed in Python). Candidates, not
   choices:
   - Citation time: when one claim cites a segment through several spans, place it by the
     span that scores best (or the claim sentence) instead of the first; a name-only span
     (an `[owner: …]` name alone, D39's anchor) never wins over a longer one. Reaches Reduce's
     notes on every set, so citation accuracy on quick, lang and long-en must not drop.
   - Map's values in Ask's documents: apply 8b's due guard (`dues.said`) and a date digit
     check (like Map's number guard: "November 1" vs said "11 November") to the Map lines
     `segment_snippet` writes, or to map.json itself. The snippet side reaches Ask and
     Reduce's input (REDUCE_HASH doesn't move, but Reduce's output can: a long-en run); the
     map.json side changes Map's stage output (say what that costs). Preferred (user): fix
     the value on Map's items, so map.json holds it once for every reader (Ask's snippets,
     Reduce, search, 8b's "Summarise competitor research by 1 November" task); the
     snippet-only side leaves the wrong value in map.json. Say where it runs so a
     `--rerun reduce` run picks it up without re-running Map, and propose the snippet side
     only if step 0 shows a reason. Show must-not-fire cases: a due said in other words, a
     cross-language due (8b's digits-only rule).
   - Removing "end of next week" from map.md's example: a Map prompt change (seeds 1–3,
     every set, example not in evals/gen/script.py). Say whether the Python fix makes it
     unnecessary.
   - a-f15 (a task Map missed under a misspelled name) may stay a miss: say whether anything
     short of Map recall reaches it, and don't fit a rule to it alone.
   - An Ask prompt rule ("answer for the person the question names") only if step 0 shows
     the Python fixes leave a-f61 wrong; ASK_PROMPTS hashes move, so every cached answer
     re-runs.
1. The fixes, with tests: a claim with a name-only span and a full span on one segment places
   at the full span; a guarded or repaired Map line in the snippet; and must-not-fire tests
   (CLAUDE.md: a fallback needs one): one span per segment unchanged, a said due kept, a
   Hindi due on an English meeting kept.
2. Measure: replay first. Citation fixes: `parse_answer` on the cached raw replies and
   `build_notes` + `restore_owners` on stored notes.json `raw`, scored with `evals.score_item`
   on a copy of each run (copy, don't delete). Snippet changes move the Ask cache key
   (sources hash), so they need R7B: quick and lang -c seed=1 --no-ask with `--rerun reduce`
   are fine; long-en (~10 min with `--rerun reduce`, Ask included): ask first. Report Ask
   accuracy, stated, citation accuracy and near-miss, and notes recall, citation accuracy and
   made-up items, before and after.
3. Rewrite the baselines the fix moves on purpose, with the reason in the commit.
Rules from CLAUDE.md apply. Ask me before every full or long run, AMI runs, the opt-in model
tests and new dependencies (none should be needed). quick and lang runs are fine. The rest of
the post-v1 list stays open (items 3, 4, 5, 8 and 10) unless step 0 shows it shares a cause.
Don't bump the version: I'll ask separately. When done: tick the checklist, add new decisions
to §20 as D-numbers, update §5.6–5.7 (citation placement, Ask documents), add a NOTES.md entry
and update the post-v1 list, make sure the gate passes, and open a PR (ask before pushing).
```

### Kickoff prompt for Phase 8d (missed action items)
```text
Continue cohr: Phase 8d (missed action items, post-v1 item 3, widened). Read CLAUDE.md, then
PLAN.md §5.5–5.6 (Map, its guards, Reduce's task handling), §18 (action recall, owner and due
matching), §19, D33, D34, D40 and D41 in §20, and NOTES.md: the 8c entry ("Open"), the
post-v1 list and the 8d kickoff entry. Work on branch phase-8d-missed-tasks from main (v1.4.0
is released: #61, #63).
Correction to the post-v1 candidate ("a Map prompt rule, judged on seeds 1–3"): Map writing a
task only as a decision is 2 of the 17 action items missed in reports
20261028-172217-quick.json, 20261028-165932-lang.json and 20261028-172216-long.json. From
the kickoff check (scripts/trace_facts.py for English; lang map.json read by hand, since the
trace matches English only):
- Decision only, no Map task (2): long-en f22 seg 6 "Tom will update the pricing page by the
  2nd of November." (the notes have it in no section); xling-es f8 seg 0 "Victor will write
  the notes for the next version."
- Map task without its owner (3): clean-sw and clean-bn f11 "Prepare customer survey by end
  of next week" (owner null; Maria), conflict-de f6 "Verfasste Release-Notizen bis Mittwoch"
  (owner null, past tense; Grace).
- Map task, final notes lost or changed it (5): long-en f31 (Laura, shared drive) and f82
  (Grace, competitor research; 8c's repaired date) are in map.json and in no notes section;
  f40 is in the notes as "Omar Crews | Read marketing emails"; clean-ar f17 (Tom's video,
  seg 1) is in map.json, not the notes; conflict-de f12 is in the notes with its owner lost.
  long-en's pass-1 notes aren't stored, so where pass 1 or pass 2 lost them needs a replay.
- Map never wrote it (7): long-en f14, f15 (ASR "Oma"), f19, f71, f86; noisy-en f7;
  conflict-de f5. Only Map recall reaches these.
The kickoff's scan of every stored map.json for decisions shaped "<subject> will/to/should
…" with no Map task in the segment (coverage < 0.6) found 22: the 2 above, xling-ja 3
("Onboarding checklist will be retained") and 17 on AMI, all product decisions with no person
("The remote control will incorporate corporate image", "Voice recognition will not be
included"). So a rule keyed on the verb alone would move AMI's real decisions (8b lesson);
the subject has to be a person the meeting names. Read the run files yourself before building
on this (CLAUDE.md).
0. Proposal before code (ask me). First, without models: re-check the four groups on the
   stored runs and the reports' fact lists; for the 5 the notes lost, find the stage that lost
   each (Reduce's raw reply, its parse, D33's move, D40's due guard and vague-task move,
   restore_owners, pass 2; replay pass 1 offline where needed) and for the 3 owner-less Map
   tasks, what the cited sentence says. Then propose fixes in Python first (CLAUDE.md: prompts
   set shape, Python fixes values). Candidates, not choices:
   - Decision → Map task on map.json (where 8c put Map's fixes, so Ask, Reduce and search
     read one value): a Map decision whose subject is a name said in the meeting
     (`names.said_names`, whole word; English notes first, say what other notes languages
     need) followed by a task cue, with no Map action item in the segment at ≥ 0.6, becomes
     an action item with that owner; its due only if `dues.due_said` passes. Must-not-fire:
     every AMI hit above, "we will …", a decision a Map task already states (D33 handles
     those in Reduce). Say whether it changes MapFile (a new field moves every Ask cache key,
     8c lesson) and whether it runs where `--rerun reduce` picks it up without Map.
   - Owner-less Map tasks: fill the owner only from the task's own cited sentence (one name
     said in it, directly addressed or the subject), never from a neighbour task (map.md's
     rule). Show the must-not-fire case: a sentence naming two people.
   - Tasks the notes lose: depends on step 0's stage. A pass-1/pass-2 drop is a Reduce
     prompt or budget question (REDUCE_HASH moves; check `reduce.max_tokens` first,
     CLAUDE.md); a Python drop is fixed where it happens.
   - A map.md prompt rule ("a task someone will do goes in action_items even when it was
     also agreed") only if the Python rule can't reach a case; seeds 1–3, every set, examples
     not in evals/gen/script.py. Say what it would cost.
   - The 7 Map never wrote stay open unless step 0 finds a shared cause (e.g. dense segments:
     long-en seg 4 holds f14, f15 and f19).
1. The fixes, with tests: Victor's and Tom's decisions become owned action items; AMI's
   product decisions, "we will …" and an already-listed task don't move; an owner filled from
   its own sentence, not from one naming two people.
2. Measure: replay first. Map-side Python on copies of the stored runs (copy, don't delete),
   then `build_notes` + `restore_owners` on notes.json `raw`, scored with `evals.score_item`.
   Then quick and lang -c seed=1 --no-ask (`--rerun reduce` if the fix runs after Map) and
   long-en (~10 min with `--rerun reduce`, Ask included: ask first). Report action recall,
   owner and due accuracy, made-up items, decisions recall and Ask accuracy before and after;
   AMI coverage and decision counts from a replay of the stored AMI runs (an AMI run: ask).
3. Rewrite the baselines the fix moves on purpose, with the reason in the commit.
Rules from CLAUDE.md apply. Ask me before every full or long run, AMI runs, the opt-in model
tests and new dependencies (none should be needed). quick and lang runs are fine. The rest of
the post-v1 list stays open (items 4, 5, 8 and 10, and 8c's a-f61 Ask rule) unless step 0
shows it shares a cause. Don't bump the version: I'll ask separately. When done: tick the
checklist, add new decisions to §20 as D-numbers, update §5.5–5.6 (Map guards, task
handling), add a NOTES.md entry and update the post-v1 list, make sure the gate passes, and
open a PR (ask before pushing).
```

### Kickoff prompt for Phase 8e (action items Map never wrote)
```text
Continue cohr: Phase 8e (action items Map never wrote, 8d's "Open"; 8d's long-en made-up
4 → 6 measured alongside). Read CLAUDE.md, then PLAN.md §5.5–5.6 (Map, its guards, Reduce's
task handling), §18 (action recall, owner and due matching, made-up items), §19, D33, D34,
D40, D41 and D42 in §20, and NOTES.md: the 8d entry ("Open"), the post-v1 list and the 8e
kickoff entry. Work on branch phase-8e-map-recall from main (v1.5.0 is released: #65, #67).
Correction to the candidate I gave first ("dense segments: split them before Map"): long-en's
segments are all one size (261–299 s, 726–864 words). Seg 4 (816 words) wrote 0 of its 3 tasks
and no decision ("with no major decisions or action items"); seg 6 (804) wrote all 3. Where
a task sits in its segment doesn't predict it either: misses at 3% to 95% of the text, hits
the same. From the kickoff check on reports 20261029-064420-quick.json,
20261029-072413-lang.json and 20261029-072413-long.json (scripts/trace_facts.py; lang
map.json read by hand):
- In no map.json (8): long-en f14 "Maria, can you send the contract to legal …?", f15 "Oma
  will organize the team lunch …" (ASR for Omar), f19 "Kenji will draft the hiring plan …"
  (all seg 4), f71 "Action item for Grace fix the login timeout bug …" (seg 21), f86 "Kenji
  will order laptops …" (seg 27, just before Grace's task, which Map wrote); noisy-en f7
  "Victor, can you book the venue for the F site …?"; conflict-de f5 "Nadia, can you send the
  contract to legal …?" and f6 "Grace will write the release notes …" (a German Map task
  with no owner and no word overlap, 8d). f40 is ASR, out of scope.
- Every one is a cue sentence `names.addressee` reads (D42's cue forms). A scan of each
  segment's sentences for an addressee who owns no Map task in that segment (owners folded)
  reaches 7 of the 8. It misses noisy-en f7: ASR's "F site" makes "F" a second person, so
  `addressee` returns None. On the other stored runs it flags:
  - "Okay, will do." in 9 runs. `names._NOT_PEOPLE` lacks "okay". It was never written as an
    owner (8d reads only the sentences Map's tasks cite), but a transcript-wide scan would add
    it.
  - long-en seg 9 "Laura Schedule": `_PERSON` took ASR's capitalised verb as a surname.
    Laura's f36 is in Map.
  - clean-hi: all 6 tasks, because Map wrote the owners in Devanagari and the scan compared
    folded Latin. Compare through `names.restore` or the skeleton.
  - xling-es and xling-ja: 0 cue sentences (English cues only).
- AMI has 0 such sentences: its speakers address no one by name ("can you go on to the next
  slide?") and give themselves tasks ("I'll have to ask the manufacturing department"). So
  AMI can't show this rule misfiring, and evals/gen plants every task in these three cue forms:
  the 8b risk of a rule that fits the generator. Must-not-fire cases have to be written as
  tests from real speech: "Tom, can you hear me?", "Maria, can you go to the next slide?",
  "Kenji will be out next week", "Grace will love this".
- Each report ran Map at one seed. Whether Map also misses these at seeds 2 and 3 isn't known.
Read the run files yourself before building on this (CLAUDE.md).
0. Proposal before code (ask me). First, without models: re-run the scan on every stored run
   (fix the three false flags above in the scan, not yet in `names`), list each flag with its
   sentence, and check quick at -c seed=2 and seed=3 with --no-ask --no-judge (fine to run)
   to see if the same tasks go missing. Then propose. Candidates, not choices:
   - A Python recall net on map.json, after `guard_owners` (where Ask, Reduce and search read
     one value, as in 8c and 8d): an uncovered cue sentence becomes a Map action item. Owner
     = the addressee, task = the verb phrase after the cue, due only if `dues.due_said`
     passes. Works only when the notes are in English: Map writes in the notes language
     (conflict-de's Map is German), so say what other notes languages get.
   - A small model call on just the uncovered cue sentences ("which of these gives someone a
     task? write each as an action item", prompt in src/cohr/prompts/, notes language, no
     example lines that overlap evals/gen/script.py). The model separates "can you hear me"
     from a task, and Python keeps the owner and checks the due. Say the cost: calls per
     segment, Map's config hash, and whether `--rerun` can reach it without a full Map rerun.
   - A map.md rule or a prefill (CLAUDE.md: start the reply, don't reword), only if neither
     reaches a case; seeds 1–3, every set.
   For each: whether it changes MapFile (a new field moves every Ask cache key, 8c lesson),
   what it does to made-up items, and the `names` fixes it needs ("okay" and replies like
   "Okay, will do."; a capitalised verb after the name; "F" and other one-letter words as
   people). Also say whether long-en's 2 extra made-up tasks (seg 25, owner-less, from 8d's
   pass 2) share a cause; otherwise they stay open.
1. The fixes, with tests: the 7 sentences above become owned action items. The must-not-fire
   sentences, "Okay, will do.", a sentence naming two people and an already-listed task add
   nothing.
2. Measure: replay first. Apply a Python fix to copies of the stored runs' map.json (copy,
   don't delete) and rebuild with `build_notes` + `restore_owners` on notes.json `raw` where
   Reduce isn't needed. A Map-side addition changes Reduce's input, so then run quick and lang
   -c seed=1 --no-ask (`--rerun` from the stage the fix lives in), and long-en (Ask included;
   ask first). Report action recall, owner and due accuracy, made-up items, decisions recall,
   citation accuracy and Ask accuracy before and after. For AMI, give the scan's count (it
   should add nothing) and replay the stored runs (an AMI run: ask).
3. Rewrite the baselines the fix moves on purpose, with the reason in the commit.
Rules from CLAUDE.md apply. Ask me before every full or long run, AMI runs, the opt-in model
tests and new dependencies (none should be needed). quick and lang runs are fine. The rest of
the post-v1 list stays open (items 4, 5, 8 and 10, 8c's a-f61 Ask rule, 8d's B2 order) unless
step 0 shows it shares a cause. Don't bump the version: I'll ask separately. When done: tick
the checklist, add new decisions to §20 as D-numbers, update §5.5–5.6, add a NOTES.md entry
and update the post-v1 list, make sure the gate passes, and open a PR (ask before pushing).
```

### Kickoff prompt for Phase 8f (missed decisions)
```text
Continue cohr: Phase 8f (missed decisions). Read CLAUDE.md, then PLAN.md §5.5–5.6 (Map, its
guards, Reduce's decision and task handling), §18 (decision recall, judge-only facts in lang,
made-up items), §19, D33, D42 and D43 in §20, and NOTES.md: the 8e entry ("Open"), the
post-v1 list and the 8f kickoff entry. Work on branch phase-8f-decisions from main (v1.6.0
is released: #69, #71).
Decisions are the weakest kind in every set, and no phase has worked on them: quick 75% at
seed 0 (81.25 / 81.25 / 87.5% at seeds 1 / 2 / 3), long-en 70%, lang 21 of 28 judge-covered.
From the kickoff check on reports 20261029-112133-quick.json, 20261029-114720-long.json and
20261029-112647-lang.json (scripts/trace_facts.py; the 8e seed 1–3 quick reports):
- evals/gen plants every decision in one of three forms: "We decided to …", "We agreed to …",
  "OK, so the decision is to …". A scan of those sentences against each segment's Map
  decisions (2+ shared content words) flags exactly the English misses:
  - In no map.json (7): quick noisy-en f3 "We agreed to keep the onboarding checklist."
    (Map wrote 3 decisions there, one of them an open question: "Move the company blog name
    discussion to next time"); slide-conflict-en f2 "We decided to move the weekly sync to
    Thursdays." and f14 "… freeze new features until the beta ships." (Map wrote 2 others;
    slides don't touch either); long-en f18 (seg 4, Map decisions []), f27 (seg 6, Map's one
    "decision" is Maria's task), f30 (seg 8, []), f33 (seg 9, Map wrote f35 and f37).
  - In Map, lost later (2): clean-en f2 "We agreed to use the new logo on the website." is
    in Map's key points (×2), not its decisions, and not in the notes; long-en f49 "consolidate
    the design tools" is a Map decision (seg 13) the notes don't have. Read pass 1's raw
    (`two_pass.groups[].raw`) to see which pass dropped it.
  - long-en f55 "standardize on Postgres": ASR wrote "PostGIS"; the notes' "Standardise on
    PostGIS for analytics." counts as a made-up item. ASR, like 8e's f15/f40: out of scope.
- Unlike 8e, two misses don't move with the seed: noisy-en f3 and slide-conflict-en f2 are
  missed at seeds 0, 1, 2 and 3. Others do (clean-en f15 at seed 1 only, noisy-en f15 at
  seed 2 only). So a map.md change isn't ruled out the way it was in 8e; check it per seed.
- Map's decisions list carries other kinds too: clean-en seg 0 has 4 of its 5 "decisions"
  as tasks ("Laura to send contract to legal by Friday"; D33 moves them), noisy-en an open
  question. Whether those crowd out real decisions isn't known.
- lang: conflict-de misses the same two as slide-conflict-en (f2, f14), clean-ar 3 of 4,
  clean-sw and clean-bn "move the weekly sync to Thursdays". Map writes in the notes language,
  so the English word scan can't say where they were lost: read those map.json by hand.
  lang decisions are judge-only (`judge_only`, recall_decision is None), so run lang with the
  judge on to measure them.
- AMI has 6 sentences in these forms across 5 meetings. 4 are already in Map's key points
  or tasks; "I think we all agreed on that." refers back; "we agreed on the requirements and
  the target market" sums up an earlier meeting. Real meetings decide in other words, so the
  8b risk applies: a rule keyed to the three forms fits the generator. Must-not-fire cases
  have to be tests from real speech: "I think we all agreed on that.", "Have we decided to
  …?", "We haven't decided whether to …" (an open question), "We agreed last week to …",
  "Did we agree on the date?".
Read the run files yourself before building on this (CLAUDE.md).
0. Proposal before code (ask me). First, without models: re-run the scan on every stored run
   (AMI included), read the lang map.json files for the misses above, find where f49 and
   clean-en f2 were lost, and check quick at -c seed=2 and seed=3 with --no-ask --no-judge
   (fine to run; copy the stored runs first, 8e lesson) if the 8e reports aren't enough.
   Then propose. Candidates, not choices:
   - D43's path for decisions: the uncovered cue sentences of a segment go to one small call
     (prompt in src/cohr/prompts/, notes language, no example lines that overlap
     evals/gen/script.py), or into the same call as map_tasks.md. Say the cost per segment,
     the config hash, and what `--rerun` reaches.
   - A Python net on map.json: the verb phrase after the cue becomes a decision, English notes
     only (say what other notes languages get).
   - A map.md change or a prefill, now that two misses hold at every seed: judge on the mean
     of seeds 1–3, every set (CLAUDE.md), and say what it does to which tasks Map keeps.
   - Reduce side, for f49 and clean-en f2: a Map decision the notes lost is added (as
     `reduce.add_lost_tasks` does for tasks), and a Map key point a cue sentence makes a
     decision moves to Decisions.
   For each: whether it changes MapFile (a new field moves every Ask cache key, 8c lesson),
   what it does to made-up items and to open-question recall (a decision rule mustn't take
   "We still don't know whether we should …"), and its must-not-fire cases.
1. The fixes, with tests: the misses above become decisions with their citation. The
   must-not-fire sentences and an already-listed decision add nothing.
2. Measure: replay first. Apply a Python fix to copies of the stored runs (copy, don't
   delete) and rebuild with `build_notes` + `restore_owners` on notes.json `raw` where Reduce
   isn't needed. Then quick at seeds 0–3 --no-ask --no-judge, quick with Ask and the judge,
   lang -c seed=1 --no-ask with the judge on (`--rerun` from the stage the fix lives in), and
   long-en (Ask included; ask first). Report decision recall (lang: judge-covered), made-up
   items, open-question and action recall, citation accuracy and Ask accuracy before and
   after. For AMI, give the scan's count and replay the stored runs (an AMI run: ask).
3. Rewrite the baselines the fix moves on purpose, with the reason in the commit.
Rules from CLAUDE.md apply. Ask me before every full or long run, AMI runs, the opt-in model
tests and new dependencies (none should be needed). quick and lang runs are fine. The rest of
the post-v1 list stays open (items 4, 5, 8 and 10, 8e's "Open") unless step 0 shows it shares
a cause. Don't bump the version: I'll ask separately. When done: tick the checklist, add new
decisions to §20 as D-numbers, update §5.5–5.6, add a NOTES.md entry and update the post-v1
list, make sure the gate passes, and open a PR (ask before pushing).
```

### Kickoff prompt for Phase 8g (missed numbers)
```text
Continue cohr: Phase 8g (missed numbers, post-v1 item 4, widened). Read CLAUDE.md, then PLAN.md
§5.5–5.6 (Map, its number and date guards, Reduce), §18 (number facts, unsupported numbers),
§19, D20, D35, D38, D41 and D44 in §20, and NOTES.md: the 8f entry ("Open"), the post-v1 list
and the 8g kickoff entry. Work on branch phase-8g-numbers from main (v1.7.0 is released: #73,
#75).
After 8b–8f, numbers are long-en's weakest kind (31/40, 77.5%), and the latest reports miss 17
number facts. From the kickoff check on reports 20261029-131543-quick.json,
20261029-131544-long.json and 20261029-131543-lang.json (scripts/trace_facts.py), and the
stored runs' map.json:
- Post-v1 item 4 is mostly done. Map's guard (`mapper.check_numbers`, D20) repairs 30+
  miscopied values on the stored runs (425,000 → 42,500, 2,790,000 → 279,000; xling-ja
  13,000 → 134,000), and 8c's `guard_dates` turned long-en's "1 November" into 11 (the notes
  say "by 11 November"). One wrong digit is left: xling-ja f9. Map wrote "Monthly recurring
  revenue is $34,000, up from $134,000 last month." for 35万4000 and 13万4000. 34,000 is one
  edit from both 354,000 and 134,000, so `near_by` holds two values and the guard flags it
  (unsupported 34000). The claim's other number is 134,000 already: a source value the same
  claim states is a weak candidate. Must not fire: a claim that really says one value twice.
- A wrong repair on real speech: AMI IS1009b seg 6 "40% large buttons, 40% regular buttons".
  The speaker said "forty percent we make large buttons and the rest we make regular"; Map's
  60 (the rest) was "repaired" to 40, one digit edit away. A fix that widens repairs must
  check this case, and step 0 should say whether D20's one-edit rule needs a bound for
  one- and two-digit values (count AMI's repairs and how many are right).
- Reduce lost 6 that Map wrote as key points (trace: Map key_points×1, notes —): long-en f43
  (38%, seg 11), f66 (99, seg 18), f79 (85%, seg 24); slide-conflict-en f7 ($18,000; the trace's
  ASR column says MISS, so read the transcript first); clean-ar and clean-hi f16 (4,650, seg 1;
  Map wrote it in clean-ar's summary as well). Read pass 1's raw (`two_pass.groups[].raw`) for
  the long-en three to see which pass dropped them. This looks like 8f's Reduce losses: one
  Python step after Reduce (`reduce.add_said_decisions` for decisions, `add_lost_tasks` for
  tasks) re-added them.
- Map never wrote 8: long-en f8 (660 ms, seg 1), f11 (4,550, seg 2), f16 ($79,000, seg 4),
  f45 (520 ms, seg 11), f50 ($41,000, seg 13), f70 (374, seg 20); pdf-slides-en f12 ($8,900);
  xling-ja f7 (11,800, 1万1800). long-en's Map writes exactly 3 key points in 17 of its 28
  segments (quick and lang segments get 4–8), and the 6 long-en segments with a miss (1, 2, 4,
  11, 13, 20) wrote 3 each. Whether that
  is the segments' filler, a habit, or a token limit isn't known.
- Map's number misses move with the seed: quick misses pdf-slides f12 and slide-conflict f7 at
  seed 0, pdf-slides f6 and f10 at seed 1, slide-conflict f3 and f17 at seed 2, f17 at seed 3
  (reports 20261029-123350/123843/124331). By the 8e lesson, a map.md change only moves them;
  a net catches them.
- xling-ja f6 ($42,500) is ASR: the topic came out as "師範清" and the value isn't in the
  transcript. Out of scope.
- evals/gen plants each number as "The <metric> is <value>. That's compared with <value> last
  <period>." Real meetings say numbers in many other ways (AMI: prices, percentages, counts,
  "one thing", "two of them"), so a rule keyed to that sentence fits the generator, and AMI is
  its must-not-fire evidence (8b and 8f lessons).
Read the run files yourself before building on this (CLAUDE.md).
0. Proposal before code (ask me). First, without models: find the stage for each of the 17
   (pass 1 or pass 2 for long-en), count AMI's said numbers that are in no map.json and no
   notes, and check the IS1009b repair and AMI's other repairs. Then propose. Candidates, not
   choices:
   - Guard: drop a source value the claim already states from `near_by` (xling-ja f9), and
     whatever bound the AMI repairs argue for. Code only on Map: say whether `--rerun map`
     is needed or a replay on map.json is enough, and what it does to Ask keys.
   - Reduce side, like D44: a Map key point with a figure the notes don't state (same value,
     read with `numbers.notes_numbers` in the notes language, folded per D38) is added to Key
     points with its citation. Say what it adds on AMI and to made-up items.
   - Map side, like D43: a segment's said numbers that no Map item states go to one small
     call (prompt in src/cohr/prompts/, notes language, no example lines that overlap
     evals/gen/script.py), or a Python net that copies the sentence. Say the cost per
     segment, the config hash, what `--rerun` reaches, and what keeps it off AMI's filler
     numbers.
   - A map.md change (e.g. key point count on long segments): only with seeds 1–3 on every set.
   For each: whether it changes MapFile (a new field moves every Ask cache key, 8c lesson),
   its effect on made-up items and unsupported numbers, and its must-not-fire cases.
1. The fixes, with tests: the misses above reach the notes with their citation and value as
   said. The must-not-fire cases and an already-stated number add nothing.
2. Measure: replay first. Apply a Python fix to copies of the stored runs (copy, don't delete)
   and rebuild with `build_notes` + `restore_owners` on notes.json `raw` where Reduce isn't
   needed. Then quick at seeds 0–3 --no-ask --no-judge, quick with Ask and the judge, lang
   -c seed=1 --no-ask (`--rerun` from the stage the fix lives in), and long-en (Ask included;
   ask first). Report number recall, unsupported and unverified numbers, made-up items, fact
   recall, citation accuracy and Ask accuracy before and after. For AMI, give the counts and
   replay the stored runs (an AMI run: ask).
3. Rewrite the baselines the fix moves on purpose, with the reason in the commit.
Rules from CLAUDE.md apply. Ask me before every full or long run, AMI runs, the opt-in model
tests and new dependencies (none should be needed). quick and lang runs are fine. The rest of
the post-v1 list stays open (items 5, 8 and 10, 8f's "Open") unless step 0 shows it shares a
cause. Don't bump the version: I'll ask separately. When done: tick the checklist, add new
decisions to §20 as D-numbers, update §5.5–5.6, add a NOTES.md entry and update the post-v1
list, make sure the gate passes, and open a PR (ask before pushing).
```

### Kickoff prompt for Phase 8h (AMI coverage)
```text
Continue cohr: Phase 8h (AMI coverage, 8g's "Open"). Read CLAUDE.md, then PLAN.md §5.6
(Reduce, its markers and the fallback to Summary), §18 (AMI coverage, the judge), §19, D40–D45
in §20, and NOTES.md: the 8g entry ("Open"), the post-v1 list and the 8h kickoff entry. Work on
branch phase-8h-ami from main (v1.8.0 is released: #77, #79).
8g's AMI run (`--item ami --no-ask`, report 20261030-075941-full.json) was the first Map run on
AMI since Oct 14: coverage 52.8 → 44.9%, decisions 52.2 → 30.4% (baseline-full.json, and
20261028-125512-full.json, 8b's Reduce rerun, agrees with it). From the kickoff check on the
stored runs:
- The runs in evals/data/runs/ami-*/ were rewritten in place on Oct 30. The state before is
  kept in evals/data/ami-before-8g/ (notes.json, map.json, log.txt, manifest.json,
  segments.json of all five; Oct 28 Reduce on Oct 14 Map, git-ignored). Compare against it;
  don't overwrite it.
- IS1009b (62.5 → 43.8%, decisions 75 → 0%): R7B's reply changed shape. Before, it wrote the
  five markers once per segment (25 headings, parsed fine). Now it writes all five sections
  for segment 0 only, then "## <co>Remote Control Design</co: 0:[1,2,3,4,5,6]>" and segments
  1–6 as one flat list of 32 lines, 14 of them twice. `split_sections` files everything under
  an unknown heading as Summary (§5.6), so the notes have 34 summary claims, 5 key points and
  4 decisions (before: 10, 31, 13). The flat list holds key points and decisions both
  ("Teletext should be eliminated", "The target group is everyone with a TV"): continuing the
  section before it would file them all as open questions.
- What moved it: Reduce's prompts are unchanged since Oct 13 and it runs at temperature 0.
  map.json differs from before only in dues (8c's `guard_dates` dropped 21, so `_action_line`
  writes `[due: ?]`) and IS1009b seg 6's 60%, which 8g put back to 40 with the same reply.
  Prompt tokens 10,086 → 10,042. A 44-token change flipped the shape, so a fix that only
  brings this one reply back isn't a fix.
- The fallback has fired only on IS1009b, in 3 of its 7 Reduce runs (log.txt): "#" at Oct 14
  06:56 and 08:49, whose reports (071244, 085703) score IS1009b 56.2% against 62.5% for the
  other Oct 14 runs, and the Oct 30 heading. No other stored run in any set has an unknown
  heading. One meeting is thin evidence for a rule: a parse fix also needs tests built from
  this reply's shape.
- The other four kept the shape and moved by one or two claims: ES2004b 53.3 → 46.7%
  (abstract 77.8 → 55.6%; "3 decisions moved to action items (Map's tasks)", 8g added 4
  figures), ES2004d 33.3 → 23.8% (decisions 16.7 → 0%, 10 → 9 decision claims), TS3003c 55.6 →
  50.0% (decision claims 15 → 22, yet decisions 66.7 → 50%), IS1009c unchanged. On Oct 14,
  TS3003c scored 22.2% and 55.6% in different runs, and each meeting has 6–9 reference
  sentences per part: one sentence is 11–17 points.
- The judge's coverage answers are cached by hash (evals/data/judge/ami-*.json). If the
  cache still holds the Oct 28 notes' answers, both sides can be scored offline.
- IS1009b's s/h 2056 is a wall-clock artifact: Reduce logs "done in 77.7 s" but started 17
  minutes before (another run was using the GPU). Not a cost regression.
Read the run files yourself before building on this (CLAUDE.md).
0. Proposal before code (ask me). First, without models: score evals/data/ami-before-8g and
   the current runs with the cached judge (say which answers were missing), and split each
   meeting's change into the notes (which reference sentence lost its claim, and the Python
   step that moved it) and noise. Read IS1009b's raw before and after. Then propose.
   Candidates, not choices:
   - Parse: claims under an unknown heading are filed by their cited segment's Map item
     (the section of the Map key point, decision, task or question they best match), and a
     claim repeated with the same citation is kept once. Pure Python: replay
     `build_notes` + `restore_owners` on notes.json `raw` for the five meetings and every
     stored run in quick, lang and long. Must not fire: a reply with the five markers; a
     heading-less summary paragraph.
   - Coverage check: segments that no claim outside Summary cites get their Map items added
     (the D42–D45 pattern: `add_lost_tasks`, `add_said_decisions`, `add_lost_numbers`). Say
     what it adds on the eval sets and to made-up items.
   - Input: write the action line without "[due: ?]" when Map has none. It moves Reduce's
     input on every set (Reduce runs everywhere; REDUCE_HASH may not move): only with a
     reason beyond IS1009b and every set rerun from Reduce.
   - ES2004b/d: if the judge splits them to a Python step (6e-2's decision → action move, 8f,
     8g), say whether it's noise or a rule to tighten; AMI is that step's must-not-fire set.
   For each: whether it changes NotesDoc (FORMAT_VERSION, snapshots) or Ask keys, and what
   it does on quick, lang and long-en.
1. The fixes, with tests: IS1009b's dumped claims land in their sections with their
   citations, each once; the must-not-fire replies parse as before.
2. Measure: replay first (copy, don't delete). Then quick seed 0 with Ask and the judge, lang
   -c seed=1 --no-ask, long-en `--rerun reduce` (ask first), and AMI (`--item ami --no-ask`;
   ask first, and nothing else on the GPU so s/h is real). Report AMI coverage by part and
   support, plus fact recall, made-up items and citation accuracy on the other sets, before
   and after.
3. Rewrite the baselines the fix moves on purpose, with the reason in the commit;
   baseline-full.json only from an AMI run (no full run without asking).
Rules from CLAUDE.md apply. Ask me before every full or long run, AMI runs, the opt-in model
tests and new dependencies (none should be needed). quick and lang runs are fine. The rest of
8g's "Open" (Map's 8 never-written numbers, xling-ja f9, the judge on slide-conflict notes,
long-en's 3 key points) and post-v1 items 5, 8 and 10 stay open unless step 0 shows they
share a cause. Don't bump the version: I'll ask separately. When done: tick the checklist,
add new decisions to §20 as D-numbers, update §5.6 and §18, add a NOTES.md entry and update
the post-v1 list, make sure the gate passes, and open a PR (ask before pushing).
```
