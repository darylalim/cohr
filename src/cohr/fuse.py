"""Fuse (PLAN §5.4): slide text alignment, transcript windows, token budget. No models.

Slides without a timestamp (PDF or image decks) are placed on the transcript by token overlap,
with a monotonic assignment (decks are presented in order; slides may be skipped). Aligned
slides get `t`, `t_source="aligned"` and `align_score` written back to slides.json; the rest
stay unaligned. Slides with a video timestamp (`pts`) are never moved.

Transcript chunks are grouped into windows of `window_s`; a segment whose transcript + slide
text is over `max_segment_tokens` is split in half until it fits (or is a single chunk).
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

from cohr import schemas
from cohr.config import FuseConfig
from cohr.pipeline import Stage, StageContext
from cohr.schemas import Segment, SegmentsFile, Slide, SlidesFile, Transcript, TranscriptItem

# A score's denominator is never below this many tokens, so a two-word chunk that happens to
# share one word with a slide doesn't score 0.5.
MIN_DENOMINATOR = 8
# With at least this many chunks, tokens found in more than half of them carry no position
# signal (function words in any language, the project name) and are dropped.
DF_MIN_CHUNKS = 4

_STOPWORDS = """
a about above after again all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from
further had has have having he her here hers him his how i if in into is it its itself
just let lets me more most my no nor not now of off on once only or other our ours out
over own same she should so some such than that thats the their theirs them then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yes okay ok um uh
"""
STOPWORDS = frozenset(_STOPWORDS.split())  # English only; DF_MIN_CHUNKS covers the rest

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|\w+")


# --- tokens and scores (pure) --------------------------------------------------------------


def tokens(text: str) -> list[str]:
    """Casefolded content tokens. Numbers keep their digits ("$14,500" → "14500"); Chinese and
    Japanese runs become character bigrams, since they have no spaces between words."""
    out: list[str] = []
    for tok in _TOKEN.findall(_CJK.sub(lambda m: f" {m.group(0)} ", text.casefold())):
        if tok[0].isdigit():
            out.append(tok.replace(",", "").replace(".", ""))
        elif _CJK.fullmatch(tok):
            out += [tok[i : i + 2] for i in range(max(1, len(tok) - 1))]
        elif len(tok) > 1 and tok not in STOPWORDS:
            out.append(tok)
    return out


def slide_text(slide: Slide) -> str:
    """The slide text that alignment scores and Map reads: title, text and tables."""
    return "\n".join(p for p in (slide.title, slide.text_md, *slide.tables_md) if p)


def overlap(a: set[str], b: set[str]) -> float:
    """Shared tokens over the smaller set (at least MIN_DENOMINATOR)."""
    if not a or not b:
        return 0.0
    return len(a & b) / max(min(len(a), len(b)), MIN_DENOMINATOR)


def score_matrix(slides: Sequence[str], chunks: Sequence[str]) -> list[list[float]]:
    """scores[i][j]: slide i against chunk j, after dropping tokens common to most chunks."""
    chunk_sets = [set(tokens(c)) for c in chunks]
    common: set[str] = set()
    if len(chunk_sets) >= DF_MIN_CHUNKS:
        df = Counter(t for s in chunk_sets for t in s)
        common = {t for t, n in df.items() if n > len(chunk_sets) / 2}
    chunk_sets = [s - common for s in chunk_sets]
    return [[overlap(set(tokens(s)) - common, c) for c in chunk_sets] for s in slides]


def align(scores: Sequence[Sequence[float]], threshold: float) -> list[int | None]:
    """Monotonic slide → chunk assignment maximizing the total score (dynamic programming).

    Slides keep their order (later slides never land on earlier chunks; two slides may share a
    chunk), any slide may be skipped, and a slide is only placed on a chunk scoring ≥ threshold.
    Ties go to the earliest chunk. Returns the chunk index per slide, None for unaligned.
    """
    m = len(scores)
    n = len(scores[0]) if m else 0
    if n == 0:
        return [None] * m
    # best[i][j]: best total for the first i slides using only chunks < j.
    best = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        row, prev = best[i], best[i - 1]
        for j in range(1, n + 1):
            s = scores[i - 1][j - 1]
            place = prev[j] + s if s > 0 and s >= threshold else -math.inf
            row[j] = max(row[j - 1], prev[j], place)
    out: list[int | None] = [None] * m
    i, j = m, n
    while i > 0 and j > 0:
        if best[i][j] == best[i][j - 1]:
            j -= 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            out[i - 1] = j - 1
            i -= 1
    return out


def align_slides(
    slides: list[Slide], items: Sequence[TranscriptItem], threshold: float
) -> list[Slide]:
    """Place slides that have no video timestamp. Earlier alignments are recomputed, so
    re-running Fuse with another threshold gives the same result as a first run."""
    todo = [k for k, s in enumerate(slides) if s.t_source != "pts"]
    scores = score_matrix([slide_text(slides[k]) for k in todo], [it.text for it in items])
    placed = align(scores, threshold)
    out = list(slides)
    for row, k, j in zip(scores, todo, placed, strict=True):
        timing = (
            {"t": None, "t_source": "none", "align_score": None}
            if j is None
            else {"t": items[j].t0, "t_source": "aligned", "align_score": round(row[j], 4)}
        )
        out[k] = slides[k].model_copy(update=timing)
    return out


def estimate_tokens(text: str) -> int:
    """Conservative token count without a tokenizer: one per CJK character, one per three
    other characters (Tiny Aya's tokenizer averages ~4 characters per token in English)."""
    cjk = sum(len(m) for m in _CJK.findall(text))
    return cjk + math.ceil((len(text) - cjk) / 3)


# --- segments (pure) -----------------------------------------------------------------------


def windows(items: Sequence[TranscriptItem], window_s: float) -> list[list[TranscriptItem]]:
    """Consecutive chunks, a new window whenever adding a chunk would span more than window_s."""
    out: list[list[TranscriptItem]] = []
    for it in items:
        if out and it.t1 - out[-1][0].t0 <= window_s:
            out[-1].append(it)
        else:
            out.append([it])
    return out


def attach(groups: Sequence[Sequence[TranscriptItem]], slides: Sequence[Slide]) -> list[list[int]]:
    """Slide ids per group: every timed slide shown in the group's stretch of time (from the end
    of the previous group to its own end; the first and last stretch are open-ended), plus the
    slide still on screen when the group starts."""
    timed = sorted((s.t, s.slide_id) for s in slides if s.t is not None)
    out: list[list[int]] = []
    for g, group in enumerate(groups):
        lo = groups[g - 1][-1].t1 if g > 0 else -math.inf
        hi = group[-1].t1 if g < len(groups) - 1 else math.inf
        ids = [i for t, i in timed if lo < t <= hi]
        on_screen = [i for t, i in timed if t <= group[0].t0]
        if on_screen and on_screen[-1] not in ids:
            ids.insert(0, on_screen[-1])
        out.append(ids)
    return out


def group_text(group: Sequence[TranscriptItem]) -> str:
    return "\n".join(it.text for it in group if it.text)


def build_segments(
    items: Sequence[TranscriptItem], slides: Sequence[Slide], cfg: FuseConfig
) -> tuple[list[Segment], list[str]]:
    """Segments within the token budget, and a warning per segment that can't be split further."""
    by_id = {s.slide_id: s for s in slides}
    groups = windows(items, cfg.window_s)
    while True:
        attached = attach(groups, slides)
        sizes = [
            estimate_tokens(group_text(g) + "".join(slide_text(by_id[i]) for i in ids))
            for g, ids in zip(groups, attached, strict=True)
        ]
        over = [k for k, n in enumerate(sizes) if n > cfg.max_segment_tokens and len(groups[k]) > 1]
        if not over:
            break
        split: list[list[TranscriptItem]] = []
        for k, g in enumerate(groups):
            if k in over:
                split += [g[: len(g) // 2], g[len(g) // 2 :]]
            else:
                split.append(g)
        groups = split
    warnings = [
        f"segment {k}: ~{n} tokens is over the {cfg.max_segment_tokens} budget (one chunk)"
        for k, n in enumerate(sizes)
        if n > cfg.max_segment_tokens
    ]
    segments = [
        Segment(
            seg_id=k,
            t0=g[0].t0,
            t1=g[-1].t1,
            chunk_ids=[it.chunk_id for it in g],
            transcript_text=group_text(g),
            slide_ids=ids,
        )
        for k, (g, ids) in enumerate(zip(groups, attached, strict=True))
    ]
    return segments, warnings


# --- stage ---------------------------------------------------------------------------------


def run_stage(ctx: StageContext) -> None:
    cfg = ctx.config.fuse
    items = schemas.load(ctx.run_dir, Transcript).items
    slides = align_slides(schemas.load(ctx.run_dir, SlidesFile).slides, items, cfg.align_threshold)
    schemas.save(ctx.run_dir, SlidesFile(slides=slides))
    segments, warnings = build_segments(items, slides, cfg)
    for w in warnings:
        ctx.log.warning("fuse: %s", w)
    attached = {i for seg in segments for i in seg.slide_ids}
    unaligned = [s.slide_id for s in slides if s.slide_id not in attached]
    aligned = sum(s.t_source == "aligned" for s in slides)
    ctx.log.info(
        "fuse: %d segments, %d slides text-aligned, %d unaligned",
        len(segments),
        aligned,
        len(unaligned),
    )
    schemas.save(ctx.run_dir, SegmentsFile(segments=segments, unaligned_slide_ids=unaligned))


STAGE = Stage(
    name="fuse",
    run=run_stage,
    outputs=(SegmentsFile.FILENAME,),
    format_version=SegmentsFile.FORMAT_VERSION,
    config_keys=("fuse",),
)
