"""Reduce (PLAN §5.6): Command R7B with the `rag` template and citations → notes.json.

Every segment and every slide is one document (Phase 0: with the map notes inside a segment's
document, R7B cited only the segment, so slides get their own documents to stay citable).
The reply uses fixed English section markers; citations `<co>span</co: 0:[i,j]>` are mapped
back to segments and slides. Invalid document indices are dropped, claims without a valid
citation are kept with `citations: []`, and segment citations get a refined jump time from
the best-matching transcript chunk.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cohr import schemas
from cohr.config import language_name, load_prompt
from cohr.fuse import overlap, slide_text, tokens
from cohr.mapper import clock, fill
from cohr.pipeline import Stage, StageContext
from cohr.schemas import (
    ActionItem,
    Citation,
    Claim,
    MapActionItem,
    MapFile,
    NotesDoc,
    NotesSections,
    Segment,
    SegmentNotes,
    SegmentsFile,
    Slide,
    SlidesFile,
    Transcript,
    TranscriptItem,
)

SECTIONS = ("summary", "key_points", "decisions", "action_items", "open_questions")
ACTION_MATCH_MIN = 0.6  # token overlap for the map.json owner/due fallback (§5.6)
MIN_TRANSCRIPT_CHARS = 300  # below this per segment, trimming can't make the input fit

_WRAPPER = re.compile(r"<\|[A-Z_]+\|>")
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]*(.+?)[ \t]*$")
_CITE = re.compile(r"<co>(.*?)</co:\s*([^>]+)>", re.S)
_STRAY_TAG = re.compile(r"</?co\b[^>]*>")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_NEW_ITEM = re.compile(r"\n(?=[ \t]*(?:[-*•]|\d+[.)])\s)")
_NEW_SENTENCE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])|\n")
_ACTION = re.compile(
    r"^(?:\[\s*owner\s*:\s*([^\]]*)\]\s*)?(?:\[\s*due\s*:\s*([^\]]*)\]\s*)?(.*)$", re.I | re.S
)
_NONE = {"none", "n/a", "-", "—", "nothing"}
_UNKNOWN = {"", "?", "none", "null", "unknown", "n/a", "-"}


class ReduceTooLong(RuntimeError):
    pass


Target = tuple[Literal["seg", "slide"], int]


# --- documents (pure) ----------------------------------------------------------------------


def slide_label(slide_id: int) -> str:
    return f"slide {slide_id + 1}"


def segment_title(seg: Segment) -> str:
    title = f"{clock(seg.t0)}–{clock(seg.t1)}"
    if seg.slide_ids:
        title += " · slides " + ",".join(str(i + 1) for i in seg.slide_ids)
    return title


def _bullets(label: str, items: Sequence[str]) -> list[str]:
    return [f"{label}:", *(f"- {i}" for i in items)] if items else []


def _action_line(a: MapActionItem) -> str:
    return f"[owner: {a.owner or '?'}] [due: {a.due or '?'}] {a.task}"


def segment_snippet(notes: SegmentNotes | None, transcript: str, max_chars: int | None) -> str:
    lines: list[str] = []
    if notes is not None:
        lines += [f"Summary: {notes.summary}"] if notes.summary else []
        lines += _bullets("Key points", notes.key_points)
        lines += _bullets("Decisions", notes.decisions)
        lines += _bullets("Action items", [_action_line(a) for a in notes.action_items])
        lines += _bullets("Open questions", notes.open_questions)
    if max_chars is not None and len(transcript) > max_chars:
        transcript = transcript[:max_chars].rsplit(" ", 1)[0] + " …"
    lines += ["", f"Transcript: {transcript}"]
    return "\n".join(lines).strip()


def slide_snippet(slide: Slide) -> str:
    return "\n".join(p for p in (slide_text(slide), slide.chart_notes) if p)


def documents(
    segments: Sequence[Segment],
    notes: Mapping[int, SegmentNotes],
    slides: Sequence[Slide],
    max_transcript_chars: int | None = None,
) -> tuple[list[dict[str, str]], list[Target]]:
    """The `rag` documents (segments, then slides) and what each index cites."""
    docs, targets = [], []
    for seg in segments:
        snippet = segment_snippet(notes.get(seg.seg_id), seg.transcript_text, max_transcript_chars)
        docs.append({"title": segment_title(seg), "snippet": snippet})
        targets.append(("seg", seg.seg_id))
    for slide in slides:
        title = slide_label(slide.slide_id) + (f": {slide.title}" if slide.title else "")
        docs.append({"title": title, "snippet": slide_snippet(slide)})
        targets.append(("slide", slide.slide_id))
    return docs, targets


# --- reply parsing (pure) ------------------------------------------------------------------


def strip_wrapper(reply: str) -> str:
    """The text between <|START_RESPONSE|> and <|END_RESPONSE|>, special tokens removed."""
    if "<|START_RESPONSE|>" in reply:
        reply = reply.split("<|START_RESPONSE|>", 1)[1]
    return _WRAPPER.sub("", reply.split("<|END_RESPONSE|>", 1)[0])


@dataclass
class SplitReply:
    sections: dict[str, str]
    unknown: list[str] = field(default_factory=list)  # headings that aren't one of SECTIONS
    missing: list[str] = field(default_factory=list)


def split_sections(text: str) -> SplitReply:
    """Split on the fixed English markers. Text before the first marker and under an unknown
    heading goes to the summary; localized headings are never parsed."""
    parts: dict[str, list[str]] = {s: [] for s in SECTIONS}
    unknown: list[str] = []
    seen: set[str] = set()
    current = "summary"
    for line in text.splitlines():
        if m := _HEADING.match(line):
            name = m.group(1).strip(" *:`").replace(" ", "_").casefold()
            if name in parts:
                current = name
                seen.add(name)
            else:
                unknown.append(m.group(1))
                current = "summary"
            continue
        parts[current].append(line)
    return SplitReply(
        sections={k: "\n".join(v).strip() for k, v in parts.items()},
        unknown=unknown,
        missing=[s for s in SECTIONS if s not in seen],
    )


def doc_indices(ref: str) -> list[int]:
    """`0:[1,2]` → [1, 2] (the leading 0 is the tool call, not a document)."""
    groups = re.findall(r"\[([^\]]*)\]", ref)
    found = groups if groups else [ref.split(":", 1)[-1]]
    return [int(i) for g in found for i in re.findall(r"\d+", g)]


Piece = tuple[str, list[int] | None]  # text, and the cited document indices for a <co> span


def pieces(text: str) -> list[Piece]:
    out: list[Piece] = []
    pos = 0
    for m in _CITE.finditer(text):
        if m.start() > pos:
            out.append((text[pos : m.start()], None))
        out.append((m.group(1), doc_indices(m.group(2))))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], None))
    return out


@dataclass
class RawClaim:
    text: str
    spans: list[tuple[str, list[int]]]  # cited span text, document indices


def _clean(text: str) -> str:
    text = _BULLET.sub("", _STRAY_TAG.sub("", text))
    return " ".join(text.split())


def claims(text: str, *, sentences: bool) -> list[RawClaim]:
    """One claim per list item, or per sentence (summary). A <co> span is never split, so its
    citation stays with the claim that contains it. "- none" items are skipped."""
    boundary = _NEW_SENTENCE if sentences else _NEW_ITEM
    out: list[RawClaim] = []
    buf = ""
    spans: list[tuple[str, list[int]]] = []

    def flush() -> None:
        nonlocal buf, spans
        cleaned = _clean(buf)
        if cleaned and cleaned.strip(" .").casefold() not in _NONE:
            out.append(RawClaim(cleaned, spans))
        buf, spans = "", []

    for piece, indices in pieces(text):
        if indices is not None:
            buf += piece
            spans.append((piece, indices))
            continue
        if (
            sentences
            and buf
            and (
                (buf.rstrip() == buf and buf[-1:] in ".!?" and piece[:1].isspace())
                or buf[-1:] in "。！？"
            )
        ):
            flush()
        pos = 0
        for m in boundary.finditer(piece):
            buf += piece[pos : m.start()]
            flush()
            pos = m.end()
        buf += piece[pos:]
    flush()
    return out


# --- citations (pure) ----------------------------------------------------------------------


def refine_time(
    span: str, items: Sequence[TranscriptItem], seg_t0: float, min_score: float
) -> float:
    """t0 of the transcript chunk that best matches the cited span (ties: earliest), or the
    segment's start when no chunk scores ≥ min_score (§5.6, D19)."""
    span_tokens = set(tokens(span))
    best_t, best = seg_t0, 0.0
    for it in items:
        score = overlap(span_tokens, set(tokens(it.text)))
        if score > best:
            best_t, best = it.t0, score
    return best_t if best >= min_score else seg_t0


Refine = Callable[[int, str], float]  # (seg_id, span) → jump time


def resolve(raw: RawClaim, targets: Sequence[Target], refine: Refine) -> tuple[Claim, int]:
    """The claim with its valid citations (first occurrence of each target), and how many
    out-of-range indices were dropped."""
    citations: list[Citation] = []
    seen: set[Target] = set()
    dropped = 0
    for span, indices in raw.spans:
        for i in indices:
            if not 0 <= i < len(targets):
                dropped += 1
                continue
            kind, target_id = targets[i]
            if (kind, target_id) in seen:
                continue
            seen.add((kind, target_id))
            t = refine(target_id, span) if kind == "seg" else None
            citations.append(Citation(kind=kind, id=target_id, t=t))
    return Claim(text=raw.text, citations=citations), dropped


def _known(value: str | None) -> str | None:
    value = (value or "").strip()
    return None if value.casefold() in _UNKNOWN else value


def _match_score(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    return len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0


def action_item(claim: Claim, map_notes: Mapping[int, SegmentNotes]) -> ActionItem:
    """Owner and due from `[owner: …] [due: …] task`; if the line doesn't have them, copied
    from the best-matching map.json action item of a cited segment (overlap ≥ 0.6)."""
    m = _ACTION.match(claim.text)
    if m and (m.group(1) is not None or m.group(2) is not None):
        task = m.group(3).strip() or claim.text
        return ActionItem(
            text=task, citations=claim.citations, owner=_known(m.group(1)), due=_known(m.group(2))
        )
    candidates = [
        a
        for c in claim.citations
        if c.kind == "seg" and c.id in map_notes
        for a in map_notes[c.id].action_items
    ]
    best, score = None, 0.0
    for a in candidates:
        s = _match_score(claim.text, a.task)
        if s > score:
            best, score = a, s
    owner, due = (best.owner, best.due) if best and score >= ACTION_MATCH_MIN else (None, None)
    return ActionItem(text=claim.text, citations=claim.citations, owner=owner, due=due)


@dataclass
class ReduceStats:
    claims: int = 0
    uncited: int = 0
    dropped: int = 0
    unknown_markers: list[str] = field(default_factory=list)
    missing_markers: list[str] = field(default_factory=list)


def build_notes(
    reply: str,
    targets: Sequence[Target],
    refine: Refine,
    map_notes: Mapping[int, SegmentNotes],
    language: str,
) -> tuple[NotesDoc, ReduceStats]:
    split = split_sections(strip_wrapper(reply))
    stats = ReduceStats(unknown_markers=split.unknown, missing_markers=split.missing)
    out: dict[str, list[Any]] = {}
    for name in SECTIONS:
        section: list[Any] = []
        for raw in claims(split.sections[name], sentences=name == "summary"):
            claim, dropped = resolve(raw, targets, refine)
            stats.claims += 1
            stats.dropped += dropped
            stats.uncited += not claim.citations
            section.append(action_item(claim, map_notes) if name == "action_items" else claim)
        out[name] = section
    doc = NotesDoc(sections=NotesSections(**out), passes=1, language=language, raw=reply)
    return doc, stats


def segment_refiner(
    segments: Sequence[Segment], items: Sequence[TranscriptItem], min_score: float
) -> Refine:
    by_seg = {s.seg_id: s for s in segments}
    by_chunk = {it.chunk_id: it for it in items}

    def refine(seg_id: int, span: str) -> float:
        seg = by_seg[seg_id]
        chunks = [by_chunk[c] for c in seg.chunk_ids if c in by_chunk]
        return refine_time(span, chunks, seg.t0, min_score)

    return refine


# --- model (MLX) ---------------------------------------------------------------------------


class CommandR7B:
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def prompt(self, instruction: str, docs: list[dict[str, str]]) -> str:
        """The `rag` chat template on the underlying HF tokenizer (Phase 0, D7)."""
        hf = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        return hf.apply_chat_template(
            [{"role": "user", "content": instruction}],
            documents=docs,
            chat_template="rag",
            enable_citations=True,
            tokenize=False,
            add_generation_prompt=True,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        return generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        )


def load(path: Path) -> CommandR7B:
    from mlx_lm import load as load_lm

    loaded = load_lm(str(path))
    return CommandR7B(loaded[0], loaded[1])


# --- stage ---------------------------------------------------------------------------------


def fit_prompt(
    lm: Any,
    instruction: str,
    segments: Sequence[Segment],
    notes: Mapping[int, SegmentNotes],
    slides: Sequence[Slide],
    max_tokens: int,
) -> tuple[str, list[Target], int, int | None]:
    """The prompt within max_tokens: transcripts are trimmed (25% at a time) when the whole
    meeting doesn't fit. Two-pass reduce (Phase 6) replaces this for inputs over the cap."""
    limit: int | None = None
    while True:
        docs, targets = documents(segments, notes, slides, limit)
        prompt = lm.prompt(instruction, docs)
        size = lm.count_tokens(prompt)
        if size <= max_tokens:
            return prompt, targets, size, limit
        longest = max((len(s.transcript_text) for s in segments), default=0)
        limit = int((limit if limit is not None else longest) * 0.75)
        if limit < MIN_TRANSCRIPT_CHARS:
            raise ReduceTooLong(
                f"the notes input is {size} tokens, over the {max_tokens} cap even with "
                "trimmed transcripts; two-pass reduce for long meetings is not built yet"
            )


def run_stage(ctx: StageContext) -> None:
    cfg = ctx.config.reduce
    language = ctx.config.target_language
    segments = schemas.load(ctx.run_dir, SegmentsFile).segments
    slides = schemas.load(ctx.run_dir, SlidesFile).slides
    map_notes = {n.seg_id: n for n in schemas.load(ctx.run_dir, MapFile).notes}
    items = schemas.load(ctx.run_dir, Transcript).items
    if not segments and not slides:
        ctx.log.warning("reduce: no speech and no slides, writing empty notes")
        empty = NotesSections(
            summary=[], key_points=[], decisions=[], action_items=[], open_questions=[]
        )
        schemas.save(ctx.run_dir, NotesDoc(sections=empty, passes=1, language=language, raw=""))
        return
    instruction = fill(load_prompt("reduce"), target_language=language_name(language))
    with ctx.model("reduce") as lm:
        prompt, targets, size, limit = fit_prompt(
            lm, instruction, segments, map_notes, slides, cfg.max_input_tokens
        )
        if limit is not None:
            ctx.log.warning("reduce: transcripts trimmed to %d characters per segment", limit)
        ctx.report(0.1, f"{len(targets)} documents, {size} tokens")
        reply = lm.generate(
            prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature, seed=ctx.config.seed
        )
    refine = segment_refiner(segments, items, cfg.citation_min_score)
    doc, stats = build_notes(reply, targets, refine, map_notes, language)
    if stats.unknown_markers:
        ctx.log.warning(
            "reduce: unknown section markers %s (text kept in summary)", stats.unknown_markers
        )
    if stats.missing_markers:
        ctx.log.warning("reduce: missing section markers %s", stats.missing_markers)
    if stats.dropped:
        ctx.log.warning("reduce: dropped %d citations to unknown documents", stats.dropped)
    ctx.log.info(
        "reduce: %d claims, %d uncited, %d documents, %d prompt tokens",
        stats.claims,
        stats.uncited,
        len(targets),
        size,
    )
    schemas.save(ctx.run_dir, doc)


STAGE = Stage(
    name="reduce",
    run=run_stage,
    outputs=(NotesDoc.FILENAME,),
    format_version=NotesDoc.FORMAT_VERSION,
    config_keys=("seed", "target_language", "reduce", "models.reduce"),
    prompts=("reduce",),
    models=("reduce",),
)
