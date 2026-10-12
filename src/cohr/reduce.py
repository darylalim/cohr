"""Reduce (PLAN §5.6): Command R7B with the `rag` template and citations → notes.json.

Every segment and every slide is one document (Phase 0: with the map notes inside a segment's
document, R7B cited only the segment, so slides get their own documents to stay citable).
The reply uses fixed English section markers; citations `<co>span</co: 0:[i,j]>` are mapped
back to segments and slides. Invalid document indices are dropped, claims without a valid
citation are kept with `citations: []`, and segment citations get a refined jump time from
the best-matching transcript chunk.

Over `max_input_tokens`, two passes (§4): pass 1 is a normal cited reduce per group of
consecutive segments, written into a group summary with inline markers `[seg-N]` /
`[slide-N]`; pass 2 cites the group summaries, and each of its citations resolves back to
segments and slides (markers in the span, else the best-matching pass-1 claim, else every
segment of the group). Final citations never point at a group.
"""

import re
from collections import Counter
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
    NotesGroup,
    NotesSections,
    Segment,
    SegmentNotes,
    SegmentsFile,
    Slide,
    SlidesFile,
    Transcript,
    TranscriptItem,
    TwoPass,
)

SECTIONS = ("summary", "key_points", "decisions", "action_items", "open_questions")
ACTION_MATCH_MIN = 0.6  # token overlap for the map.json owner/due fallback (§5.6)
MIN_TRANSCRIPT_CHARS = 300  # below this per segment, trimming can't make the input fit
GROUP_FILL = 0.75  # D26: a pass-1 group's prompt fills up to this share of max_input_tokens
CLAIM_MATCH_MIN = 0.5  # pass-2 span → pass-1 claim token overlap, when the span has no marker

_WRAPPER = re.compile(r"<\|[A-Z_]+\|>")
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]*(.+?)[ \t]*$")
_CITE = re.compile(r"<co>(.*?)</co:\s*([^>]+)>", re.S)
_STRAY_TAG = re.compile(r"</?co\b[^>]*>")
_MARKER = re.compile(r"\s*\[(seg|slide)-(\d+)\]")  # two-pass group summaries (§4)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_NEW_ITEM = re.compile(r"\n(?=[ \t]*(?:[-*•]|\d+[.)])\s)")
_NEW_SENTENCE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])|\n")
_ACTION = re.compile(
    r"^(?:\[\s*owner\s*:\s*([^\]]*)\]\s*)?(?:\[\s*due\s*:\s*([^\]]*)\]\s*)?(.*)$", re.I | re.S
)
_SENTENCE = re.compile(r"[^.!?。！？]+(?:[.!?。！？]+|$)")
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
            # R7B sometimes cites the marker itself: "## <co>SUMMARY</co: 0:[0]>" (4b-1 eval).
            heading = _STRAY_TAG.sub("", m.group(1))
            name = heading.strip(" *:`").replace(" ", "_").casefold()
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
    text = _BULLET.sub("", _MARKER.sub("", _STRAY_TAG.sub("", text)))
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


def sentence_offset(span_tokens: set[str], text: str) -> float:
    """Where in `text` (0–1, by characters) the sentence best matching the span starts."""
    best_pos, best, pos = 0, 0.0, 0
    for m in _SENTENCE.finditer(text):
        score = overlap(span_tokens, set(tokens(m.group(0))))
        if score > best:
            best_pos, best = m.start() + len(m.group(0)) - len(m.group(0).lstrip()), score
        pos = m.end()
    return best_pos / max(pos, 1)


def refine_time(
    span: str, items: Sequence[TranscriptItem], seg_t0: float, min_score: float
) -> float:
    """Jump time for a cited span (§5.6, D19): the transcript chunk that best matches it (ties:
    earliest), then the start of its best-matching sentence, placed by character position
    (an even speaking rate; chunks are up to 30 s, so the chunk start alone was ~13 s early
    on the eval). The segment's start when no chunk scores ≥ min_score."""
    span_tokens = set(tokens(span))
    best_it, best = None, 0.0
    for it in items:
        score = overlap(span_tokens, set(tokens(it.text)))
        if score > best:
            best_it, best = it, score
    if best_it is None or best < min_score:
        return seg_t0
    offset = sentence_offset(span_tokens, best_it.text)
    return round(best_it.t0 + offset * (best_it.t1 - best_it.t0), 2)


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
    prompt_tokens: int = 0
    trimmed: int | None = None  # characters per segment, when transcripts were trimmed
    # two-pass only (§4)
    groups: int = 0
    pass1_uncited: int = 0
    resolved: Counter[str] = field(default_factory=Counter)


Resolver = Callable[[RawClaim], tuple[Claim, int]]  # → claim, dropped citation indices


def build_notes(
    reply: str,
    targets: Sequence[Target],
    refine: Refine,
    map_notes: Mapping[int, SegmentNotes],
    language: str,
) -> tuple[NotesDoc, ReduceStats]:
    return notes_from_reply(reply, lambda raw: resolve(raw, targets, refine), map_notes, language)


def notes_from_reply(
    reply: str, resolver: Resolver, map_notes: Mapping[int, SegmentNotes], language: str
) -> tuple[NotesDoc, ReduceStats]:
    split = split_sections(strip_wrapper(reply))
    stats = ReduceStats(unknown_markers=split.unknown, missing_markers=split.missing)
    out: dict[str, list[Any]] = {}
    for name in SECTIONS:
        section: list[Any] = []
        for raw in claims(split.sections[name], sentences=name == "summary"):
            claim, dropped = resolver(raw)
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


# --- two-pass (pure, given a model for token counts) ----------------------------------------


@dataclass
class Group:
    segments: list[Segment]
    slides: list[Slide]  # the slides aligned to its segments


def aligned_slides(segments: Sequence[Segment], slides: Sequence[Slide]) -> list[Slide]:
    ids = {i for seg in segments for i in seg.slide_ids}
    return [s for s in slides if s.slide_id in ids]


def make_groups(
    lm: Any,
    instruction: str,
    segments: Sequence[Segment],
    notes: Mapping[int, SegmentNotes],
    slides: Sequence[Slide],
    budget: int,
) -> list[Group]:
    """Consecutive segments whose pass-1 prompt (with their aligned slides) stays within
    `budget` tokens; a segment over it on its own is its own group (trimmed by fit_prompt)."""
    out: list[list[Segment]] = []
    current: list[Segment] = []
    for seg in segments:
        trial = [*current, seg]
        docs, _ = documents(trial, notes, aligned_slides(trial, slides))
        if current and lm.count_tokens(lm.prompt(instruction, docs)) > budget:
            out.append(current)
            current = [seg]
        else:
            current = trial
    if current:
        out.append(current)
    return [Group(g, aligned_slides(g, slides)) for g in out]


def marker(c: Citation) -> str:
    return f"[seg-{c.id}]" if c.kind == "seg" else f"[slide-{c.id + 1}]"


def _marked(text: str, claim: Claim) -> str:
    return " ".join([text, *(marker(c) for c in claim.citations)])


def group_title(group: Group) -> str:
    first, last = group.segments[0], group.segments[-1]
    title = f"{clock(first.t0)}–{clock(last.t1)}"
    if group.slides:
        title += " · slides " + ",".join(str(s.slide_id + 1) for s in group.slides)
    return title


def group_snippet(doc: NotesDoc) -> str:
    """Pass-1 notes as a pass-2 document: every claim followed by its citation markers."""
    s = doc.sections
    actions = [
        _marked(f"[owner: {a.owner or '?'}] [due: {a.due or '?'}] {a.text}", a)
        for a in s.action_items
    ]
    lines = [f"Summary: {' '.join(_marked(c.text, c) for c in s.summary)}"] if s.summary else []
    lines += _bullets("Key points", [_marked(c.text, c) for c in s.key_points])
    lines += _bullets("Decisions", [_marked(c.text, c) for c in s.decisions])
    lines += _bullets("Action items", actions)
    lines += _bullets("Open questions", [_marked(c.text, c) for c in s.open_questions])
    return "\n".join(lines)


@dataclass
class PassOne:
    """What a pass-2 group document cites: its segments and pass-1 claims."""

    seg_ids: list[int]
    claims: list[Claim]


def all_claims(doc: NotesDoc) -> list[Claim]:
    return [c for name in SECTIONS for c in getattr(doc.sections, name)]


def span_targets(span: str, group: PassOne, known: set[Target]) -> tuple[list[Target], str, int]:
    """Where a pass-2 span citing a group points (§4), how it resolved, and how many markers
    named no known segment or slide: the markers in the span; else the citations of the
    best-matching pass-1 claim (overlap ≥ CLAIM_MATCH_MIN, none if that claim is uncited);
    else every segment of the group (coarse but valid)."""
    marks: list[Target] = [
        ("seg", int(n)) if kind == "seg" else ("slide", int(n) - 1)
        for kind, n in _MARKER.findall(span)
    ]
    valid = [m for m in marks if m in known]
    if valid:
        return valid, "marker", len(marks) - len(valid)
    text = _MARKER.sub("", span)
    best, score = None, 0.0
    for claim in group.claims:
        s = _match_score(text, claim.text)
        if s > score:
            best, score = claim, s
    if best is not None and score >= CLAIM_MATCH_MIN:
        found: list[Target] = [(c.kind, c.id) for c in best.citations]
        return found, "claim" if found else "uncited", len(marks)
    return [("seg", i) for i in group.seg_ids], "group", len(marks)


Source = PassOne | Target  # what a pass-2 document index cites


def pass2_resolver(
    sources: Sequence[Source], known: set[Target], refine: Refine, resolved: Counter[str]
) -> Resolver:
    """resolve() for pass 2: group citations become segment and slide citations, refined
    against the resolved segments with the pass-2 span (§5.6); `resolved` counts the paths."""

    def resolver(raw: RawClaim) -> tuple[Claim, int]:
        citations: list[Citation] = []
        seen: set[Target] = set()
        dropped = 0
        for span, indices in raw.spans:
            text = _MARKER.sub("", span)
            for i in indices:
                if not 0 <= i < len(sources):
                    dropped += 1
                    continue
                src = sources[i]
                if isinstance(src, PassOne):
                    found, how, unknown = span_targets(span, src, known)
                    resolved[how] += 1
                    dropped += unknown
                else:
                    found = [src]
                for kind, target_id in found:
                    if (kind, target_id) in seen:
                        continue
                    seen.add((kind, target_id))
                    t = refine(target_id, text) if kind == "seg" else None
                    citations.append(Citation(kind=kind, id=target_id, t=t))
        return Claim(text=raw.text, citations=citations), dropped

    return resolver


Generate = Callable[[str], str]  # prompt → reply
Report = Callable[[float, str], None]  # progress (0–1), message


def two_pass(
    lm: Any,
    instructions: tuple[str, str],
    segments: Sequence[Segment],
    notes: Mapping[int, SegmentNotes],
    slides: Sequence[Slide],
    unaligned: Sequence[Slide],
    refine: Refine,
    max_tokens: int,
    generate: Generate,
    report: Report,
    language: str,
) -> tuple[NotesDoc, ReduceStats]:
    """Pass 1 per group of consecutive segments, pass 2 over the group summaries and the
    slides aligned to no segment (§4)."""
    pass1, pass2 = instructions
    groups = make_groups(lm, pass1, segments, notes, slides, int(max_tokens * GROUP_FILL))
    stats = ReduceStats(groups=len(groups))
    docs: list[dict[str, str]] = []
    sources: list[Source] = []
    kept: list[NotesGroup] = []
    for k, group in enumerate(groups):
        prompt, targets, size, limit = fit_prompt(
            lm, pass1, group.segments, notes, group.slides, max_tokens
        )
        if limit is not None:
            stats.trimmed = min(limit, stats.trimmed or limit)
        report(0.1 + 0.8 * k / len(groups), f"part {k + 1}/{len(groups)}, {size} tokens")
        reply = generate(prompt)
        doc, part = build_notes(reply, targets, refine, notes, language)
        stats.pass1_uncited += part.uncited
        stats.dropped += part.dropped
        docs.append({"title": group_title(group), "snippet": group_snippet(doc)})
        sources.append(PassOne([s.seg_id for s in group.segments], all_claims(doc)))
        kept.append(
            NotesGroup(
                seg_ids=[s.seg_id for s in group.segments],
                slide_ids=[s.slide_id for s in group.slides],
                raw=reply,
            )
        )
    for slide in unaligned:
        title = slide_label(slide.slide_id) + (f": {slide.title}" if slide.title else "")
        docs.append({"title": title, "snippet": slide_snippet(slide)})
        sources.append(("slide", slide.slide_id))
    prompt = lm.prompt(pass2, docs)
    size = lm.count_tokens(prompt)
    if size > max_tokens:
        raise ReduceTooLong(
            f"the {len(groups)} part summaries are {size} tokens, over the {max_tokens} cap"
        )
    report(0.9, f"{len(docs)} part summaries and slides, {size} tokens")
    reply = generate(prompt)
    known: set[Target] = {("seg", s.seg_id) for s in segments}
    known |= {("slide", s.slide_id) for s in [*slides, *unaligned]}
    resolved: Counter[str] = Counter()
    doc, final = notes_from_reply(
        reply, pass2_resolver(sources, known, refine, resolved), notes, language
    )
    doc = doc.model_copy(
        update={"passes": 2, "two_pass": TwoPass(groups=kept, resolved=dict(resolved))}
    )
    final.groups, final.trimmed = stats.groups, stats.trimmed
    final.pass1_uncited, final.dropped = stats.pass1_uncited, final.dropped + stats.dropped
    final.resolved = resolved
    final.prompt_tokens = size
    return doc, final


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
    """The prompt within max_tokens: transcripts are trimmed (25% at a time) when the
    documents don't fit. Reduce trims only a two-pass group that's over the cap on its own
    (one very long segment); Ask trims the whole meeting until ranking (§5.7)."""
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
                f"transcripts trimmed to {MIN_TRANSCRIPT_CHARS} characters per segment"
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
    name = language_name(language)
    instruction = fill(load_prompt("reduce"), target_language=name)
    refine = segment_refiner(segments, items, cfg.citation_min_score)

    def generate(prompt: str) -> str:
        return lm.generate(
            prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature, seed=ctx.config.seed
        )

    with ctx.model("reduce") as lm:
        docs, targets = documents(segments, map_notes, slides)
        prompt = lm.prompt(instruction, docs)
        size = lm.count_tokens(prompt)
        if size <= cfg.max_input_tokens:
            ctx.report(0.1, f"{len(targets)} documents, {size} tokens")
            doc, stats = build_notes(generate(prompt), targets, refine, map_notes, language)
            stats.prompt_tokens = size
        else:
            ctx.log.info(
                "reduce: %d tokens over the %d cap, two passes", size, cfg.max_input_tokens
            )
            in_segments = aligned_slides(segments, slides)
            unaligned = [s for s in slides if s not in in_segments]
            doc, stats = two_pass(
                lm,
                (instruction, fill(load_prompt("reduce_pass2"), target_language=name)),
                segments,
                map_notes,
                in_segments,
                unaligned,
                refine,
                cfg.max_input_tokens,
                generate,
                ctx.report,
                language,
            )
    if stats.trimmed is not None:
        ctx.log.warning("reduce: transcripts trimmed to %d characters per segment", stats.trimmed)
    if stats.groups:
        ctx.log.info(
            "reduce: two passes, %d groups, %d uncited in pass 1, pass-2 citations %s",
            stats.groups,
            stats.pass1_uncited,
            dict(stats.resolved),
        )
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
        len(doc.two_pass.groups) if doc.two_pass else len(targets),
        stats.prompt_tokens,
    )
    schemas.save(ctx.run_dir, doc)


STAGE = Stage(
    name="reduce",
    run=run_stage,
    outputs=(NotesDoc.FILENAME,),
    format_version=NotesDoc.FORMAT_VERSION,
    config_keys=("seed", "target_language", "reduce", "models.reduce"),
    prompts=("reduce", "reduce_pass2"),
    models=("reduce",),
)
