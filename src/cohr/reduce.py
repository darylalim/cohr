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
from typing import Any, Literal, NamedTuple

from cohr import dues, names, numbers, schemas
from cohr.config import language_name, load_prompt
from cohr.fts import fold_digits
from cohr.fuse import match_score, overlap, slide_text, tokens
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
# D26: a pass-1 group's prompt fills up to this share of max_input_tokens. 0.75 (the plan's
# default) gave long-en 2 groups of ~23K tokens whose notes ran past the output budget; 0.4
# gives ~12K-token groups that finish (recall 37.5 → 53.8%, 6c).
GROUP_FILL = 0.4
CLAIM_MATCH_MIN = 0.5  # pass-2 span → pass-1 claim token overlap, when the span has no marker
END_RESPONSE = "<|END_RESPONSE|>"
# The reply is started with the first marker (6d-1): in German and Arabic notes R7B translated
# all five markers ("## Zusammenfassung", "## ملخص") whatever the prompt said, so every claim
# fell into Summary. Started this way it writes the other four in English too (de, ar, seeds
# 1–3) and keeps its citations. Reduce only: Ask's replies have no sections (D22).
REPLY_START = "<|START_RESPONSE|>## SUMMARY\n"

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
_UNKNOWN = {
    "",
    "?",
    "none",
    "null",
    "unknown",
    "n/a",
    "-",
    "not specified",  # AMI: Map's placeholder owner/due (8b)
    "unspecified",
    "not stated",
    "not mentioned",
    "tbd",
}
# A task that only says to keep doing something (D40, 8b): moved to Key points when it has no
# owner and no due.
_VAGUE_TASK = re.compile(r"(?:continue|ensure|keep|monitor|maintain)\b", re.I)


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
    return _WRAPPER.sub("", reply.split(END_RESPONSE, 1)[0])


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


class Anchors(NamedTuple):
    """What a span says in any language (D39): its numbers and an action item's owner."""

    values: set[float]
    owner: str | None

    def hits(self, text: str, source_language: str) -> int:
        said = numbers.source_numbers_in(fold_digits(text), source_language)
        owner = bool(self.owner and names.said_folded(self.owner, text))
        return len(self.values & said) + owner


def span_anchors(span: str, items: Sequence[TranscriptItem], notes_language: str) -> Anchors:
    """The span's numbers read in the notes' language (other scripts' digits folded) and the
    `[owner: …]` it starts with, in the meeting's spelling when the segment says it (D34:
    per chunk, मारिया matched Omar, the only m-r name there). Names elsewhere in the text
    aren't anchors: their consonants match common words (Arabic تم "was done" = Tom)."""
    m = _ACTION.match(span.strip())
    owner = _known(m.group(1)) if m else None
    if owner and not names.latin(owner):
        owner = names.restore(owner, [it.text for it in items])
    return Anchors(numbers.notes_numbers(fold_digits(span), notes_language), owner)


def anchor_offset(anchors: Anchors, text: str, source_language: str) -> float:
    """Where in `text` (0–1) the sentence with the most anchors starts (ties: earliest)."""
    best_pos, best, pos = 0, 0, 0
    for m in _SENTENCE.finditer(text):
        hits = anchors.hits(m.group(0), source_language)
        if hits > best:
            best_pos, best = m.start() + len(m.group(0)) - len(m.group(0).lstrip()), hits
        pos = m.end()
    return best_pos / max(pos, 1)


def refine_time(
    span: str,
    items: Sequence[TranscriptItem],
    seg_t0: float,
    min_score: float,
    notes_language: str = "en",
    source_language: str = "en",
) -> float:
    """Jump time for a cited span (§5.6, D19): the transcript chunk that best matches it (ties:
    earliest), then the start of its best-matching sentence, placed by character position
    (an even speaking rate; chunks are up to 30 s, so the chunk start alone was ~13 s early
    on the eval). When no chunk scores ≥ min_score (notes and meeting in different
    languages share no words), the chunk with the most of the span's anchors (D39; ties:
    word overlap, then earliest); the segment's start when no chunk has one."""
    span_tokens = set(tokens(span))
    scores = [overlap(span_tokens, set(tokens(it.text))) for it in items]
    best = max(scores, default=0.0)
    if best > 0 and best >= min_score:
        best_it = items[scores.index(best)]
        offset = sentence_offset(span_tokens, best_it.text)
        return round(best_it.t0 + offset * (best_it.t1 - best_it.t0), 2)
    anchors = span_anchors(span, items, notes_language)
    if not anchors.values and not anchors.owner:
        return seg_t0
    ranked = [
        (anchors.hits(it.text, source_language), s, -i)
        for i, (it, s) in enumerate(zip(items, scores, strict=True))
    ]
    hits, _, index = max(ranked, default=(0, 0.0, 0))
    if not hits:
        return seg_t0
    best_it = items[-index]
    offset = anchor_offset(anchors, best_it.text, source_language)
    return round(best_it.t0 + offset * (best_it.t1 - best_it.t0), 2)


Refine = Callable[[int, Sequence[str]], float]  # (seg_id, spans citing it) → jump time


def best_span(spans: Sequence[str], items: Sequence[TranscriptItem]) -> str:
    """The span that matches one of the segment's chunks best (ties: the first). R7B often
    cites a segment through a name alone and then the task (`<co>Tom</co>` needs to
    `<co>finish the post mortem…</co>`), and the name's first chunk can be minutes off (8c);
    `overlap`'s minimum denominator keeps a one-word span's score low."""
    if len(spans) == 1:
        return spans[0]
    chunk_tokens = [set(tokens(it.text)) for it in items]
    scores = [max((overlap(set(tokens(s)), c) for c in chunk_tokens), default=0.0) for s in spans]
    return spans[scores.index(max(scores))]


def cite(found: Sequence[tuple[Target, str]], refine: Refine) -> list[Citation]:
    """Citations in first-occurrence order; a segment's `t` from all the spans citing it."""
    spans: dict[Target, list[str]] = {}
    for target, span in found:
        spans.setdefault(target, []).append(span)
    return [
        Citation(kind=kind, id=target_id, t=refine(target_id, ss) if kind == "seg" else None)
        for (kind, target_id), ss in spans.items()
    ]


def resolve(raw: RawClaim, targets: Sequence[Target], refine: Refine) -> tuple[Claim, int]:
    """The claim with its valid citations (first occurrence of each target), and how many
    out-of-range indices were dropped."""
    found: list[tuple[Target, str]] = []
    dropped = 0
    for span, indices in raw.spans:
        for i in indices:
            if not 0 <= i < len(targets):
                dropped += 1
                continue
            found.append((targets[i], span))
    return Claim(text=raw.text, citations=cite(found, refine)), dropped


def _known(value: str | None) -> str | None:
    value = (value or "").strip()
    return None if value.casefold() in _UNKNOWN else value


def filed_task(claim: Claim, map_notes: Mapping[int, SegmentNotes]) -> bool:
    """A decision that is one of Map's tasks: it matches an action item of a cited segment at
    ACTION_MATCH_MIN and better than any of those segments' Map decisions (AMI ES2004b: "The
    project will explore …", 6 of 15 decisions), or as well when that action item names an
    owner (Map lists some tasks under both: "Laura to send contract to legal by Friday" in
    clean-en, 4 of 7 decisions). Without an owner a tie stays a decision: Map also writes
    decided designs as tasks ("Design the single curved side remote control", AMI TS3003c)."""
    cited = [map_notes[c.id] for c in claim.citations if c.kind == "seg" and c.id in map_notes]
    task, owned = 0.0, False
    for a in (a for n in cited for a in n.action_items):
        score = match_score(claim.text, a.task)
        if score > task or (score == task and a.owner):
            task, owned = score, bool(a.owner)
    decided = max((match_score(claim.text, d) for n in cited for d in n.decisions), default=0)
    return task >= ACTION_MATCH_MIN and (task > decided or (task == decided and owned))


def restore_owners(doc: NotesDoc, texts: Sequence[str]) -> tuple[NotesDoc, list[str]]:
    """Owners back in the meeting's spelling (`names.restore`); the changes, "लॉरा → Laura"."""
    changed: list[str] = []
    items = []
    for item in doc.sections.action_items:
        owner = names.restore(item.owner, texts) if item.owner else item.owner
        if owner != item.owner:
            changed.append(f"{item.owner} → {owner}")
            item = item.model_copy(update={"owner": owner})
        items.append(item)
    sections = doc.sections.model_copy(update={"action_items": items})
    return doc.model_copy(update={"sections": sections}), changed


def cited_text(claim: Claim, segments: Mapping[int, Segment], slides: Mapping[int, Slide]) -> str:
    """The transcripts and slide text a claim cites."""
    return "\n".join(
        segments[c.id].transcript_text if c.kind == "seg" else slide_text(slides[c.id])
        for c in claim.citations
        if c.id in (segments if c.kind == "seg" else slides)
    )


def guard_dues(
    doc: NotesDoc, segments: Sequence[Segment], slides: Sequence[Slide], source_language: str
) -> tuple[NotesDoc, list[str]]:
    """D40: a due its cited segments and slides don't say is dropped, never replaced by a
    guessed one (Map copies its prompt's example due, "end of next week", onto tasks that have
    none: 6 of long-en's made-up items, 18 AMI dues, 8b). English notes of an English meeting:
    the due's words, close together (`dues.said`). Otherwise only digits can be read: each in
    the cited text, in the meeting's language; a due without digits is kept. An uncited item
    keeps its due: there is nothing to check it against. The drops, "Update dashboard: end of
    next week"."""
    by_seg = {s.seg_id: s for s in segments}
    by_slide = {s.slide_id: s for s in slides}
    dropped: list[str] = []
    items = []
    for item in doc.sections.action_items:
        text = cited_text(item, by_seg, by_slide)
        if item.due and text and not dues.due_said(item.due, text, doc.language, source_language):
            dropped.append(f"{item.text}: {item.due}")
            item = item.model_copy(update={"due": None})
        items.append(item)
    sections = doc.sections.model_copy(update={"action_items": items})
    return doc.model_copy(update={"sections": sections}), dropped


def move_vague_tasks(doc: NotesDoc) -> tuple[NotesDoc, list[str]]:
    """D40: an action item with no owner and no due that only says to keep doing something
    ("Continue monitoring retention metrics…", "Ensure the dashboard is up to date") is a
    point the meeting made, not a task: moved to Key points, never deleted (§5.6). English notes
    only; tasks with an owner or a due stay ("Run the user conference online", long-en)."""
    if doc.language != "en":
        return doc, []
    items: list[ActionItem] = []
    moved: list[Claim] = []
    for item in doc.sections.action_items:
        if item.owner is None and item.due is None and _VAGUE_TASK.match(item.text):
            moved.append(Claim(text=item.text, citations=item.citations))
        else:
            items.append(item)
    if not moved:
        return doc, []
    sections = doc.sections.model_copy(
        update={"action_items": items, "key_points": [*doc.sections.key_points, *moved]}
    )
    return doc.model_copy(update={"sections": sections}), [c.text for c in moved]


def _map_owner(claim: Claim, map_notes: Mapping[int, SegmentNotes]) -> str | None:
    """The owner of the cited segments' Map task that `claim` matches best (≥
    ACTION_MATCH_MIN); None when that task has none, or tasks tie with different owners."""
    scored = [
        (match_score(claim.text, a.task), _known(a.owner))
        for c in claim.citations
        if c.kind == "seg" and c.id in map_notes
        for a in map_notes[c.id].action_items
    ]
    best = max((s for s, _ in scored), default=0.0)
    owners = {o for s, o in scored if s == best}
    if best < ACTION_MATCH_MIN or len(owners) != 1:
        return None
    return owners.pop()


def owned_span(owner: str, task: str) -> str:
    """A task's text as Refine reads it, so its owner anchors the citation in any language
    (D39: conflict-de's German task placed at its segment's start without it, −223 s)."""
    return f"[owner: {owner}] {task}"


def fill_owners(
    doc: NotesDoc, map_notes: Mapping[int, SegmentNotes], refine: Refine
) -> tuple[NotesDoc, list[str]]:
    """D42: an action item R7B wrote as `[owner: ?]` takes the owner of the Map task it states
    (conflict-de: "Erstellung des Einstellungsplans" is Tom's Map task), and its segment
    citations are placed again with the owner. When another item already gives that owner the
    same task (ACTION_MATCH_MIN), the citations go there instead: long-en's pass 2 wrote
    Laura's board presentation twice, once as `[owner: ?]`. The fills, "Tom: …"."""
    filled: list[str] = []
    items = list(doc.sections.action_items)
    merged: set[int] = set()
    for k, item in enumerate(doc.sections.action_items):
        if item.owner is not None or not (owner := _map_owner(item, map_notes)):
            continue
        filled.append(f"{owner}: {item.text}")
        same = [
            i
            for i, other in enumerate(items)
            if i != k
            and i not in merged
            and (other.owner or "").casefold() == owner.casefold()
            and match_score(other.text, item.text) >= ACTION_MATCH_MIN
        ]
        if same:
            other = items[same[0]]
            new = [c for c in item.citations if c not in other.citations]
            items[same[0]] = other.model_copy(update={"citations": [*other.citations, *new]})
            merged.add(k)
            continue
        span = owned_span(owner, item.text)
        citations = [
            c.model_copy(update={"t": refine(c.id, [span])}) if c.kind == "seg" else c
            for c in item.citations
        ]
        items[k] = item.model_copy(update={"owner": owner, "citations": citations})
    kept = [item for i, item in enumerate(items) if i not in merged]
    sections = doc.sections.model_copy(update={"action_items": kept})
    return doc.model_copy(update={"sections": sections}), filled


def _first_seg(claim: Claim) -> int | None:
    return min((c.id for c in claim.citations if c.kind == "seg"), default=None)


def add_lost_tasks(
    doc: NotesDoc,
    map_notes: Mapping[int, SegmentNotes],
    segments: Sequence[Segment],
    texts: Sequence[str],
    refine: Refine,
) -> tuple[NotesDoc, list[str]]:
    """D42: a Map task with an owner said in its segment that no action item or decision of
    the notes states (ACTION_MATCH_MIN) is added, citing its segment, before the first item
    citing a later one. R7B drops some although its prompt keeps every task: pass 2 lost
    long-en's "Summarize competitor research" (Grace) that pass 1 had, and clean-ar's one-pass
    reply Tom's video. Owner-less Map tasks are left to R7B (AMI writes decided designs as
    tasks). The additions, "Grace: …"."""
    by_seg = {s.seg_id: s for s in segments}
    items = list(doc.sections.action_items)
    stated = [c.text for c in [*items, *doc.sections.decisions]]
    added: list[str] = []
    for seg_id in sorted(map_notes):
        if seg_id not in by_seg:
            continue
        for a in map_notes[seg_id].action_items:
            owner = _known(a.owner)
            if not owner or any(match_score(a.task, t) >= ACTION_MATCH_MIN for t in stated):
                continue
            if not names.said_folded(names.restore(owner, texts), by_seg[seg_id].transcript_text):
                continue
            task = ActionItem(
                text=a.task,
                citations=cite([(("seg", seg_id), owned_span(owner, a.task))], refine),
                owner=owner,
                due=_known(a.due),
            )
            later = [
                i for i, it in enumerate(items) if (s := _first_seg(it)) is not None and s > seg_id
            ]
            items.insert(later[0] if later else len(items), task)
            stated.append(a.task)
            added.append(f"{owner}: {a.task}")
    if not added:
        return doc, []
    sections = doc.sections.model_copy(update={"action_items": items})
    return doc.model_copy(update={"sections": sections}), added


def add_task(items: list[ActionItem], task: ActionItem) -> None:
    """Append a moved decision; when an action item already states it (ACTION_MATCH_MIN),
    add its citations there instead of listing the task twice."""
    for i, item in enumerate(items):
        if match_score(item.text, task.text) >= ACTION_MATCH_MIN:
            new = [c for c in task.citations if c not in item.citations]
            items[i] = item.model_copy(update={"citations": [*item.citations, *new]})
            return
    items.append(task)


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
        s = match_score(claim.text, a.task)
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
    moved: int = 0  # decisions moved to action items (filed_task)
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
    tasks: list[Claim] = []  # decisions that are Map's tasks: listed after the action items
    for name in SECTIONS:
        section: list[Any] = []
        for raw in claims(split.sections[name], sentences=name == "summary"):
            claim, dropped = resolver(raw)
            stats.claims += 1
            stats.dropped += dropped
            stats.uncited += not claim.citations
            if name == "decisions" and filed_task(claim, map_notes):
                tasks.append(claim)
            else:
                section.append(action_item(claim, map_notes) if name == "action_items" else claim)
        out[name] = section
    for claim in tasks:
        add_task(out["action_items"], action_item(claim, map_notes))
    stats.moved = len(tasks)
    doc = NotesDoc(sections=NotesSections(**out), passes=1, language=language, raw=reply)
    return doc, stats


def segment_refiner(
    segments: Sequence[Segment],
    items: Sequence[TranscriptItem],
    min_score: float,
    notes_language: str,
    source_language: str,
) -> Refine:
    """Refines citations of spans written in `notes_language` against a meeting held in
    `source_language` (D39's anchors read each in its own language)."""
    by_seg = {s.seg_id: s for s in segments}
    by_chunk = {it.chunk_id: it for it in items}

    def refine(seg_id: int, spans: Sequence[str]) -> float:
        seg = by_seg[seg_id]
        chunks = [by_chunk[c] for c in seg.chunk_ids if c in by_chunk]
        span = best_span(spans, chunks)
        return refine_time(span, chunks, seg.t0, min_score, notes_language, source_language)

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
    lines += _bullets("Decisions", [_marked(c.text, c) for c in s.decisions])
    lines += _bullets("Action items", actions)
    lines += _bullets("Open questions", [_marked(c.text, c) for c in s.open_questions])
    lines += _bullets("Key points", [_marked(c.text, c) for c in s.key_points])
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
        s = match_score(text, claim.text)
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
        found: list[tuple[Target, str]] = []
        dropped = 0
        for span, indices in raw.spans:
            text = _MARKER.sub("", span)
            for i in indices:
                if not 0 <= i < len(sources):
                    dropped += 1
                    continue
                src = sources[i]
                if isinstance(src, PassOne):
                    targets, how, unknown = span_targets(span, src, known)
                    resolved[how] += 1
                    dropped += unknown
                else:
                    targets = [src]
                found += [(t, text) for t in targets]
        return Claim(text=raw.text, citations=cite(found, refine)), dropped

    return resolver


Generate = Callable[[str], str]  # prompt → reply


class Reply(NamedTuple):
    text: str
    finish: str | None  # mlx-lm's finish reason: "stop" (end of turn), "length" (the cap)


def cut_off(reply: Reply) -> bool:
    """The reply stopped at max_tokens before R7B closed it (6e-1: 4 of 5 AMI notes ended at
    2048 tokens and looked complete). A reply whose last token is <|END_RESPONSE|> at the cap
    is complete."""
    return reply.finish == "length" and END_RESPONSE not in reply.text


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
        stats.moved += part.moved
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
    final.moved += stats.moved
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
        return self.complete(prompt, max_tokens=max_tokens, temperature=temperature, seed=seed).text

    def complete(self, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> Reply:
        """mlx-lm's `generate` (the same text), keeping the finish reason it drops."""
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        text, finish = "", None
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        ):
            text += response.text
            finish = response.finish_reason
        return Reply(text, finish)


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
    documents don't fit: only a two-pass group over the cap on its own (one very long
    segment)."""
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
    language = ctx.config.notes_language  # English for an "Other (beta)" target
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
    refine = segment_refiner(
        segments, items, cfg.citation_min_score, language, ctx.config.source_language
    )

    truncated = 0

    def generate(prompt: str) -> str:
        nonlocal truncated
        reply = lm.complete(
            prompt + REPLY_START,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            seed=ctx.config.seed,
        )
        truncated += cut_off(reply)
        return REPLY_START + reply.text

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
                (
                    fill(load_prompt("reduce_part"), target_language=name),
                    fill(load_prompt("reduce_pass2"), target_language=name),
                ),
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
    texts = [s.transcript_text for s in segments] + [slide_text(s) for s in slides]
    doc, filled = fill_owners(doc, map_notes, refine)
    if filled:
        ctx.log.info("reduce: owners from Map's tasks: %s", "; ".join(filled))
    doc, added = add_lost_tasks(doc, map_notes, segments, texts, refine)
    if added:
        ctx.log.info("reduce: Map's tasks the notes lost, added: %s", "; ".join(added))
    doc, restored = restore_owners(doc, texts)
    if restored:
        ctx.log.info("reduce: owners in the meeting's spelling: %s", ", ".join(restored))
    doc, dropped = guard_dues(doc, segments, slides, ctx.config.source_language)
    if dropped:
        ctx.log.info("reduce: dues not said where cited, dropped: %s", "; ".join(dropped))
    doc, vague = move_vague_tasks(doc)
    if vague:
        ctx.log.info("reduce: %d vague tasks moved to key points: %s", len(vague), vague)
    if truncated:
        ctx.log.warning(
            "reduce: %d repl%s cut at reduce.max_tokens = %d: the notes end early",
            truncated,
            "y" if truncated == 1 else "ies",
            cfg.max_tokens,
        )
        doc = doc.model_copy(update={"truncated": truncated})
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
    if stats.moved:
        ctx.log.info("reduce: %d decisions moved to action items (Map's tasks)", stats.moved)
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
    prompts=("reduce", "reduce_part", "reduce_pass2"),
    models=("reduce",),
)
