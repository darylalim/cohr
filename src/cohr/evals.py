"""Quality evaluation (PLAN §18): planted-facts scoring, reports, baseline comparison, runner.

The generator (`evals/gen/`, outside the package) writes each item's media and `truth.json`;
this module defines that file's format, runs the pipeline on the item and scores the run with
exact checks. The LLM judge (`judge.py`) adds the secondary metrics afterwards: support rate
(each cited claim against the text it cites) and judge recall (cross-language text facts). Its
answers are cached per item and kept out of the release gate until D15/D16 say otherwise.

Fact matching (§18):
- numbers: the planted value appears in any claim (numbers normalized to digits first);
- decisions: a claim in Decisions holds ≥ 60% of the decision's content tokens;
- action items: a claim in Action items with the same owner and ≥ 60% of the task's tokens,
  whose stated due date (if any) is the planted one.
Cross-language items (notes language ≠ script language) match only exact fields: numbers,
owners and due-date digits. Pure-text decisions there need the judge and are counted apart.

A decision or action item in the notes that matches no planted fact is a made-up item (a real
task with no owner stated is incomplete, not made up). An
action item with the right task but someone else's owner or due date counts as made up too:
it states something that wasn't said.
"""

import contextlib
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cohr import ami, ask, asr, atomic, fleurs, ingest, judge, pipeline, render, schemas, storage
from cohr.ami import AmiSpec
from cohr.config import Config, prompt_hash
from cohr.fleurs import FleursSpec
from cohr.fuse import slide_text, tokens
from cohr.judge import CachedJudge, Generate, Label, Supported
from cohr.mapper import clock, source_numbers
from cohr.numbers import digits, notes_numbers, numbers_in, written_values
from cohr.reduce import SECTIONS, slide_label, slide_snippet
from cohr.schemas import (
    ActionItem,
    ChunksFile,
    Citation,
    Claim,
    Manifest,
    MapFile,
    NotesDoc,
    Segment,
    SegmentsFile,
    Slide,
    SlidesFile,
    Span,
    Transcript,
    TranslatedNotes,
)

CITATION_TOLERANCE_S = 15.0  # D19
TEXT_MATCH_MIN = 0.6  # share of a fact's content tokens found in a claim
SLIDE_TITLE_MATCH_MIN = 0.5
FRAME_S = 0.01  # VAD F1 frame size

# Release-gate thresholds (§18, D16): each is max(PLAN default, 2 × the run-to-run noise of
# 3 quick runs with pipeline seeds 1–3, same media; NOTES.md 6a). Rates: the largest drop
# allowed (points); counts: the largest rise allowed; stages: the largest slowdown (share).
# 6b (D24, same rule on the 6b media and final config): near-miss (3 per item) and made-up
# open questions. Report-only (not here): the one-question-per-item Ask rates (follow-up,
# German alone and after history: 25-point steps on quick) and the judge metrics (D15).
MAX_DROP: dict[str, float] = {
    "fact_recall": 0.059,  # D16 noise 2.94 points; 6c: 5.9 (89.7, 85.3, 83.8%), kept (user)
    "citation_accuracy": 0.069,  # D25: noise 3.4 points (96.7, 96.6, 100%)
    "ask_accuracy": 0.05,  # noise 0; Ask has no PLAN default: fact recall's
    "ask_citation_accuracy": 0.03,  # noise 0; citation accuracy's default
    "ask_near_miss": 0.167,  # D24: noise 8.3 points (7, 7, 6 of 12)
}
MAX_RISE: dict[str, float] = {
    "made_up_items": 2,  # noise 1 (4, 3, 3)
    "made_up_open_questions": 2,  # D24: noise 1 (0, 1, 0)
    "uncited_claims": 0,  # noise 0
    "gap_words_per_min": 0.0,  # noise 0
}
# How a pass-2 citation of a group resolved (reduce.span_targets; report-only).
PASS2 = ("marker", "claim", "group", "uncited")
MAX_STAGE_SLOWDOWN = 0.25  # noisiest stage: reduce, 10.5%

GEN_DIR = Path("evals") / "gen"
SETS_DIR = Path("evals") / "sets"
DATA_DIR = Path("evals") / "data"
REPORTS_DIR = Path("evals") / "reports"
BASELINE = Path("evals") / "baseline.json"
# The release gate (§18) fails only on these sets; others (long, 6c: one run, no noise
# measured) compare with their own baseline, report-only.
GATED_SETS = ("quick", "full")
JUDGE_CACHE_DIR = DATA_DIR / "judge"  # judge answers per item (gitignored with evals/data)
CALIBRATION = GEN_DIR / "judge_calibration.json"
AGREEMENT_MIN = 0.8  # D15: judge metrics may enter the gate only above this (Phase 6)
# CC BY attribution (§18): the last line of every report with FLEURS or AMI in it.
EXTERNAL_DATA_FOOTER = (
    "Evaluation data: AMI Meeting Corpus (University of Edinburgh et al., CC BY 4.0); "
    "FLEURS (Google, CC BY 4.0). Data was not modified or redistributed."
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- eval sets (evals/sets/<name>.yaml) ----------------------------------------------------


class ItemSpec(_Model):
    """One synthetic meeting: how the generator builds it and how the pipeline runs on it."""

    id: str
    seed: int
    language: str = "en"  # script / source language
    target: str = "en"  # notes language
    media: Literal["audio", "video"] = "video"  # video = screen recording of the slides
    slides: Literal["video", "pdf"] = "video"  # pdf = a separate deck, text-aligned (§5.4)
    topics: int = 8
    decisions: int = 4
    actions: int = 5
    gaps: list[Literal["silence", "noise", "music"]] = ["silence"]
    gap_s: float = 12.0
    background_noise: float = 0.0  # pink noise amplitude under the speech
    conflict: bool = False  # a misspoken number, correct on the slide (§5.5)
    regions: list[str] = []  # long meetings: the topic loop once per region (6c)
    discussion: int = 4  # discussion lines per topic
    open_questions: tuple[int, int] = (1, 2)  # how many are planted: random in this range


class EvalSet(_Model):
    name: str
    include: list[str] = []  # other sets whose items come first (full: quick, lang)
    gate: list[str] = []  # included sets whose items gate (D31; full: quick); empty = all items
    items: list[ItemSpec] = []
    fleurs: FleursSpec | None = None  # Transcribe-only clips (full, 6e-1)
    ami: AmiSpec | None = None  # real meetings, audio only (full, 6e-1)


def load_set(name: str, root: Path = Path()) -> EvalSet:
    import yaml  # eval dependency group

    path = root / SETS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no eval set {name!r} ({path})")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    eval_set = EvalSet.model_validate({"name": name, **data})
    if unknown := set(eval_set.gate) - set(eval_set.include):
        raise ValueError(f"set {name!r} gates on sets it doesn't include: {sorted(unknown)}")
    included = {other: load_set(other, root).items for other in eval_set.include}
    items = [it for its in included.values() for it in its]
    return eval_set.model_copy(
        update={
            "items": [*items, *eval_set.items],
            "gate": [it.id for other in eval_set.gate for it in included[other]],
        }
    )


# --- truth.json (written by evals/gen) -----------------------------------------------------


class TruthLine(_Model):
    voice: str
    text: str
    t0: float
    t1: float


class TruthFact(_Model):
    id: str
    kind: Literal["decision", "action", "number", "open_question"]
    text: str
    owner: str | None = None
    due: str | None = None
    value: float | None = None
    misspoken: float | None = None
    t0: float
    t1: float
    slide: int | None = None  # truth slide index when the fact is on a slide


class TruthSlide(_Model):
    slide_id: int
    title: str
    cells: list[str]  # every table cell, header included
    t0: float  # stretch of the meeting where it's discussed
    t1: float


class Truth(_Model):
    item: ItemSpec
    gen_hash: str  # hash of the generator source; a change regenerates the item
    language: str
    duration_s: float
    media: str  # file names inside the item dir
    deck: str | None = None
    script_text: str
    lines: list[TruthLine]
    facts: list[TruthFact]
    speech_spans: list[Span]
    gap_spans: list[Span]
    slides: list[TruthSlide]


def gen_hash(root: Path = Path()) -> str:
    h = hashlib.sha256()
    for path in sorted((root / GEN_DIR).glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


# --- normalization (pure) ------------------------------------------------------------------


def stem(word: str) -> str:
    """Crude suffix stripping so "syncs"/"sync" and "moved"/"move" match."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word.removesuffix("e") if len(word) > 3 else word


def content(text: str) -> set[str]:
    """Stemmed content tokens (stopwords dropped, numbers as digits)."""
    return {stem(t) for t in tokens(digits(text))}


def coverage(fact: str, claim: str) -> float:
    """Share of the fact's content tokens found in the claim."""
    f = content(fact)
    return len(f & content(claim)) / len(f) if f else 0.0


_DUE_FILLER = {"by", "the", "of", "on", "at", "before", "until", "due", "end"}
_MONTH_ABBR = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "sept": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}


def due_key(due: str) -> frozenset[str]:
    """ "the twentieth of October" and "Oct 20" → {"20", "october"}; "end of next week" →
    {"next", "week", "end"} ("end" kept when it's what distinguishes the date)."""
    words = [w.lower().strip(".,") for w in digits(due).split()]
    words = [_MONTH_ABBR.get(w, w) for w in words if w]
    key = {w for w in words if w not in _DUE_FILLER}
    if "end" in words:
        key.add("end")
    return frozenset(key)


def same_due(stated: str, planted: str) -> bool:
    return due_key(stated) == due_key(planted)


def wer_text(text: str) -> str:
    """Text normalized for WER: digits, no punctuation, "%" → "percent", "$X" → "X dollars"."""
    text = digits(text.replace("%", " percent").replace("-", " "))
    text = re.sub(r"\$\s*(\d+(?:\.\d+)?)", r"\1 dollars", text)
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


# --- fact matching (pure) ------------------------------------------------------------------

Section = Literal["summary", "key_points", "decisions", "action_items", "open_questions"]


def fold(name: str) -> str:
    """Casefolded, accents dropped: Spanish Transcribe writes "María" for Maria (6d-2)."""
    decomposed = unicodedata.normalize("NFKD", name.strip().casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _owner_in(claim: Claim, owner: str) -> bool | None:
    """True/False when the claim states an owner (field or name in the text), None if not."""
    if isinstance(claim, ActionItem) and claim.owner:
        return fold(claim.owner) == fold(owner)
    return True if re.search(rf"\b{re.escape(fold(owner))}\b", fold(claim.text)) else None


def _stated_due(claim: Claim) -> str | None:
    return claim.due if isinstance(claim, ActionItem) and claim.due else None


def action_fit(
    claim: Claim, fact: TruthFact, cross_language: bool = False, language: str | None = None
) -> str:
    """One of: "match"; "wrong" (right task, someone else's owner or due date); "partial"
    (right task, no owner stated: not found, but not made up either, e.g. an action item
    repeated under Decisions); "none"."""
    assert fact.owner is not None
    owner = _owner_in(claim, fact.owner)
    stated_due = _stated_due(claim)
    if cross_language:  # owner names and due-date digits survive translation
        if owner is not True:
            return "none"
        want = numbers_in(fact.due or "")
        got = notes_numbers(stated_due or claim.text, language)
        return "wrong" if want and stated_due and not want <= got else "match"
    if coverage(fact.text, claim.text) < TEXT_MATCH_MIN:
        return "none"
    if owner is False or (stated_due and fact.due and not same_due(stated_due, fact.due)):
        return "wrong"
    return "match" if owner else "partial"


def fact_fit(
    claim: Claim, fact: TruthFact, cross_language: bool = False, language: str | None = None
) -> str:
    """`language`: the notes language, for reading their numbers (None = English)."""
    if fact.kind == "number":
        return "match" if fact.value in notes_numbers(claim.text, language) else "none"
    if fact.kind == "action":
        return action_fit(claim, fact, cross_language, language)
    if cross_language:
        return "none"  # pure-text decision: judge only (4b-2)
    return "match" if coverage(fact.text, claim.text) >= TEXT_MATCH_MIN else "none"


# Where a fact counts as found, in the order its citation is checked.
HOME: dict[str, tuple[Section, ...]] = {
    "decision": ("decisions",),
    "action": ("action_items",),
    "number": ("key_points", "decisions", "action_items", "summary", "open_questions"),
    "open_question": ("open_questions",),
}
# Kinds in fact recall and citation accuracy (the gated metrics). Planted open questions (6b)
# have their own recall, so adding them to the generator didn't change what these measure.
RECALL_KINDS = ("decision", "action", "number")
TEXT_KINDS = ("decision", "open_question")  # matched by text only: judge-only cross-language


Notes = NotesDoc | TranslatedNotes


def citation_lists(doc: Notes) -> list[list[list[Citation]]]:
    """Each section's claims' citations, in order: identical before and after translation."""
    return [[c.citations for c in getattr(doc.sections, name)] for name in SECTIONS]


def claims_by_section(doc: Notes) -> dict[Section, list[Claim]]:
    return {name: list(getattr(doc.sections, name)) for name in SECTIONS}


class FactResult(_Model):
    id: str
    kind: str
    text: str
    found: bool
    judge_only: bool = False  # cross-language text fact: found only by the judge
    judge_covered: bool | None = None  # judge's answer for a judge_only fact (None: not judged)
    section: str | None = None
    claim: str | None = None
    citation_ok: bool | None = None
    t_error: float | None = None  # nearest segment citation's t − fact t0 (D19)
    slide_cited: bool = False


def citation_check(
    claim: Claim, fact: TruthFact, slide_ids: Mapping[int, int]
) -> tuple[bool, float | None, bool]:
    """(ok, signed error of the nearest seg citation, cites the fact's slide)."""
    errors = [c.t - fact.t0 for c in claim.citations if c.kind == "seg" and c.t is not None]
    err = min(errors, key=abs) if errors else None
    want = slide_ids.get(fact.slide) if fact.slide is not None else None
    slide_ok = want is not None and any(c.kind == "slide" and c.id == want for c in claim.citations)
    ok = slide_ok or (err is not None and abs(err) <= CITATION_TOLERANCE_S)
    return ok, err, slide_ok


def score_facts(
    facts: Sequence[TruthFact],
    doc: Notes,
    slide_ids: Mapping[int, int],
    cross_language: bool = False,
    language: str | None = None,
) -> list[FactResult]:
    sections = claims_by_section(doc)
    out = []
    for fact in facts:
        judge_only = cross_language and fact.kind in TEXT_KINDS
        result = FactResult(
            id=fact.id, kind=fact.kind, text=fact.text, found=False, judge_only=judge_only
        )
        for name in HOME[fact.kind]:
            claim = next(
                (
                    c
                    for c in sections[name]
                    if fact_fit(c, fact, cross_language, language) == "match"
                ),
                None,
            )
            if claim is not None:
                ok, err, slide_ok = citation_check(claim, fact, slide_ids)
                result = result.model_copy(
                    update={
                        "found": True,
                        "section": name,
                        "claim": claim.text,
                        "citation_ok": ok,
                        "t_error": None if err is None else round(err, 2),
                        "slide_cited": slide_ok,
                    }
                )
                break
        out.append(result)
    return out


def made_up(
    facts: Sequence[TruthFact],
    doc: Notes,
    cross_language: bool = False,
    language: str | None = None,
) -> tuple[list[str], list[str]]:
    """(made-up decisions/action items, the subset that is a real task with a wrong owner or
    due date). Cross-language: only claims with an exact field can be checked (numbers,
    owners); others are left to the judge (4b-2)."""
    sections = claims_by_section(doc)
    made, wrong = [], []
    for name in ("decisions", "action_items"):
        for claim in sections[name]:
            fits = [fact_fit(claim, f, cross_language, language) for f in facts]
            if "match" in fits or "partial" in fits:
                continue
            if "wrong" in fits:
                wrong.append(claim.text)
            elif (
                cross_language
                and not notes_numbers(claim.text, language)
                and _owner_unknown(claim, facts)
            ):
                continue
            made.append(claim.text)
    return made, wrong


def made_up_questions(
    facts: Sequence[TruthFact], doc: Notes, cross_language: bool = False
) -> list[str]:
    """Open-questions claims that match no planted open question (token overlap). Filler talk
    turned into a question lands here (4b-2). Cross-language: not checkable without the judge."""
    if cross_language:
        return []
    planted = [f for f in facts if f.kind == "open_question"]
    return [
        c.text
        for c in doc.sections.open_questions
        if not any(fact_fit(c, f) == "match" for f in planted)
    ]


def _owner_unknown(claim: Claim, facts: Sequence[TruthFact]) -> bool:
    return not any(f.owner and _owner_in(claim, f.owner) is not None for f in facts)


# --- stage metrics (pure) ------------------------------------------------------------------


def _frames(spans: Iterable[tuple[float, float]], n: int) -> list[bool]:
    out = [False] * n
    for t0, t1 in spans:
        for i in range(max(0, int(t0 / FRAME_S)), min(n, math.ceil(t1 / FRAME_S))):
            out[i] = True
    return out


def speech_f1(truth: Sequence[Span], chunks: Sequence[Span], duration: float) -> float:
    n = math.ceil(duration / FRAME_S)
    want = _frames(((s.t0, s.t1) for s in truth), n)
    got = _frames(((c.t0, c.t1) for c in chunks), n)
    tp = sum(a and b for a, b in zip(want, got, strict=True))
    fp = sum(b and not a for a, b in zip(want, got, strict=True))
    fn = sum(a and not b for a, b in zip(want, got, strict=True))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _inside(span: Span, gaps: Sequence[Span]) -> float:
    return sum(max(0.0, min(span.t1, g.t1) - max(span.t0, g.t0)) for g in gaps)


def gap_text(transcript: Transcript, gaps: Sequence[Span]) -> tuple[int, list[str]]:
    """Words transcribed from chunks lying mostly (≥ 50%) inside non-speech gaps."""
    words, texts = 0, []
    for it in transcript.items:
        dur = it.t1 - it.t0
        if dur > 0 and _inside(it, gaps) >= dur / 2 and it.text.strip():
            words += len(it.text.split())
            texts.append(it.text)
    return words, texts


CER_LANGUAGES = ("ja", "ko", "zh")  # no spaces between words: scored by character (§18)
_GROUPED = re.compile(r"(?<=\d)[.,\u00a0\u202f ](?=\d{3}(?!\d))")
_UNIT_WORDS = {
    "es": {"%": " por ciento ", "$": " dólares "},
    "ja": {"%": "パーセント", "$": "ドル"},
}


def asr_text(text: str, language: str) -> str:
    """Text normalized for WER/CER in a language other than English (6d-2): NFKC, thousands
    separators dropped between digits ("42.500", "42,500" → "42500", as Transcribe writes
    either for the script's plain "42500"), "%" spelled the way the script says it, no
    punctuation. Number words aren't read: the script writes digits."""
    text = _GROUPED.sub("", unicodedata.normalize("NFKC", text))
    for sign, word in _UNIT_WORDS.get(language, {}).items():
        text = text.replace(sign, word)
    text = re.sub(r"[^\w\s]", " ", text.casefold())
    if language in CER_LANGUAGES:
        return "".join(text.split())
    return " ".join(text.split())


def asr_error(reference: str, hypothesis: str, language: str) -> tuple[str, float]:
    """("wer", WER) in English (unchanged) and other spaced languages, ("cer", CER) in ja/ko/zh."""
    return asr_corpus_error([reference], [hypothesis], language)


def asr_corpus_error(
    references: Sequence[str], hypotheses: Sequence[str], language: str
) -> tuple[str, float]:
    """asr_error over many utterances: total edits / total reference words (or characters)."""
    import jiwer  # eval dependency group

    norm = wer_text if language == "en" else lambda t: asr_text(t, language)
    ref, hyp = [norm(t) for t in references], [norm(t) for t in hypotheses]
    if language in CER_LANGUAGES:
        return "cer", float(jiwer.cer(ref, hyp))
    return "wer", float(jiwer.wer(ref, hyp))


def _slide_body(slide: Slide) -> str:
    return "\n".join([slide.title, slide.text_md, *slide.tables_md])


def _title_norm(title: str) -> str:
    return " ".join(re.sub(r"[#*_`]", " ", title).casefold().split())


def match_slides(truth: Sequence[TruthSlide], slides: Sequence[Slide]) -> dict[int, int]:
    """truth slide_id → output slide_id, by title tokens (each output slide used once)."""
    out: dict[int, int] = {}
    used: set[int] = set()
    for ts in truth:
        want = set(tokens(ts.title))
        best, score = None, 0.0
        for s in slides:
            if s.slide_id in used:
                continue
            got = set(tokens(s.title)) or set(tokens(_slide_body(s)))
            sc = len(want & got) / len(want) if want else 0.0
            if sc > score:
                best, score = s, sc
        if best is not None and score >= SLIDE_TITLE_MATCH_MIN:
            out[ts.slide_id] = best.slide_id
            used.add(best.slide_id)
    return out


def _cell_found(cell: str, body: str) -> bool:
    nums = numbers_in(cell)
    if nums:
        return nums <= numbers_in(body)
    want = content(cell)
    return bool(want) and want <= content(body)


def slide_metrics(
    truth: Sequence[TruthSlide], slides: Sequence[Slide], ids: Mapping[int, int]
) -> dict[str, float]:
    by_id = {s.slide_id: s for s in slides}
    titles = cells = cells_ok = nums = nums_ok = 0
    for ts in truth:
        out = by_id.get(ids[ts.slide_id]) if ts.slide_id in ids else None
        body = _slide_body(out) if out else ""
        titles += bool(out and _title_norm(out.title) == _title_norm(ts.title))
        for cell in ts.cells:
            cells += 1
            cells_ok += _cell_found(cell, body)
            for v in numbers_in(cell):
                nums += 1
                nums_ok += v in numbers_in(body)
    n = len(truth) or 1
    return {
        "slides_found": len(ids) / n,
        "slide_title_exact": titles / n,
        "slide_cell_accuracy": cells_ok / cells if cells else 0.0,
        "slide_number_recall": nums_ok / nums if nums else 0.0,
    }


def alignment_accuracy(
    truth: Sequence[TruthSlide],
    slides: Sequence[Slide],
    ids: Mapping[int, int],
    chunks: Sequence[Span],
) -> float:
    """Share of slides placed on a chunk that overlaps the stretch where they're discussed."""
    by_id = {s.slide_id: s for s in slides}
    by_t0 = {c.t0: c for c in chunks}
    ok = 0
    for ts in truth:
        out = by_id.get(ids.get(ts.slide_id, -1))
        if out is None or out.t_source != "aligned" or out.t is None or out.t not in by_t0:
            continue
        chunk = by_t0[out.t]
        ok += chunk.t0 < ts.t1 and chunk.t1 > ts.t0
    return ok / len(truth) if truth else 0.0


def notes_language(doc: Notes, candidates: Iterable[str]) -> str | None:
    from lingua import IsoCode639_1, LanguageDetectorBuilder

    text = " ".join(c.text for name in SECTIONS for c in getattr(doc.sections, name))
    codes = [getattr(IsoCode639_1, c.upper()) for c in sorted(set(candidates))]
    detector = LanguageDetectorBuilder.from_iso_codes_639_1(*codes).build()
    lang = detector.detect_language_of(text) if text.strip() else None
    return lang.iso_code_639_1.name.lower() if lang else None


# --- per-item report -----------------------------------------------------------------------


class StageCost(_Model):
    seconds: float | None
    s_per_audio_hour: float | None
    peak_mem_gb: float | None
    reused: bool  # skipped in this eval run: timing is from the run that produced it


class ClaimSupport(_Model):
    section: str
    claim: str  # as the judge saw it (action items with owner and due)
    citations: list[str]  # "seg 0", "slide 3"
    supported: Supported
    reason: str


class ItemReport(_Model):
    id: str
    run_id: str
    seed: int
    duration_s: float
    metrics: dict[str, float | int | None]
    facts: list[FactResult]
    made_up: list[str]
    made_up_questions: list[str] = []  # open questions matching no planted one (6b)
    wrong_fields: list[str]
    uncited: list[str]
    gap_texts: list[str]
    conflict: dict[str, bool] = {}
    number_repairs: list[str] = []  # Map number guard (§5.5): "420,500 → 42,500"
    unverified_numbers: list[str] = []  # flagged by the guard in map.json
    unsupported_numbers: list[str] = []  # in notes.json claims, in no transcript or slide
    support: list[ClaimSupport] = []  # judge answers per cited claim (empty: not judged)
    ask: list["AskResult"] = []  # Ask eval answers (empty: not asked)
    stages: dict[str, StageCost]


def stage_costs(manifest: Manifest, duration_s: float, since: datetime) -> dict[str, StageCost]:
    out = {}
    for name, st in manifest.stages.items():
        secs = (
            (st.finished_at - st.started_at).total_seconds()
            if st.finished_at and st.started_at
            else None
        )
        out[name] = StageCost(
            seconds=None if secs is None else round(secs, 2),
            s_per_audio_hour=None if secs is None else round(secs / (duration_s / 3600), 1),
            peak_mem_gb=None if st.peak_mem_bytes is None else round(st.peak_mem_bytes / 1e9, 2),
            reused=st.started_at is None or st.started_at < since,
        )
    return out


EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def unsupported_numbers(
    claims: Sequence[Claim],
    transcript: Transcript,
    slides: Sequence[Slide],
    language: str | None = None,
    source_language: str | None = None,
) -> list[str]:
    """Numbers in the notes found in no transcript or slide text: did Reduce (or Map, with the
    guard off) change digits? Values as digits, in first-seen order. Notes in another
    `language`: a written number counts if any of its readings is in the sources, so Arabic
    "42,500" (also 42.5 there) isn't listed. The sources are read the way the meeting's
    `source_language` writes numbers ("11.800" and "35万4000", 6d-2)."""
    sources = source_numbers(
        " ".join(i.text for i in transcript.items),
        [slide_text(s) for s in slides],
        source_language,
    )
    out: dict[float, None] = {}
    for c in claims:
        fields = [c.text, *((c.owner or "", c.due or "") if isinstance(c, ActionItem) else ())]
        text = " ".join(fields)
        if language not in (None, "en"):
            for values in written_values(text, language):
                if not values & sources:
                    out.setdefault(min(values))
            continue
        for v in sorted(numbers_in(text)):
            if v not in sources:
                out.setdefault(v)
    return [str(int(v)) if v.is_integer() else str(v) for v in out]


def score_item(truth: Truth, run_dir: Path, since: datetime = EPOCH) -> ItemReport:
    """Score a finished run against its truth. Reads run files only through schemas."""
    spec = truth.item
    manifest = schemas.load(run_dir, Manifest)
    chunks = schemas.load(run_dir, ChunksFile)
    transcript = schemas.load(run_dir, Transcript)
    slides = schemas.load(run_dir, SlidesFile).slides
    english = schemas.load(run_dir, NotesDoc)
    doc = render.shown_notes(run_dir)  # notes_translated.json for an "Other (beta)" target
    translated = isinstance(doc, TranslatedNotes)
    map_notes = schemas.load(run_dir, MapFile).notes
    cross = spec.language != spec.target

    ids = match_slides(truth.slides, slides)
    facts = score_facts(truth.facts, doc, ids, cross, spec.target)
    made, wrong = made_up(truth.facts, doc, cross, spec.target)
    made_q = made_up_questions(truth.facts, doc, cross)
    words, texts = gap_text(transcript, truth.gap_spans)
    gap_min = sum(g.t1 - g.t0 for g in truth.gap_spans) / 60
    all_claims = [c for name in SECTIONS for c in getattr(doc.sections, name)]
    uncited = [c.text for c in all_claims if not c.citations]
    scored = [f for f in facts if not f.judge_only and f.kind in RECALL_KINDS]
    found = [f for f in scored if f.found]
    lang = notes_language(doc, {spec.language, spec.target, "en"})
    repairs = [f"{r.original} → {r.repaired}" for n in map_notes for r in n.number_repairs]
    unverified = [u for n in map_notes for u in n.unverified_numbers]
    unsupported = unsupported_numbers(all_claims, transcript, slides, spec.target, spec.language)
    heard = " ".join(i.text for i in transcript.items)
    asr_kind, asr_err = asr_error(truth.script_text, heard, spec.language)

    metrics: dict[str, float | int | None] = {
        "facts": len(scored),
        "facts_found": len(found),
        "fact_recall": len(found) / len(scored) if scored else None,
        "citation_accuracy": (
            sum(bool(f.citation_ok) for f in found) / len(found) if found else None
        ),
        "claims": len(all_claims),
        "uncited_claims": len(uncited),
        "made_up_items": len(made),
        "made_up_open_questions": None if cross else len(made_q),
        "wrong_owner_or_due": len(wrong),
        "judge_only_facts": sum(f.judge_only for f in facts),
        "vad_speech_f1": round(speech_f1(truth.speech_spans, chunks.chunks, truth.duration_s), 4),
        "gap_words": words,
        "gap_minutes": round(gap_min, 3),
        "gap_words_per_min": round(words / gap_min, 2) if gap_min else 0.0,
        "wer": round(asr_err, 4) if asr_kind == "wer" else None,
        "cer": round(asr_err, 4) if asr_kind == "cer" else None,  # ja/ko/zh scripts (6d-2)
        **slide_metrics(truth.slides, slides, ids),
        "output_language_ok": int(lang == spec.target),
        "map_numbers_repaired": len(repairs),
        "map_numbers_unverified": len(unverified),
        "unsupported_numbers": len(unsupported),
        "reduce_groups": len(english.two_pass.groups) if english.two_pass else 0,
        # two-pass (§4, D26): how pass-2 citations of a group resolved
        **{
            f"pass2_{k}": (english.two_pass.resolved.get(k, 0) if english.two_pass else 0)
            for k in PASS2
        },
        # "Other (beta)" (6d-1): every claim keeps the English notes' citations, in order
        "translation_citations_same": (
            int(citation_lists(doc) == citation_lists(english)) if translated else None
        ),
        "untranslated_claims": doc.untranslated if isinstance(doc, TranslatedNotes) else None,
    }
    if spec.slides == "pdf":
        metrics["slide_alignment"] = alignment_accuracy(truth.slides, slides, ids, chunks.chunks)
    for kind in (*RECALL_KINDS, "open_question"):
        of_kind = [f for f in facts if f.kind == kind and not f.judge_only]
        metrics[f"recall_{kind}"] = (
            sum(f.found for f in of_kind) / len(of_kind) if of_kind else None
        )

    conflict = {}
    for fact, result in zip(truth.facts, facts, strict=True):
        if fact.misspoken is not None:
            conflict = {
                "slide_value_used": result.found,
                "misspoken_value_used": any(
                    fact.misspoken in notes_numbers(c.text, spec.target) for c in all_claims
                ),
            }

    return ItemReport(
        id=spec.id,
        run_id=run_dir.name,
        seed=spec.seed,
        duration_s=truth.duration_s,
        metrics={k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        facts=facts,
        made_up=made,
        made_up_questions=made_q,
        wrong_fields=wrong,
        uncited=uncited,
        gap_texts=texts,
        conflict=conflict,
        number_repairs=repairs,
        unverified_numbers=unverified,
        unsupported_numbers=unsupported,
        stages=stage_costs(manifest, truth.duration_s, since),
    )


# --- judge (secondary metrics) ------------------------------------------------------------


def segment_block(seg: Segment) -> str:
    """A cited segment as the judge sees it: its transcript only. Reduce also saw Map's notes
    in the segment's document, but those are derived text; judging against them would let a
    Map error support itself."""
    return f"[transcript {clock(seg.t0)}–{clock(seg.t1)}]\n{seg.transcript_text}"


def slide_block(slide: Slide) -> str:
    return f"[{slide_label(slide.slide_id)}]\n{slide_snippet(slide)}"


def cited_text(claim: Claim, segments: Mapping[int, Segment], slides: Mapping[int, Slide]) -> str:
    blocks = []
    for kind, i in dict.fromkeys((c.kind, c.id) for c in claim.citations):
        if kind == "seg" and i in segments:
            blocks.append(segment_block(segments[i]))
        elif kind == "slide" and i in slides:
            blocks.append(slide_block(slides[i]))
    return "\n\n".join(blocks)


SECTION_LABEL = {
    "summary": "Summary",
    "key_points": "Key point",
    "decisions": "Decision",
    "action_items": "Action item",
    "open_questions": "Open question",
}


def claim_text(claim: Claim, section: str | None = None) -> str:
    """The claim as the judge sees it: action items carry their owner and due date, and the
    section comes first when given (an open question or a decision is a claim about what was
    asked or agreed, not just about its words)."""
    text = claim.text
    if isinstance(claim, ActionItem):
        fields = [
            f"owner: {claim.owner}" if claim.owner else "",
            f"due: {claim.due}" if claim.due else "",
        ]
        if extra := "; ".join(f for f in fields if f):
            text = f"{text} ({extra})"
    return f"{SECTION_LABEL[section]}: {text}" if section else text


def judge_support(
    run_dir: Path, sections: Mapping[Section, list[Claim]], judge_: CachedJudge
) -> list[ClaimSupport]:
    """The judge's support answer for every cited claim, against the text it cites."""
    segments = {s.seg_id: s for s in schemas.load(run_dir, SegmentsFile).segments}
    slides = {s.slide_id: s for s in schemas.load(run_dir, SlidesFile).slides}
    support = []
    for name, claims in sections.items():
        for claim in claims:
            if not claim.citations:
                continue
            text = claim_text(claim, name)
            answer = judge_.support(text, cited_text(claim, segments, slides))
            support.append(
                ClaimSupport(
                    section=name,
                    claim=text,
                    citations=[f"{c.kind} {c.id}" for c in claim.citations],
                    supported=answer.supported,
                    reason=answer.reason,
                )
            )
    return support


def judge_item(report: ItemReport, run_dir: Path, judge_: CachedJudge) -> ItemReport:
    """Add the judge's answers to a scored item: support for every cited claim, coverage for
    every judge_only fact. Uncited claims aren't judged (they're counted as uncited)."""
    sections = claims_by_section(render.shown_notes(run_dir))
    support = judge_support(run_dir, sections, judge_)
    facts = []
    for f in report.facts:
        if f.judge_only:
            claims = [claim_text(c) for name in HOME[f.kind] for c in sections[name]]
            f = f.model_copy(update={"judge_covered": judge_.coverage(f.text, claims).covered})
        facts.append(f)

    def count(label: Supported) -> int:
        return sum(s.supported == label for s in support)

    only = [f for f in facts if f.judge_only]
    metrics = {
        **report.metrics,
        "cited_claims": len(support),
        "support_rate": round(count("yes") / len(support), 4) if support else None,
        "support_partial": count("partial"),
        "support_no": count("no"),
        "support_unjudged": count("unjudged"),
        "judge_covered": sum(bool(f.judge_covered) for f in only),
        "judge_recall": round(sum(bool(f.judge_covered) for f in only) / len(only), 4)
        if only
        else None,
    }
    return report.model_copy(update={"facts": facts, "support": support, "metrics": metrics})


class JudgeCost(_Model):
    loaded: bool  # False: every answer came from the cache
    calls: int
    cached: int
    seconds: float
    s_per_audio_hour: float | None
    peak_mem_gb: float | None


def judge_cache(root: Path, name: str) -> Path:
    return root / JUDGE_CACHE_DIR / f"{name}.json"


@contextlib.contextmanager
def judge_session(models: Any, log: Callable[[str], None]):
    """A CachedJudge that loads the judge only on its first cache miss, after releasing the
    pipeline models (§4: it's never resident together with them). Releases it at the end."""
    with contextlib.ExitStack() as stack:

        def load() -> Generate:
            if models is None:
                raise RuntimeError("the judge needs a ModelManager (answers not in the cache)")
            log("judge: releasing the pipeline models, loading the judge")
            models.release()
            models.reset_peak()
            return stack.enter_context(models.use("judge")).generate

        revision = (
            models.spec("judge") if models is not None else Config().models["judge"]
        ).revision
        cj = CachedJudge(revision, load)
        try:
            yield cj
        finally:
            cj.save()
    if models is not None and cj.loaded:
        models.release("judge")


def judge_items(
    items: Sequence[tuple[ItemReport, Path]],
    *,
    root: Path,
    models: Any,
    log: Callable[[str], None] = print,
    meetings: Sequence[tuple["AmiReport", Path]] = (),
) -> tuple[list[ItemReport], JudgeCost, list["AmiReport"]]:
    """Judge the synthetic items and the AMI meetings in one session (one judge load)."""
    hours = sum(rep.duration_s for rep, _ in [*items, *meetings]) / 3600
    t0 = time.perf_counter()
    with judge_session(models, log) as cj:
        out = []
        for rep, run_dir in items:
            cj.open(judge_cache(root, rep.id))
            out.append(judge_item(rep, run_dir, cj))
        judged = []
        for meeting, run_dir in meetings:
            cj.open(judge_cache(root, meeting.id))
            judged.append(judge_ami(meeting, run_dir, cj))
        peak = models.peak_memory() if models is not None and cj.loaded else None
    secs = time.perf_counter() - t0
    log(f"judge: {cj.calls} calls, {cj.hits} cached, {secs:.1f} s")
    cost = JudgeCost(
        loaded=cj.loaded,
        calls=cj.calls,
        cached=cj.hits,
        seconds=round(secs, 2),
        s_per_audio_hour=round(secs / hours, 1) if hours else None,
        peak_mem_gb=None if peak is None else round(peak / 1e9, 2),
    )
    return out, cost, judged


# --- Ask eval (§18 Ask row, Phase 6a) ------------------------------------------------------
#
# Questions come from truth.json through templates (no LLM), so every run asks the same ones.
# Per item: one per number fact, one per action, a follow-up after the first action, two
# German number questions (alone; after the English action + follow-up turns) and one
# near-miss (a number's label with a period or team the meeting never mentions).
#
# Scoring: a question is correct when the answer states the fact (numbers exact, owners exact,
# task via token overlap, as in fact recall) and a claim stating it cites its segment within
# ±15 s (D19) or its slide. Near-miss rule (NOTES.md, 6a): going through the claims in order,
# the answer passes if a claim saying the meeting doesn't give it (a negation) comes before or
# with the first claim stating a number from the meeting, or no claim states one; ask.md asks
# for "not given, then what it does give", so a related fact after the negation is fine.

AskKind = Literal["number", "action", "followup", "de_alone", "de_history", "near_miss"]
ASK_KINDS: tuple[AskKind, ...] = (
    "number",
    "action",
    "followup",
    "de_alone",
    "de_history",
    "near_miss",
)
ASK_PROMPTS = ("ask", "ask_history", "ask_turn")
ASK_CACHE_DIR = DATA_DIR / "ask"  # answers per item (gitignored with evals/data)

# Slide row label (evals/gen/script.py TOPICS) → how a question names it, English and German.
NUMBER_LABELS: dict[str, tuple[str, str]] = {
    "Approved budget": ("the approved budget", "das genehmigte Budget"),
    "Week-2 return rate": ("the week-2 return rate", "die Rückkehrquote in Woche 2"),
    "Open tickets": ("the number of open tickets", "die Anzahl offener Tickets"),
    "New trial signups": ("the number of new trial signups", "die Anzahl neuer Testanmeldungen"),
    "Page load (ms)": ("the page load time", "die Ladezeit der Seite"),
    "Monthly recurring revenue": (
        "the monthly recurring revenue",
        "der monatlich wiederkehrende Umsatz",
    ),
    "Candidates in pipeline": (
        "the number of candidates in the pipeline",
        "die Anzahl der Kandidaten in der Pipeline",
    ),
    "Monthly cloud bill": ("the monthly cloud bill", "die monatliche Cloud-Rechnung"),
}
# Periods and teams the generator never says (it says "last quarter/month/week/release").
NEAR_MISS = (
    "What was {label} two years ago?",
    "What is {label} for the marketing team?",
    "What is {label} for next year?",
)
FOLLOW_UP = "Who owns that, and by when?"
_NOT_GIVEN = re.compile(
    r"\b(?:not|no|never|none|unknown|cannot)\b|n['’]t\b|\bkeine?n?\b|\bnicht\b", re.I
)
_GERMAN_THOUSANDS = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b")


class AskQuestion(_Model):
    id: str
    kind: AskKind
    question: str
    fact: str  # truth fact id (near-miss: the fact whose label it borrows)
    after: list[str] = []  # earlier question ids whose turns are the history, oldest first


_REGION = re.compile(r"^(.*) \((\w+)\)$")  # long meetings: "Open tickets (North)"
ASK_SAMPLE = {"number": 12, "action": 6}  # long meetings: most questions asked per kind (6c)


def _label(fact: TruthFact) -> tuple[str, str]:
    """(English, German) question label; a long meeting's region names which one."""
    label = fact.text.rsplit(": ", 1)[0]
    if label in NUMBER_LABELS:  # "Page load (ms)" is a quick label, not a region
        return NUMBER_LABELS[label]
    if m := _REGION.match(label):
        en, de = NUMBER_LABELS[m.group(1)]
        return f"{en} for the {m.group(2)} region", f"{de} in der Region {m.group(2)}"
    return NUMBER_LABELS[label]


def _action_question(fact: TruthFact, by_due: bool) -> str:
    if not by_due:
        return f"What does {fact.owner} need to do?"
    due = fact.due or ""
    due = f"the {due}" if due.startswith("end of") else due
    return f"What does {fact.owner} need to do by {due}?"


def sample(facts: list[TruthFact], n: int) -> list[TruthFact]:
    """At most n facts, evenly spread over the meeting (all of them when there are ≤ n)."""
    if len(facts) <= n:
        return facts
    return [facts[i * len(facts) // n] for i in range(n)]


def asks(spec: ItemSpec) -> bool:
    """Whether the Ask eval has questions for the item: its templates are keyed by the English
    script's fact labels (NUMBER_LABELS), so meetings in other languages (xling-*) get none."""
    return spec.language == "en"


def ask_questions(truth: Truth) -> list[AskQuestion]:
    """The item's Ask questions, deterministic from truth.json."""
    numbers = sample([f for f in truth.facts if f.kind == "number"], ASK_SAMPLE["number"])
    # long-en gives owners several tasks: the question names the due date too, and is asked
    # only when owner and due pick out one task.
    all_actions = [f for f in truth.facts if f.kind == "action"]
    owners = Counter(f.owner for f in all_actions)
    pairs = Counter((f.owner, f.due) for f in all_actions)
    actions = [f for f in all_actions if pairs[(f.owner, f.due)] == 1]
    actions = sample(actions, ASK_SAMPLE["action"])
    out = [
        AskQuestion(id=f"n-{f.id}", kind="number", question=f"What is {_label(f)[0]}?", fact=f.id)
        for f in numbers
    ]
    out += [
        AskQuestion(
            id=f"a-{f.id}",
            kind="action",
            question=_action_question(f, owners[f.owner] > 1),
            fact=f.id,
        )
        for f in actions
    ]
    history: list[str] = []
    if actions:
        first = f"a-{actions[0].id}"
        out.append(
            AskQuestion(
                id="followup",
                kind="followup",
                question=FOLLOW_UP,
                fact=actions[0].id,
                after=[first],
            )
        )
        history = [first, "followup"]
    if numbers:
        de = numbers[0]
        out.append(
            AskQuestion(
                id="de-alone",
                kind="de_alone",
                question=f"Wie hoch ist {_label(de)[1]}?",
                fact=de.id,
            )
        )
        de = numbers[min(1, len(numbers) - 1)]
        out.append(
            AskQuestion(
                id="de-history",
                kind="de_history",
                question=f"Wie hoch ist {_label(de)[1]}?",
                fact=de.id,
                after=history,
            )
        )
        # Every template, each on another number (the last ones), so one item asks three
        # near-misses (6b: one per item was a 25-point step).
        for i, template in enumerate(NEAR_MISS):
            miss = numbers[-1 - i % len(numbers)]
            out.append(
                AskQuestion(
                    id=f"near-miss-{i + 1}",
                    kind="near_miss",
                    question=template.format(label=_label(miss)[0]),
                    fact=miss.id,
                )
            )
    return out


def answer_numbers(text: str) -> set[float]:
    """numbers_in, plus German thousands separators ("42.500 $" → 42500). Eval only:
    numbers.py stays English, as the Map guard uses it."""
    german = _GERMAN_THOUSANDS.sub(lambda m: m.group(0).replace(".", ""), text)
    return numbers_in(text) | numbers_in(german)


def due_in(text: str, due: str) -> bool:
    """The planted due date is stated somewhere in free text ("by the end of next week")."""
    words = {
        _MONTH_ABBR.get(w, w)
        for w in re.findall(r"\w+", re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", digits(text).lower()))
    }
    return due_key(due) <= words


def meeting_numbers(truth: Truth) -> set[float]:
    """Every number spoken or shown, except 1–31 (dates; "week two"): what a near-miss answer
    must not give as its answer."""
    cells = " ".join(c for s in truth.slides for c in s.cells)
    return {v for v in answer_numbers(f"{truth.script_text} {cells}") if v > 31}


class AskResult(_Model):
    id: str
    kind: AskKind
    question: str
    fact: str
    answer: str  # claim texts
    citations: list[str]  # "seg 0 @12.5", "slide 3"
    language: str
    stated: bool  # the fact is in the answer (near-miss: no planted value given as the answer)
    citation_ok: bool | None  # a claim stating it cites right (None: not stated / near-miss)
    t_error: float | None = None
    correct: bool


def _cite_label(c: Any) -> str:
    return f"{c.kind} {c.id}" + (f" @{c.t:g}" if c.t is not None else "")


def near_miss_ok(claims: Sequence[Claim], planted: set[float]) -> bool:
    for claim in claims:
        if _NOT_GIVEN.search(claim.text):
            return True
        if answer_numbers(claim.text) & planted:
            return False
    return True


def score_answer(
    q: AskQuestion,
    answer: ask.Answer,
    facts: Mapping[str, TruthFact],
    slide_ids: Mapping[int, int],
    planted: set[float],
) -> AskResult:
    fact = facts[q.fact]
    claims = answer.claims
    stating: list[Claim]
    if q.kind == "near_miss":
        stating = []
        stated = near_miss_ok(claims, planted)
    elif q.kind == "followup":
        assert fact.owner is not None and fact.due is not None
        text = " ".join(c.text for c in claims)
        stated = bool(re.search(rf"\b{re.escape(fact.owner)}\b", text)) and due_in(text, fact.due)
        stating = [
            c
            for c in claims
            if re.search(rf"\b{re.escape(fact.owner)}\b", c.text) or due_in(c.text, fact.due)
        ]
    elif fact.kind == "number":
        stating = [c for c in claims if fact.value in answer_numbers(c.text)]
        stated = bool(stating)
    else:
        stating = [c for c in claims if fact_fit(c, fact) == "match"]
        stated = bool(stating)

    ok, err = None, None
    if q.kind != "near_miss" and stated:
        checks = [citation_check(c, fact, slide_ids) for c in stating]
        ok = any(c[0] for c in checks)
        errs = [c[1] for c in checks if c[1] is not None]
        err = min(errs, key=abs) if errs else None
    return AskResult(
        id=q.id,
        kind=q.kind,
        question=q.question,
        fact=q.fact,
        answer=" ".join(c.text for c in claims),
        citations=[_cite_label(c) for claim in claims for c in claim.citations],
        language=answer.language,
        stated=stated,
        citation_ok=ok,
        t_error=None if err is None else round(err, 2),
        correct=stated if q.kind == "near_miss" else bool(stated and ok),
    )


def ask_metrics(results: Sequence[AskResult]) -> dict[str, float | int | None]:
    """Ask accuracy and citation accuracy over the English fact questions (number, action);
    one rate each for the follow-up, German alone / after history and the near-miss."""

    def rate(kinds: tuple[str, ...]) -> float | None:
        of = [r for r in results if r.kind in kinds]
        return round(sum(r.correct for r in of) / len(of), 4) if of else None

    facts = [r for r in results if r.kind in ("number", "action")]
    stated = [r for r in facts if r.stated]
    return {
        "ask_questions": len(results),
        "ask_accuracy": rate(("number", "action")),
        "ask_stated": round(len(stated) / len(facts), 4) if facts else None,
        "ask_citation_accuracy": (
            round(sum(bool(r.citation_ok) for r in stated) / len(stated), 4) if stated else None
        ),
        "ask_followup": rate(("followup",)),
        "ask_xlang_alone": rate(("de_alone",)),
        "ask_xlang_history": rate(("de_history",)),
        "ask_near_miss": rate(("near_miss",)),
    }


def sources_hash(sources: ask.Sources) -> str:
    h = hashlib.sha256()
    for group in (sources.segments, list(sources.notes.values()), sources.slides, sources.items):
        for m in group:
            h.update(m.model_dump_json().encode("utf-8"))
            h.update(b"\0")
    return h.hexdigest()


def ask_cache_key(question: str, history: Sequence[ask.Turn], config: Config, sources: str) -> str:
    """Question + history + every Ask prompt's hash + the R7B revision + the run's documents
    and the config Ask reads (seed, target language, Reduce settings)."""
    h = hashlib.sha256()
    parts = (
        question,
        *(f"{t.question}\n{t.answer}" for t in history),
        config.models["reduce"].revision,
        *(prompt_hash(p) for p in ASK_PROMPTS),
        sources,
        str(config.seed),
        config.target_language,
        config.reduce.model_dump_json(),
    )
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _answer_entry(a: ask.Answer) -> dict[str, Any]:
    return {
        "claims": [c.model_dump(mode="json") for c in a.claims],
        "language": a.language,
        "prompt_tokens": a.prompt_tokens,
        "turns_dropped": a.turns_dropped,
        "ranked": a.ranked,
        "dropped_citations": a.dropped_citations,
        "raw": a.raw,
    }


def _answer_from(question: str, entry: Mapping[str, Any]) -> ask.Answer:
    return ask.Answer(
        question=question,
        claims=[Claim.model_validate(c) for c in entry["claims"]],
        language=entry["language"],
        prompt_tokens=entry["prompt_tokens"],
        turns_dropped=entry["turns_dropped"],
        ranked=[tuple(t) for t in r] if (r := entry.get("ranked")) is not None else None,
        dropped_citations=entry["dropped_citations"],
        raw=entry["raw"],
    )


AskFn = Callable[[Path, str, Sequence[ask.Turn], ask.Sources], ask.Answer]


class CachedAsk:
    """Ask answers through an on-disk cache (one JSON file per item), like CachedJudge.
    `answer_fn` (ask.run with the shared ModelManager) is called only on a cache miss."""

    def __init__(self, answer_fn: AskFn):
        self._answer = answer_fn
        self.calls = 0
        self.hits = 0
        self._path: Path | None = None
        self._entries: dict[str, dict[str, Any]] = {}

    def open(self, path: Path) -> None:
        self.save()
        self._path = path
        self._entries = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def save(self) -> None:
        if self._path is not None:
            atomic.write_text(self._path, json.dumps(self._entries, indent=1, sort_keys=True))

    def answer(
        self,
        run_dir: Path,
        config: Config,
        sources: ask.Sources,
        question: str,
        history: Sequence[ask.Turn],
    ) -> ask.Answer:
        key = ask_cache_key(question, history, config, sources_hash(sources))
        if (entry := self._entries.get(key)) is not None:
            try:
                answer = _answer_from(question, entry)
            except (KeyError, ValueError):
                pass
            else:
                self.hits += 1
                return answer
        self.calls += 1
        answer = self._answer(run_dir, question, history, sources)
        self._entries[key] = _answer_entry(answer)
        return answer


def ask_item(report: ItemReport, truth: Truth, run_dir: Path, cache: CachedAsk) -> ItemReport:
    """Ask the item's questions (history from the earlier answers, as the page keeps it) and
    add the results and Ask metrics to its report."""
    config = Config.model_validate(schemas.load(run_dir, Manifest).config)
    sources = ask.load_sources(run_dir)
    facts = {f.id: f for f in truth.facts}
    slide_ids = match_slides(truth.slides, sources.slides)
    planted = meeting_numbers(truth)
    answers: dict[str, ask.Answer] = {}
    results = []
    for q in ask_questions(truth):
        history = [answers[i].turn() for i in q.after]
        answers[q.id] = cache.answer(run_dir, config, sources, q.question, history)
        results.append(score_answer(q, answers[q.id], facts, slide_ids, planted))
    metrics = {**report.metrics, **ask_metrics(results)}
    return report.model_copy(update={"ask": results, "metrics": metrics})


ItemReport.model_rebuild()


def ask_session(models: Any) -> CachedAsk:
    """A CachedAsk whose misses go through ask.run on the shared ModelManager (R7B is still
    resident from Reduce under the default policy)."""

    def answer(
        run_dir: Path, question: str, history: Sequence[ask.Turn], sources: ask.Sources
    ) -> ask.Answer:
        if models is None:
            raise RuntimeError("Ask needs a ModelManager (answers not in the cache)")
        return ask.run(models, run_dir, question, history, sources)

    return CachedAsk(answer)


def ask_cache(root: Path, name: str) -> Path:
    return root / ASK_CACHE_DIR / f"{name}.json"


# --- set report, baseline comparison -------------------------------------------------------


# --- FLEURS (§18, 6e-1): Transcribe alone on read sentences ----------------------------------
#
# Each clip is one sentence, so it goes to Transcribe whole (no VAD, no chunking). Error per
# language is over the 20 clips together (edits / reference words or characters). The refs
# are FLEURS's raw transcriptions; English number words are read as digits (wer_text), other
# languages' aren't (asr_text), so a number Transcribe writes in digits counts as an error
# there. Report-only (D31): kept out of `summary`'s gated keys and `stages`.


class ClipResult(_Model):
    file: str
    reference: str
    hypothesis: str
    error: float


class FleursRow(_Model):
    language: str
    metric: Literal["wer", "cer"]
    error: float
    clips: int
    audio_s: float
    seconds: float  # Transcribe time on these clips (no download)
    s_per_audio_hour: float | None
    results: list[ClipResult]


def read_clip(path: Path) -> Any:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if sr != asr.SR:
        raise ValueError(f"{path}: {sr} Hz, Transcribe needs {asr.SR}")
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def fleurs_languages(spec: FleursSpec | None, items: Sequence[str]) -> list[str]:
    """Languages to run: all with no --item filter or `fleurs`; `fleurs-<lang>` picks one."""
    if spec is None:
        return []
    if not items or "fleurs" in items:
        return list(spec.languages)
    return [lang for lang in spec.languages if f"fleurs-{lang}" in items]


def score_fleurs(language: str, sample: fleurs.Sample, hypotheses: Sequence[str]) -> FleursRow:
    refs = [c.text for c in sample.items]
    metric, error = asr_corpus_error(refs, hypotheses, language)
    results = [
        ClipResult(
            file=c.file,
            reference=c.text,
            hypothesis=h,
            error=round(asr_error(c.text, h, language)[1], 4),
        )
        for c, h in zip(sample.items, hypotheses, strict=True)
    ]
    return FleursRow(
        language=language,
        metric="cer" if metric == "cer" else "wer",
        error=round(error, 4),
        clips=len(results),
        audio_s=0.0,
        seconds=0.0,
        s_per_audio_hour=None,
        results=results,
    )


def run_fleurs(
    spec: FleursSpec,
    languages: Sequence[str],
    *,
    root: Path,
    models: Any,
    overrides: Mapping[str, Any],
    fs: fleurs.FileSystem | None = None,
    log: Callable[[str], None] = print,
) -> list[FleursRow]:
    if models is None:
        raise RuntimeError("FLEURS needs a ModelManager")
    rows = []
    for lang in languages:
        sample, folder = fleurs.load_sample(spec, lang, root, fs=fs, log=log)
        audio = [read_clip(folder / c.file) for c in sample.items]
        models.set_config(apply_overrides(Config(source_language=lang), overrides))
        t0 = time.perf_counter()
        with models.use("transcribe") as model:
            hyps = [model.transcribe(a, lang).strip() for a in audio]
        secs = time.perf_counter() - t0
        audio_s = sum(len(a) for a in audio) / asr.SR
        row = score_fleurs(lang, sample, hyps).model_copy(
            update={
                "audio_s": round(audio_s, 1),
                "seconds": round(secs, 2),
                "s_per_audio_hour": round(secs / audio_s * 3600, 1) if audio_s else None,
            }
        )
        log(f"fleurs-{lang}: {row.metric} {row.error:.1%}, {secs:.1f} s")
        rows.append(row)
    return rows


def fleurs_summary(rows: Sequence[FleursRow]) -> dict[str, float | None]:
    """Mean WER over the spaced languages, mean CER over ja/ko/zh, and Transcribe's speed."""

    def mean(metric: str) -> float | None:
        vals = [r.error for r in rows if r.metric == metric]
        return round(sum(vals) / len(vals), 4) if vals else None

    audio_s = sum(r.audio_s for r in rows)
    secs = sum(r.seconds for r in rows)
    return {
        "fleurs_wer": mean("wer"),
        "fleurs_cer": mean("cer"),
        "fleurs_s_per_audio_hour": round(secs / audio_s * 3600, 1) if audio_s else None,
    }


# --- AMI (§18, 6e-1): real meetings, audio only, English notes ------------------------------
#
# No planted facts: the judge asks, for each reference sentence of the meeting's official
# summary, whether a claim in its sections states it (AMI_HOME: abstract → Summary + Key
# points, decisions → Decisions, actions → Action items, problems → Open questions), and
# whether each cited claim is supported by what it cites. Seconds per audio hour per stage on
# these meetings is the definition of done's speed check (§9). Report-only (D31): AMI rows
# stay out of `items`, so their uncited claims don't reach the gated totals.


# AMI summary parts → the sections their sentences are looked for in (§18). The abstract is a
# 3–13 sentence account of the meeting; our Summary is one claim and the Key points carry the
# rest, so both are searched (6e-1).
AMI_HOME: dict[str, tuple[Section, ...]] = {
    "abstract": ("summary", "key_points"),
    "decisions": ("decisions",),
    "actions": ("action_items",),
    "problems": ("open_questions",),
}


class SentenceResult(_Model):
    part: str  # abstract, decisions, actions, problems
    text: str
    covered: bool | None = None  # None: not judged


class AmiReport(_Model):
    id: str
    run_id: str
    duration_s: float
    metrics: dict[str, float | int | None]
    sentences: list[SentenceResult]
    uncited: list[str]
    support: list[ClaimSupport] = []
    stages: dict[str, StageCost]


def ami_meetings(spec: AmiSpec | None, items: Sequence[str]) -> list[str]:
    """Meetings to run: all with no --item filter or `ami`; `ami-<id>` picks one."""
    if spec is None:
        return []
    if not items or "ami" in items:
        return list(spec.meetings)
    return [m for m in spec.meetings if f"ami-{m}" in items]


def score_ami(
    meeting: str, refs: Mapping[str, list[str]], run_dir: Path, duration_s: float, since: datetime
) -> AmiReport:
    """The judge-free part: claims, uncited claims, stage costs; sentences not yet judged."""
    doc = render.shown_notes(run_dir)
    claims = [c for name in SECTIONS for c in getattr(doc.sections, name)]
    uncited = [c.text for c in claims if not c.citations]
    return AmiReport(
        id=f"ami-{meeting}",
        run_id=run_dir.name,
        duration_s=duration_s,
        metrics={
            "claims": len(claims),
            "uncited_claims": len(uncited),
            **{f"claims_{name}": len(getattr(doc.sections, name)) for name in SECTIONS},
            "output_language_ok": int(notes_language(doc, {"en"}) == "en"),
            "truncated": schemas.load(run_dir, NotesDoc).truncated,  # Reduce cut (6e-2)
        },
        sentences=[SentenceResult(part=part, text=t) for part, ts in refs.items() for t in ts],
        uncited=uncited,
        stages=stage_costs(schemas.load(run_dir, Manifest), duration_s, since),
    )


def ami_coverage(sentences: Sequence[SentenceResult], part: str | None = None) -> float | None:
    """Share of reference sentences the judge found (unjudged ones count as not found)."""
    of_part = [s for s in sentences if part is None or s.part == part]
    if all(s.covered is None for s in of_part):  # none, or not judged
        return None
    return round(sum(bool(s.covered) for s in of_part) / len(of_part), 4)


def judge_ami(report: AmiReport, run_dir: Path, judge_: CachedJudge) -> AmiReport:
    sections = claims_by_section(render.shown_notes(run_dir))
    support = judge_support(run_dir, sections, judge_)
    sentences = []
    for s in report.sentences:
        claims = [claim_text(c) for name in AMI_HOME[s.part] for c in sections[name]]
        sentences.append(s.model_copy(update={"covered": judge_.coverage(s.text, claims).covered}))
    yes = sum(c.supported == "yes" for c in support)
    metrics = {
        **report.metrics,
        "coverage": ami_coverage(sentences),
        **{f"coverage_{part}": ami_coverage(sentences, part) for part in ami.PARTS},
        "cited_claims": len(support),
        "support_rate": round(yes / len(support), 4) if support else None,
        "support_partial": sum(c.supported == "partial" for c in support),
        "support_no": sum(c.supported == "no" for c in support),
        "support_unjudged": sum(c.supported == "unjudged" for c in support),
    }
    return report.model_copy(
        update={"sentences": sentences, "support": support, "metrics": metrics}
    )


def run_ami(
    spec: AmiSpec,
    meetings: Sequence[str],
    *,
    root: Path,
    models: Any,
    overrides: Mapping[str, Any],
    fresh: bool = False,
    log: Callable[[str], None] = print,
    progress: pipeline.ProgressFn | None = None,
) -> list[tuple[AmiReport, Path]]:
    import soundfile as sf

    out = []
    for meeting in meetings:
        wav, refs = ami.ensure(meeting, root, log)
        duration = float(sf.info(wav).duration)
        config = apply_overrides(Config(source_language="en", target_language="en"), overrides)
        name = f"ami-{meeting}"
        if fresh:
            shutil.rmtree(root / DATA_DIR / "runs" / name, ignore_errors=True)
        run_dir = reuse_run(name, [storage.Source(wav, "audio")], root, config, duration)
        since = datetime.now(UTC)
        log(f"{name}: running {run_dir.name} ({duration / 60:.1f} min of audio)")
        t0 = time.perf_counter()
        pipeline.run(run_dir, config=config, models=models, progress=progress)
        secs = time.perf_counter() - t0
        log(f"{name}: pipeline {secs:.1f} s ({secs / duration * 3600 / 60:.1f} min per audio hour)")
        out.append((score_ami(meeting, refs, run_dir, duration, since), run_dir))
    return out


def ami_summary(reports: Sequence[AmiReport]) -> dict[str, float | int | None]:
    """Coverage over all meetings' sentences (per part and in all), support rate over all
    cited claims, and the pipeline's seconds per audio hour (the §9 speed check)."""
    sentences = [s for r in reports for s in r.sentences]
    support = [c for r in reports for c in r.support]
    stages = stage_summary(reports)
    return {
        "ami_meetings": len(reports),
        "ami_audio_s": round(sum(r.duration_s for r in reports), 1),
        "ami_coverage": ami_coverage(sentences),
        **{f"ami_coverage_{part}": ami_coverage(sentences, part) for part in ami.PARTS},
        "ami_support_rate": (
            round(sum(c.supported == "yes" for c in support) / len(support), 4) if support else None
        ),
        "ami_uncited_claims": sum(int(r.metrics.get("uncited_claims") or 0) for r in reports),
        "ami_truncated": sum(int(r.metrics.get("truncated") or 0) for r in reports),
        "ami_s_per_audio_hour": stages["total"]["s_per_audio_hour"],
        "ami_peak_mem_gb": stages["total"]["peak_mem_gb"],
    }


class Report(_Model):
    created_at: datetime
    set: str
    git_sha: str | None
    overrides: dict[str, Any] = {}
    models: dict[str, str]  # stage → repo@revision
    prompts: dict[str, str]  # prompt name → sha256[:12]
    items: list[ItemReport]
    summary: dict[str, float | int | None]
    stages: dict[str, dict[str, float | None]]  # stage → s_per_audio_hour, peak_mem_gb
    citation_errors: list[float] = Field(default_factory=list)  # every t_error, for D19
    judge: JudgeCost | None = None  # eval-only cost, kept out of `stages` and the gate
    ask: JudgeCost | None = None  # Ask eval cost (same shape; R7B already resident)
    # D31: the metrics and stages the gate compares, over the gated items only (full: the
    # quick items); None = the whole summary and stages (quick, and reports before 6e-1)
    gate_items: list[str] | None = None
    gate_summary: dict[str, float | int | None] | None = None
    gate_stages: dict[str, dict[str, float | None]] | None = None
    fleurs: list[FleursRow] = []  # full only: per language, report-only (D31)
    ami: list[AmiReport] = []  # full only: per meeting, report-only (D31)
    ami_stages: dict[str, dict[str, float | None]] = {}  # stage → s/h, peak GB on AMI
    baseline: str | None = None
    regressions: list[str] = []


def summarize(items: Sequence[ItemReport]) -> dict[str, float | int | None]:
    """Micro averages over items: recall over all facts, accuracy over all found facts."""
    scored = [f for it in items for f in it.facts if not f.judge_only]
    facts = [f for f in scored if f.kind in RECALL_KINDS]
    found = [f for f in facts if f.found]

    def total(key: str) -> int:
        return sum(int(it.metrics.get(key) or 0) for it in items)

    def mean(key: str) -> float | None:
        vals = [float(v) for it in items if (v := it.metrics.get(key)) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def recall(kind: str) -> float | None:
        of_kind = [f for f in scored if f.kind == kind]
        return round(sum(f.found for f in of_kind) / len(of_kind), 4) if of_kind else None

    gap_min = sum(float(it.metrics.get("gap_minutes") or 0.0) for it in items)
    judged = [it for it in items if "cited_claims" in it.metrics]
    cited = total("cited_claims")
    yes = sum(s.supported == "yes" for it in judged for s in it.support)
    only = [f for it in judged for f in it.facts if f.judge_only]
    return {
        "items": len(items),
        "facts": len(facts),
        "fact_recall": round(len(found) / len(facts), 4) if facts else None,
        "citation_accuracy": (
            round(sum(bool(f.citation_ok) for f in found) / len(found), 4) if found else None
        ),
        "uncited_claims": total("uncited_claims"),
        "made_up_items": total("made_up_items"),
        "made_up_open_questions": total("made_up_open_questions"),
        "wrong_owner_or_due": total("wrong_owner_or_due"),
        "gap_words": total("gap_words"),
        "gap_words_per_min": round(total("gap_words") / gap_min, 2) if gap_min else 0.0,
        "vad_speech_f1": mean("vad_speech_f1"),
        "wer": mean("wer"),
        "cer": mean("cer"),
        "slide_title_exact": mean("slide_title_exact"),
        "slide_cell_accuracy": mean("slide_cell_accuracy"),
        "slide_number_recall": mean("slide_number_recall"),
        "slide_alignment": mean("slide_alignment"),
        "output_language_ok": mean("output_language_ok"),
        # "Other (beta)" items only (6d-1); None when the set has none
        "translation_citations_same": mean("translation_citations_same"),
        "untranslated_claims": (
            total("untranslated_claims")
            if any(it.metrics.get("untranslated_claims") is not None for it in items)
            else None
        ),
        **{f"recall_{kind}": recall(kind) for kind in (*RECALL_KINDS, "open_question")},
        "map_numbers_repaired": total("map_numbers_repaired"),
        "map_numbers_unverified": total("map_numbers_unverified"),
        "unsupported_numbers": total("unsupported_numbers"),
        "reduce_groups": total("reduce_groups"),
        **{f"pass2_{k}": total(f"pass2_{k}") for k in PASS2},
        # judge (secondary, not in the gate): None when the set wasn't judged
        "support_rate": round(yes / cited, 4) if judged and cited else None,
        "support_partial": total("support_partial") if judged else None,
        "support_no": total("support_no") if judged else None,
        "support_unjudged": total("support_unjudged") if judged else None,
        "judge_recall": (
            round(sum(bool(f.judge_covered) for f in only) / len(only), 4) if only else None
        ),
        **(ask_metrics([r for it in items for r in it.ask]) if any(it.ask for it in items) else {}),
        "audio_s": round(sum(it.duration_s for it in items), 1),
    }


def stage_summary(
    items: Sequence[ItemReport] | Sequence["AmiReport"],
) -> dict[str, dict[str, float | None]]:
    """Seconds per audio hour per stage over all items, and the highest peak memory."""
    hours = sum(it.duration_s for it in items) / 3600
    names = dict.fromkeys(n for it in items for n in it.stages)
    out: dict[str, dict[str, float | None]] = {}
    for name in names:
        costs = [it.stages[name] for it in items if name in it.stages]
        secs = [c.seconds for c in costs if c.seconds is not None]
        peaks = [c.peak_mem_gb for c in costs if c.peak_mem_gb is not None]
        out[name] = {
            "s_per_audio_hour": round(sum(secs) / hours, 1) if secs and hours else None,
            "peak_mem_gb": max(peaks) if peaks else None,
        }
    secs_total = [v["s_per_audio_hour"] for v in out.values() if v["s_per_audio_hour"]]
    out["total"] = {
        "s_per_audio_hour": round(sum(secs_total), 1) if secs_total else None,
        "peak_mem_gb": max((v["peak_mem_gb"] or 0.0 for v in out.values()), default=None),
    }
    return out


def gated_summary(report: Report) -> dict[str, float | int | None]:
    return report.summary if report.gate_summary is None else report.gate_summary


def gated_stages(report: Report) -> dict[str, dict[str, float | None]]:
    return report.stages if report.gate_stages is None else report.gate_stages


def compare(report: Report, baseline: Report) -> list[str]:
    """Release-gate regressions (§18): empty means the report passes. A set with a `gate`
    list (full, D31) compares only its gated items' summary and stages."""
    s, b = gated_summary(report), gated_summary(baseline)
    out = []

    def drop(key: str, limit: float) -> None:
        new, old = s.get(key), b.get(key)
        if new is not None and old is not None and float(old) - float(new) > limit + 1e-9:
            out.append(f"{key} dropped {float(old) - float(new):.1%} ({old} → {new})")

    def rise(key: str, limit: float) -> None:
        new, old = s.get(key), b.get(key)
        if new is not None and old is not None and float(new) - float(old) > limit + 1e-9:
            out.append(f"{key} increased ({old} → {new})")

    for key, limit in MAX_DROP.items():
        drop(key, limit)
    for key, limit in MAX_RISE.items():
        rise(key, limit)
    base_stages = gated_stages(baseline)
    for stage, cost in gated_stages(report).items():
        new = cost.get("s_per_audio_hour")
        old = base_stages.get(stage, {}).get("s_per_audio_hour")
        if new and old and new > old * (1 + MAX_STAGE_SLOWDOWN):
            out.append(f"stage {stage} {new / old - 1:.0%} slower ({old} → {new} s/audio h)")
    return out


def noise(reports: Sequence[Report]) -> dict[str, float]:
    """Run-to-run noise for D16: per summary metric, max − min over the reports; per stage,
    the slowest over the fastest s/audio h minus 1 (key "stage:<name>")."""
    out: dict[str, float] = {}
    for key in reports[0].summary if reports else ():
        vals = [float(v) for r in reports if (v := r.summary.get(key)) is not None]
        if len(vals) == len(reports):
            out[key] = round(max(vals) - min(vals), 4)
    for stage in reports[0].stages if reports else ():
        vals = [v for r in reports if (v := r.stages.get(stage, {}).get("s_per_audio_hour"))]
        if len(vals) == len(reports):
            out[f"stage:{stage}"] = round(max(vals) / min(vals) - 1, 4)
    return out


# --- markdown ------------------------------------------------------------------------------


def _pct(v: float | int | None) -> str:
    return "—" if v is None else f"{float(v):.1%}"


def _num(v: float | int | None) -> str:
    return "—" if v is None else f"{v:g}"


PCT_KEYS = {
    "fact_recall",
    "citation_accuracy",
    "vad_speech_f1",
    "wer",
    "cer",
    "fleurs_wer",
    "fleurs_cer",
    "ami_coverage",
    "ami_coverage_abstract",
    "ami_coverage_decisions",
    "ami_coverage_actions",
    "ami_coverage_problems",
    "ami_support_rate",
    "slides_found",
    "slide_title_exact",
    "slide_cell_accuracy",
    "slide_number_recall",
    "slide_alignment",
    "output_language_ok",
    "recall_decision",
    "recall_action",
    "recall_number",
    "recall_open_question",
    "support_rate",
    "judge_recall",
    "ask_accuracy",
    "ask_stated",
    "ask_citation_accuracy",
    "ask_followup",
    "ask_xlang_alone",
    "ask_xlang_history",
    "ask_near_miss",
}


def error_histogram(errors: Sequence[float], edges: Sequence[float]) -> list[tuple[str, int]]:
    """Counts of |error| per bucket: [0, e1], (e1, e2], …, > last edge."""
    out, lo = [], None
    for hi in edges:
        n = sum((lo is None or abs(e) > lo) and abs(e) <= hi for e in errors)
        out.append((f"{lo or 0:g}–{hi:g} s", n))
        lo = hi
    out.append((f"> {lo:g} s", sum(abs(e) > (lo or 0) for e in errors)))
    return out


def render_markdown(report: Report) -> str:
    s = report.summary
    lines = [
        f"# Eval `{report.set}` · {report.created_at:%Y-%m-%d %H:%M} · {report.git_sha or '?'}",
        "",
    ]
    if report.overrides:
        lines += [f"Overrides: `{json.dumps(report.overrides)}`", ""]
    if report.gate_items is not None:
        gate = ", ".join(report.gate_items) or "none run"
        lines += [f"Gate (D31): {gate}; everything else is report-only.", ""]
    if report.baseline:
        verdict = "**FAIL**" if report.regressions else "pass"
        lines += [f"Baseline `{report.baseline}`: {verdict}", ""]
        lines += [f"- {r}" for r in report.regressions] + ([""] if report.regressions else [])
    lines += ["| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {_pct(v) if k in PCT_KEYS else _num(v)} |" for k, v in s.items()]

    lines += ["", "## Items", ""]
    keys = [
        "fact_recall",
        "citation_accuracy",
        "made_up_items",
        "made_up_open_questions",
        "uncited_claims",
        "gap_words",
        "wer",
        "cer",
        "slide_alignment",
        "support_rate",
        "ask_accuracy",
    ]
    lines += ["| Item | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for it in report.items:
        vals = [_pct(it.metrics.get(k)) if k in PCT_KEYS else _num(it.metrics.get(k)) for k in keys]
        lines.append(f"| {it.id} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Cost (s per audio hour, peak GB)",
        "",
        "| Stage | s/h | GB |",
        "|---|---|---|",
    ]
    for name, c in report.stages.items():
        lines.append(f"| {name} | {_num(c['s_per_audio_hour'])} | {_num(c['peak_mem_gb'])} |")
    if j := report.judge:
        lines.append(f"| judge (eval only) | {_num(j.s_per_audio_hour)} | {_num(j.peak_mem_gb)} |")
        lines += ["", f"Judge: {j.calls} calls, {j.cached} cached, {j.seconds:g} s."]
    if a := report.ask:
        lines += ["", f"Ask: {a.calls} calls, {a.cached} cached, {a.seconds:g} s."]

    if report.fleurs:
        lines += [
            "",
            "## FLEURS (Transcribe only, report-only)",
            "",
            "| Language | Metric | Error | Clips | Audio s | s/h |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {r.language} | {r.metric} | {_pct(r.error)} | {r.clips} | {_num(r.audio_s)} "
            f"| {_num(r.s_per_audio_hour)} |"
            for r in report.fleurs
        ]
        lines += ["", "Worst clip per language:", ""]
        for r in report.fleurs:
            if worst := max(r.results, key=lambda c: c.error, default=None):
                lines.append(
                    f"- {r.language} {_pct(worst.error)}: {worst.reference} → "
                    f"{worst.hypothesis or '(empty)'}"
                )

    if report.ami:
        parts = list(ami.PARTS)
        lines += [
            "",
            "## AMI (real meetings, report-only)",
            "",
            "| Meeting | min | coverage | "
            + " | ".join(parts)
            + " | support | uncited | cut | s/h |",
            "|---" * (len(parts) + 7) + "|",
        ]
        for m in report.ami:
            cov = [_pct(m.metrics.get(f"coverage_{p}")) for p in parts]
            total = sum(c.seconds or 0.0 for c in m.stages.values())
            lines.append(
                f"| {m.id} | {m.duration_s / 60:.1f} | {_pct(m.metrics.get('coverage'))} | "
                + " | ".join(cov)
                + f" | {_pct(m.metrics.get('support_rate'))} | "
                f"{_num(m.metrics.get('uncited_claims'))} | {_num(m.metrics.get('truncated'))} | "
                f"{total / m.duration_s * 3600:.0f} |"
            )
        lines += ["", "AMI cost (s per audio hour, peak GB):", "", "| Stage | s/h | GB |"]
        lines.append("|---|---|---|")
        for name, c in report.ami_stages.items():
            lines.append(f"| {name} | {_num(c['s_per_audio_hour'])} | {_num(c['peak_mem_gb'])} |")

    errs = report.citation_errors
    if errs:
        lines += ["", "## Citation error |t − t0| (D19)", "", "| Bucket | Facts |", "|---|---|"]
        lines += [f"| {b} | {n} |" for b, n in error_histogram(errs, (5, 15, 30, 60))]

    for it in report.items:
        missed = [f for f in it.facts if not f.found and not f.judge_only]
        off = [f for f in it.facts if f.found and not f.citation_ok]
        numbers = it.number_repairs or it.unverified_numbers or it.unsupported_numbers
        weak = [c for c in it.support if c.supported != "yes"]
        judged_facts = [f for f in it.facts if f.judge_only and f.judge_covered is not None]
        asked_wrong = [r for r in it.ask if not r.correct]
        if not (
            missed
            or off
            or it.made_up
            or it.made_up_questions
            or it.uncited
            or it.gap_texts
            or it.conflict
            or numbers
            or weak
            or judged_facts
            or asked_wrong
        ):
            continue
        lines += ["", f"## {it.id} ({it.run_id})", ""]
        lines += [f"- missed {f.kind}: {f.text}" for f in missed]
        lines += [f"- citation off by {_num(f.t_error)} s: {f.text}" for f in off]
        lines += [f"- made up: {t}" for t in it.made_up]
        lines += [f"- made-up open question: {t}" for t in it.made_up_questions]
        lines += [f"- wrong owner/due: {t}" for t in it.wrong_fields]
        lines += [f"- uncited: {t}" for t in it.uncited]
        lines += [f"- text in a gap: {t}" for t in it.gap_texts]
        if it.conflict:
            lines.append(f"- slide conflict: {json.dumps(it.conflict)}")
        lines += [f"- Map number repaired: {r}" for r in it.number_repairs]
        lines += [f"- Map number unverified: {u}" for u in it.unverified_numbers]
        lines += [f"- number in no source: {n}" for n in it.unsupported_numbers]
        lines += [
            f"- support {c.supported} ({', '.join(c.citations)}): {c.claim} · {c.reason}"
            for c in weak
        ]
        lines += [
            f"- judge {'covered' if f.judge_covered else 'missed'} {f.kind}: {f.text}"
            for f in judged_facts
        ]
        lines += [
            f"- ask {r.kind} {_ask_verdict(r)}: {r.question} → {r.answer or '(empty)'}"
            + (f" [{', '.join(r.citations)}]" if r.citations else "")
            for r in asked_wrong
        ]
    for m in report.ami:
        missed = [s for s in m.sentences if s.covered is False and s.part != "abstract"]
        weak = [c for c in m.support if c.supported != "yes"]
        if not (missed or weak or m.uncited):
            continue
        lines += ["", f"## {m.id} ({m.run_id})", ""]
        lines += [f"- missed {s.part}: {s.text}" for s in missed]
        lines += [f"- uncited: {t}" for t in m.uncited]
        lines += [
            f"- support {c.supported} ({', '.join(c.citations)}): {c.claim} · {c.reason}"
            for c in weak
        ]
    if report.fleurs or report.ami:
        lines += ["", "---", "", EXTERNAL_DATA_FOOTER]
    return "\n".join(lines) + "\n"


def _ask_verdict(r: AskResult) -> str:
    if r.kind == "near_miss":
        return "gave a meeting number"
    if not r.stated:
        return "missed"
    return "cited off" + (f" by {_num(r.t_error)} s" if r.t_error is not None else "")


# --- runner --------------------------------------------------------------------------------


def git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
        )
    except OSError:
        return None
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return (out.stdout.strip() + ("-dirty" if dirty else "")) or None


def apply_overrides(config: Config, overrides: Mapping[str, Any]) -> Config:
    """`{"fuse.window_s": 120}` → a new Config (validated)."""
    data = config.model_dump(mode="json")
    for dotted, value in overrides.items():
        node = data
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node[part]
        if leaf not in node:
            raise KeyError(f"unknown config key {dotted!r}")
        node[leaf] = value
    return Config.model_validate(data)


def parse_override(text: str) -> tuple[str, Any]:
    key, sep, value = text.partition("=")
    if not sep:
        raise ValueError(f"expected key=value, got {text!r}")
    try:
        return key.strip(), json.loads(value)
    except json.JSONDecodeError:
        return key.strip(), value


def ensure_item(spec: ItemSpec, root: Path, log: Callable[[str], None]) -> tuple[Truth, Path]:
    """Generate the item into evals/data/gen/<id> unless an up-to-date one is there."""
    item_dir = root / DATA_DIR / "gen" / spec.id
    truth_path = item_dir / "truth.json"
    current = gen_hash(root)
    if truth_path.is_file():
        truth = Truth.model_validate_json(truth_path.read_bytes())
        if truth.item == spec and truth.gen_hash == current:
            return truth, item_dir
    log(f"{spec.id}: generating (seed {spec.seed})")
    subprocess.run(
        [sys.executable, "-m", "evals.gen", spec.model_dump_json(), str(item_dir)],
        cwd=root,
        check=True,
    )
    return Truth.model_validate_json(truth_path.read_bytes()), item_dir


def prepare_run(truth: Truth, item_dir: Path, root: Path, config: Config) -> Path:
    """Reuse the item's run when its inputs are the files just generated (the pipeline's skip
    logic then re-runs only what a config or prompt change affects); otherwise start fresh."""
    sources = [storage.Source(item_dir / truth.media, truth.item.media)]
    if truth.deck:
        sources.append(storage.Source(item_dir / truth.deck, "slides"))
    return reuse_run(truth.item.id, sources, root, config, truth.duration_s)


def reuse_run(
    name: str, sources: list[storage.Source], root: Path, config: Config, duration_s: float
) -> Path:
    """The item's run in evals/data/runs/<name>/ if its inputs are these files, else a new one."""
    runs = root / DATA_DIR / "runs" / name
    want = sorted(storage.sha256_file(s.path) for s in sources)
    for run_dir in storage.list_runs(runs):
        manifest = schemas.load(run_dir, Manifest)
        if sorted(i.sha256 for i in manifest.inputs) == want:
            return run_dir
        storage.delete_run(run_dir, runs)
    return storage.create_run(runs, sources, config, name=name, duration_s=duration_s)


def run_set(
    name: str,
    *,
    root: Path = Path(),
    items: Sequence[str] = (),
    seed: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    fresh: bool = False,
    models: Any = None,
    judge_: bool = True,
    ask_: bool = True,
    log: Callable[[str], None] = print,
    progress: pipeline.ProgressFn | None = None,
) -> Report:
    """Generate, run and score every item of a set, then judge them (after every pipeline run,
    so the judge never shares memory with the pipeline models). `models` is a ModelManager
    (shared, so resident models stay loaded across items)."""
    eval_set = load_set(name, root)
    specs = [s for s in eval_set.items if not items or s.id in items]
    fleurs_langs = fleurs_languages(eval_set.fleurs, items)
    meetings = ami_meetings(eval_set.ami, items)
    if not specs and not fleurs_langs and not meetings:
        raise ValueError(f"no items {list(items)} in set {name!r}")
    if specs or meetings:  # the generator and ingest need it; FLEURS doesn't
        ingest.require()
    overrides = dict(overrides or {})
    reports: list[ItemReport] = []
    run_dirs: list[Path] = []
    config = Config()
    asker = ask_session(models)
    ask_secs = 0.0
    for spec in specs:
        if seed is not None:
            spec = spec.model_copy(update={"seed": seed})
        truth, item_dir = ensure_item(spec, root, log)
        config = apply_overrides(
            Config(source_language=spec.language, target_language=spec.target), overrides
        )
        if fresh:
            shutil.rmtree(root / DATA_DIR / "runs" / spec.id, ignore_errors=True)
        run_dir = prepare_run(truth, item_dir, root, config)
        since = datetime.now(UTC)
        log(f"{spec.id}: running {run_dir.name}")
        t0 = time.perf_counter()
        pipeline.run(run_dir, config=config, models=models, progress=progress)
        log(f"{spec.id}: pipeline {time.perf_counter() - t0:.1f} s")
        report = score_item(truth, run_dir, since)
        if ask_ and not asks(spec):
            log(f"{spec.id}: ask skipped ({spec.language} meeting: questions are English-only)")
        elif ask_:
            t0 = time.perf_counter()
            asker.open(ask_cache(root, spec.id))
            try:
                report = ask_item(report, truth, run_dir, asker)
            finally:
                asker.save()
            ask_secs += time.perf_counter() - t0
            log(f"{spec.id}: ask {len(report.ask)} questions, {time.perf_counter() - t0:.1f} s")
        reports.append(report)
        run_dirs.append(run_dir)

    fleurs_rows = []
    if eval_set.fleurs and fleurs_langs:
        fleurs_rows = run_fleurs(
            eval_set.fleurs, fleurs_langs, root=root, models=models, overrides=overrides, log=log
        )

    ami_runs = []
    if eval_set.ami and meetings:
        ami_runs = run_ami(
            eval_set.ami,
            meetings,
            root=root,
            models=models,
            overrides=overrides,
            fresh=fresh,
            log=log,
            progress=progress,
        )
    ami_reports = [r for r, _ in ami_runs]

    judge_cost = None
    if judge_:
        reports, judge_cost, ami_reports = judge_items(
            list(zip(reports, run_dirs, strict=True)),
            root=root,
            models=models,
            log=log,
            meetings=ami_runs,
        )
    errors = [f.t_error for it in reports for f in it.facts if f.t_error is not None]
    hours = sum(r.duration_s for r in reports) / 3600
    gated = [r for r in reports if r.id in eval_set.gate]
    ask_cost = (
        JudgeCost(
            loaded=asker.calls > 0,
            calls=asker.calls,
            cached=asker.hits,
            seconds=round(ask_secs, 2),
            s_per_audio_hour=round(ask_secs / hours, 1) if hours else None,
            peak_mem_gb=None,
        )
        if ask_
        else None
    )
    return Report(
        created_at=datetime.now().astimezone(),
        set=name,
        git_sha=git_sha(root),
        overrides=overrides,
        models={k: f"{m.repo}@{m.revision[:7]}" for k, m in config.models.items()},
        prompts={p: prompt_hash(p)[:12] for p in ("vision", "map", "map_retry", "reduce")}
        | ({p: prompt_hash(p)[:12] for p in judge.PROMPTS} if judge_ else {})
        | ({p: prompt_hash(p)[:12] for p in ASK_PROMPTS} if ask_ else {}),
        items=reports,
        summary=summarize(reports)
        | (fleurs_summary(fleurs_rows) if fleurs_rows else {})
        | (ami_summary(ami_reports) if ami_reports else {}),
        stages=stage_summary(reports),
        citation_errors=sorted(errors),
        judge=judge_cost,
        ask=ask_cost,
        fleurs=fleurs_rows,
        ami=ami_reports,
        ami_stages=stage_summary(ami_reports) if ami_reports else {},
        gate_items=[r.id for r in gated] if eval_set.gate else None,
        gate_summary=summarize(gated) if eval_set.gate else None,
        gate_stages=stage_summary(gated) if eval_set.gate else None,
    )


def write_report(report: Report, root: Path = Path()) -> Path:
    stem_ = report.created_at.strftime("%Y%m%d-%H%M%S")
    path = root / REPORTS_DIR / f"{stem_}-{report.set}.json"
    atomic.write_text(path, report.model_dump_json(indent=2) + "\n")
    atomic.write_text(path.with_suffix(".md"), render_markdown(report))
    return path


def baseline_path(set_name: str, root: Path = Path()) -> Path:
    """evals/baseline.json for quick; evals/baseline-<set>.json for any other set."""
    return root / (
        BASELINE if set_name == "quick" else BASELINE.with_name(f"baseline-{set_name}.json")
    )


def write_baseline(report: Report, root: Path = Path()) -> Path:
    """Rewrite the set's baseline. Commit it only with the reason (§18)."""
    path = baseline_path(report.set, root)
    atomic.write_text(path, report.model_dump_json(indent=2) + "\n")
    return path


def load_report(path: Path) -> Report:
    return Report.model_validate_json(path.read_bytes())


# --- judge calibration (D15) ---------------------------------------------------------------


class CalibrationPair(_Model):
    id: str
    source: str  # where the claim and citation come from (a quick run, or how it was changed)
    claim: str
    cited: list[str]  # keys into Calibration.sources
    label: Label
    reason: str


class Calibration(_Model):
    """evals/gen/judge_calibration.json: hand-labeled claim / cited-text pairs (§18, D15)."""

    about: str
    sources: dict[str, str]  # key → a segment_block / slide_block from a quick run
    pairs: list[CalibrationPair]

    def cited_text(self, pair: CalibrationPair) -> str:
        return "\n\n".join(self.sources[k] for k in pair.cited)


def load_calibration(root: Path = Path()) -> Calibration:
    return Calibration.model_validate_json((root / CALIBRATION).read_bytes())


class CalibrationResult(_Model):
    id: str
    label: Label
    judge: Supported
    reason: str


class CalibrationReport(_Model):
    created_at: datetime
    git_sha: str | None
    model: str  # repo@revision
    prompts: dict[str, str]
    results: list[CalibrationResult]
    agreement: float  # 3-way label equal (D15 passes at ≥ AGREEMENT_MIN)
    agreement_binary: float  # yes vs not-yes
    confusion: dict[str, dict[str, int]]  # label → judge answer → count
    passed: bool
    cost: JudgeCost


def agreement(results: Sequence[CalibrationResult]) -> tuple[float, float]:
    """(3-way, yes vs not-yes). "unjudged" disagrees with every label."""
    if not results:
        return 0.0, 0.0
    exact = sum(r.judge == r.label for r in results)
    binary = sum(
        r.judge != "unjudged" and (r.judge == "yes") == (r.label == "yes") for r in results
    )
    return exact / len(results), binary / len(results)


def confusion(results: Sequence[CalibrationResult]) -> dict[str, dict[str, int]]:
    answers: tuple[Supported, ...] = ("yes", "partial", "no", "unjudged")
    return {
        label: {a: sum(r.label == label and r.judge == a for r in results) for a in answers}
        for label in ("yes", "partial", "no")
    }


def calibrate(
    *, root: Path = Path(), models: Any = None, log: Callable[[str], None] = print
) -> CalibrationReport:
    cal = load_calibration(root)
    t0 = time.perf_counter()
    with judge_session(models, log) as cj:
        cj.open(judge_cache(root, "calibration"))
        results = []
        for pair in cal.pairs:
            answer = cj.support(pair.claim, cal.cited_text(pair))
            results.append(
                CalibrationResult(
                    id=pair.id, label=pair.label, judge=answer.supported, reason=answer.reason
                )
            )
        peak = models.peak_memory() if models is not None and cj.loaded else None
    secs = time.perf_counter() - t0
    exact, binary = agreement(results)
    spec = Config().models["judge"]
    return CalibrationReport(
        created_at=datetime.now().astimezone(),
        git_sha=git_sha(root),
        model=f"{spec.repo}@{spec.revision[:7]}",
        prompts={p: prompt_hash(p)[:12] for p in judge.PROMPTS},
        results=results,
        agreement=round(exact, 4),
        agreement_binary=round(binary, 4),
        confusion=confusion(results),
        passed=exact >= AGREEMENT_MIN,
        cost=JudgeCost(
            loaded=cj.loaded,
            calls=cj.calls,
            cached=cj.hits,
            seconds=round(secs, 2),
            s_per_audio_hour=None,
            peak_mem_gb=None if peak is None else round(peak / 1e9, 2),
        ),
    )


def render_calibration(report: CalibrationReport) -> str:
    verdict = "pass" if report.passed else "**FAIL**"
    lines = [
        f"# Judge calibration (D15) · {report.created_at:%Y-%m-%d %H:%M} · {report.git_sha or '?'}",
        "",
        f"Judge `{report.model}`: 3-way agreement {_pct(report.agreement)} ({verdict} at "
        f"{AGREEMENT_MIN:.0%}), yes vs not-yes {_pct(report.agreement_binary)}.",
        "",
        "| Label \\ judge | yes | partial | no | unjudged |",
        "|---|---|---|---|---|",
    ]
    for label, row in report.confusion.items():
        lines.append(f"| {label} | " + " | ".join(str(n) for n in row.values()) + " |")
    lines += ["", "## Disagreements", ""]
    lines += [
        f"- {r.id}: label {r.label}, judge {r.judge} · {r.reason}"
        for r in report.results
        if r.judge != r.label
    ] or ["- none"]
    c = report.cost
    lines += [
        "",
        f"Cost: {c.calls} calls, {c.cached} cached, {c.seconds:g} s, peak "
        f"{_num(c.peak_mem_gb)} GB.",
    ]
    return "\n".join(lines) + "\n"


def write_calibration_report(report: CalibrationReport, root: Path = Path()) -> Path:
    path = root / REPORTS_DIR / f"{report.created_at:%Y%m%d-%H%M%S}-judge-calibration.json"
    atomic.write_text(path, report.model_dump_json(indent=2) + "\n")
    atomic.write_text(path.with_suffix(".md"), render_calibration(report))
    return path
