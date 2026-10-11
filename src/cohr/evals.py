"""Quality evaluation (PLAN §18): planted-facts scoring, reports, baseline comparison, runner.

The generator (`evals/gen/`, outside the package) writes each item's media and `truth.json`;
this module defines that file's format, runs the pipeline on the item and scores the run with
exact checks only. The LLM judge (support rate, judge recall) is Phase 4b-2.

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

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cohr import atomic, pipeline, schemas, storage
from cohr.config import Config, prompt_hash
from cohr.fuse import slide_text, tokens
from cohr.numbers import digits, numbers_in
from cohr.reduce import SECTIONS
from cohr.schemas import (
    ActionItem,
    ChunksFile,
    Claim,
    Manifest,
    MapFile,
    NotesDoc,
    Slide,
    SlidesFile,
    Span,
    Transcript,
)

CITATION_TOLERANCE_S = 15.0  # D19
TEXT_MATCH_MIN = 0.6  # share of a fact's content tokens found in a claim
SLIDE_TITLE_MATCH_MIN = 0.5
FRAME_S = 0.01  # VAD F1 frame size

# Release-gate thresholds (§18; D16 revisits them in Phase 6).
MAX_RECALL_DROP = 0.05
MAX_CITATION_DROP = 0.03
MAX_STAGE_SLOWDOWN = 0.25

GEN_DIR = Path("evals") / "gen"
SETS_DIR = Path("evals") / "sets"
DATA_DIR = Path("evals") / "data"
REPORTS_DIR = Path("evals") / "reports"
BASELINE = Path("evals") / "baseline.json"


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


class EvalSet(_Model):
    name: str
    items: list[ItemSpec]


def load_set(name: str, root: Path = Path()) -> EvalSet:
    import yaml  # eval dependency group

    path = root / SETS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no eval set {name!r} ({path})")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvalSet.model_validate({"name": name, **data})


# --- truth.json (written by evals/gen) -----------------------------------------------------


class TruthLine(_Model):
    voice: str
    text: str
    t0: float
    t1: float


class TruthFact(_Model):
    id: str
    kind: Literal["decision", "action", "number"]
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


def _owner_in(claim: Claim, owner: str) -> bool | None:
    """True/False when the claim states an owner (field or name in the text), None if not."""
    if isinstance(claim, ActionItem) and claim.owner:
        return claim.owner.strip().casefold() == owner.casefold()
    return True if re.search(rf"\b{re.escape(owner)}\b", claim.text, re.I) else None


def _stated_due(claim: Claim) -> str | None:
    return claim.due if isinstance(claim, ActionItem) and claim.due else None


def action_fit(claim: Claim, fact: TruthFact, cross_language: bool = False) -> str:
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
        got = numbers_in(stated_due or claim.text)
        return "wrong" if want and stated_due and not want <= got else "match"
    if coverage(fact.text, claim.text) < TEXT_MATCH_MIN:
        return "none"
    if owner is False or (stated_due and fact.due and not same_due(stated_due, fact.due)):
        return "wrong"
    return "match" if owner else "partial"


def fact_fit(claim: Claim, fact: TruthFact, cross_language: bool = False) -> str:
    if fact.kind == "number":
        return "match" if fact.value in numbers_in(claim.text) else "none"
    if fact.kind == "action":
        return action_fit(claim, fact, cross_language)
    if cross_language:
        return "none"  # pure-text decision: judge only (4b-2)
    return "match" if coverage(fact.text, claim.text) >= TEXT_MATCH_MIN else "none"


# Where a fact counts as found, in the order its citation is checked.
HOME: dict[str, tuple[Section, ...]] = {
    "decision": ("decisions",),
    "action": ("action_items",),
    "number": ("key_points", "decisions", "action_items", "summary", "open_questions"),
}


def claims_by_section(doc: NotesDoc) -> dict[Section, list[Claim]]:
    return {name: list(getattr(doc.sections, name)) for name in SECTIONS}


class FactResult(_Model):
    id: str
    kind: str
    text: str
    found: bool
    judge_only: bool = False  # cross-language text fact: needs the judge (4b-2)
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
    doc: NotesDoc,
    slide_ids: Mapping[int, int],
    cross_language: bool = False,
) -> list[FactResult]:
    sections = claims_by_section(doc)
    out = []
    for fact in facts:
        judge_only = cross_language and fact.kind == "decision"
        result = FactResult(
            id=fact.id, kind=fact.kind, text=fact.text, found=False, judge_only=judge_only
        )
        for name in HOME[fact.kind]:
            claim = next(
                (c for c in sections[name] if fact_fit(c, fact, cross_language) == "match"),
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
    facts: Sequence[TruthFact], doc: NotesDoc, cross_language: bool = False
) -> tuple[list[str], list[str]]:
    """(made-up decisions/action items, the subset that is a real task with a wrong owner or
    due date). Cross-language: only claims with an exact field can be checked (numbers,
    owners); others are left to the judge (4b-2)."""
    sections = claims_by_section(doc)
    made, wrong = [], []
    for name in ("decisions", "action_items"):
        for claim in sections[name]:
            fits = [fact_fit(claim, f, cross_language) for f in facts]
            if "match" in fits or "partial" in fits:
                continue
            if "wrong" in fits:
                wrong.append(claim.text)
            elif cross_language and not numbers_in(claim.text) and _owner_unknown(claim, facts):
                continue
            made.append(claim.text)
    return made, wrong


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


def wer(reference: str, hypothesis: str) -> float:
    import jiwer  # eval dependency group

    return float(jiwer.wer(wer_text(reference), wer_text(hypothesis)))


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


def notes_language(doc: NotesDoc, candidates: Iterable[str]) -> str | None:
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


class ItemReport(_Model):
    id: str
    run_id: str
    seed: int
    duration_s: float
    metrics: dict[str, float | int | None]
    facts: list[FactResult]
    made_up: list[str]
    wrong_fields: list[str]
    uncited: list[str]
    gap_texts: list[str]
    conflict: dict[str, bool] = {}
    number_repairs: list[str] = []  # Map number guard (§5.5): "420,500 → 42,500"
    unverified_numbers: list[str] = []  # flagged by the guard in map.json
    unsupported_numbers: list[str] = []  # in notes.json claims, in no transcript or slide
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
    claims: Sequence[Claim], transcript: Transcript, slides: Sequence[Slide]
) -> list[str]:
    """Numbers in the notes found in no transcript or slide text: did Reduce (or Map, with the
    guard off) change digits? Values as digits, in first-seen order."""
    sources = numbers_in(" ".join(i.text for i in transcript.items)).union(
        *(numbers_in(slide_text(s)) for s in slides)
    )
    out: dict[float, None] = {}
    for c in claims:
        fields = [c.text, *((c.owner or "", c.due or "") if isinstance(c, ActionItem) else ())]
        for v in sorted(numbers_in(" ".join(fields))):
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
    doc = schemas.load(run_dir, NotesDoc)
    map_notes = schemas.load(run_dir, MapFile).notes
    cross = spec.language != spec.target

    ids = match_slides(truth.slides, slides)
    facts = score_facts(truth.facts, doc, ids, cross)
    made, wrong = made_up(truth.facts, doc, cross)
    words, texts = gap_text(transcript, truth.gap_spans)
    gap_min = sum(g.t1 - g.t0 for g in truth.gap_spans) / 60
    all_claims = [c for name in SECTIONS for c in getattr(doc.sections, name)]
    uncited = [c.text for c in all_claims if not c.citations]
    scored = [f for f in facts if not f.judge_only]
    found = [f for f in scored if f.found]
    lang = notes_language(doc, {spec.language, spec.target, "en"})
    repairs = [f"{r.original} → {r.repaired}" for n in map_notes for r in n.number_repairs]
    unverified = [u for n in map_notes for u in n.unverified_numbers]
    unsupported = unsupported_numbers(all_claims, transcript, slides)

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
        "wrong_owner_or_due": len(wrong),
        "judge_only_facts": sum(f.judge_only for f in facts),
        "vad_speech_f1": round(speech_f1(truth.speech_spans, chunks.chunks, truth.duration_s), 4),
        "gap_words": words,
        "gap_minutes": round(gap_min, 3),
        "gap_words_per_min": round(words / gap_min, 2) if gap_min else 0.0,
        "wer": round(wer(truth.script_text, " ".join(i.text for i in transcript.items)), 4),
        **slide_metrics(truth.slides, slides, ids),
        "output_language_ok": int(lang == spec.target),
        "map_numbers_repaired": len(repairs),
        "map_numbers_unverified": len(unverified),
        "unsupported_numbers": len(unsupported),
    }
    if spec.slides == "pdf":
        metrics["slide_alignment"] = alignment_accuracy(truth.slides, slides, ids, chunks.chunks)
    for kind in ("decision", "action", "number"):
        of_kind = [f for f in scored if f.kind == kind]
        metrics[f"recall_{kind}"] = (
            sum(f.found for f in of_kind) / len(of_kind) if of_kind else None
        )

    conflict = {}
    for fact, result in zip(truth.facts, facts, strict=True):
        if fact.misspoken is not None:
            conflict = {
                "slide_value_used": result.found,
                "misspoken_value_used": any(
                    fact.misspoken in numbers_in(c.text) for c in all_claims
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
        wrong_fields=wrong,
        uncited=uncited,
        gap_texts=texts,
        conflict=conflict,
        number_repairs=repairs,
        unverified_numbers=unverified,
        unsupported_numbers=unsupported,
        stages=stage_costs(manifest, truth.duration_s, since),
    )


# --- set report, baseline comparison -------------------------------------------------------


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
    baseline: str | None = None
    regressions: list[str] = []


def summarize(items: Sequence[ItemReport]) -> dict[str, float | int | None]:
    """Micro averages over items: recall over all facts, accuracy over all found facts."""
    facts = [f for it in items for f in it.facts if not f.judge_only]
    found = [f for f in facts if f.found]

    def total(key: str) -> int:
        return sum(int(it.metrics.get(key) or 0) for it in items)

    def mean(key: str) -> float | None:
        vals = [float(v) for it in items if (v := it.metrics.get(key)) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def recall(kind: str) -> float | None:
        of_kind = [f for f in facts if f.kind == kind]
        return round(sum(f.found for f in of_kind) / len(of_kind), 4) if of_kind else None

    gap_min = sum(float(it.metrics.get("gap_minutes") or 0.0) for it in items)
    return {
        "items": len(items),
        "facts": len(facts),
        "fact_recall": round(len(found) / len(facts), 4) if facts else None,
        "citation_accuracy": (
            round(sum(bool(f.citation_ok) for f in found) / len(found), 4) if found else None
        ),
        "uncited_claims": total("uncited_claims"),
        "made_up_items": total("made_up_items"),
        "wrong_owner_or_due": total("wrong_owner_or_due"),
        "gap_words": total("gap_words"),
        "gap_words_per_min": round(total("gap_words") / gap_min, 2) if gap_min else 0.0,
        "vad_speech_f1": mean("vad_speech_f1"),
        "wer": mean("wer"),
        "slide_title_exact": mean("slide_title_exact"),
        "slide_cell_accuracy": mean("slide_cell_accuracy"),
        "slide_number_recall": mean("slide_number_recall"),
        "slide_alignment": mean("slide_alignment"),
        "output_language_ok": mean("output_language_ok"),
        **{f"recall_{kind}": recall(kind) for kind in ("decision", "action", "number")},
        "map_numbers_repaired": total("map_numbers_repaired"),
        "map_numbers_unverified": total("map_numbers_unverified"),
        "unsupported_numbers": total("unsupported_numbers"),
        "audio_s": round(sum(it.duration_s for it in items), 1),
    }


def stage_summary(items: Sequence[ItemReport]) -> dict[str, dict[str, float | None]]:
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


def compare(report: Report, baseline: Report) -> list[str]:
    """Release-gate regressions (§18): empty means the report passes."""
    s, b = report.summary, baseline.summary
    out = []

    def drop(key: str, limit: float) -> None:
        new, old = s.get(key), b.get(key)
        if new is not None and old is not None and float(old) - float(new) > limit + 1e-9:
            out.append(f"{key} dropped {float(old) - float(new):.1%} ({old} → {new})")

    def rise(key: str) -> None:
        new, old = s.get(key), b.get(key)
        if new is not None and old is not None and float(new) > float(old) + 1e-9:
            out.append(f"{key} increased ({old} → {new})")

    drop("fact_recall", MAX_RECALL_DROP)
    drop("citation_accuracy", MAX_CITATION_DROP)
    for key in ("made_up_items", "uncited_claims", "gap_words_per_min"):
        rise(key)
    for stage, cost in report.stages.items():
        new = cost.get("s_per_audio_hour")
        old = baseline.stages.get(stage, {}).get("s_per_audio_hour")
        if new and old and new > old * (1 + MAX_STAGE_SLOWDOWN):
            out.append(f"stage {stage} {new / old - 1:.0%} slower ({old} → {new} s/audio h)")
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
    "slides_found",
    "slide_title_exact",
    "slide_cell_accuracy",
    "slide_number_recall",
    "slide_alignment",
    "output_language_ok",
    "recall_decision",
    "recall_action",
    "recall_number",
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
        "uncited_claims",
        "gap_words",
        "wer",
        "slide_alignment",
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

    errs = report.citation_errors
    if errs:
        lines += ["", "## Citation error |t − t0| (D19)", "", "| Bucket | Facts |", "|---|---|"]
        lines += [f"| {b} | {n} |" for b, n in error_histogram(errs, (5, 15, 30, 60))]

    for it in report.items:
        missed = [f for f in it.facts if not f.found and not f.judge_only]
        off = [f for f in it.facts if f.found and not f.citation_ok]
        numbers = it.number_repairs or it.unverified_numbers or it.unsupported_numbers
        if not (
            missed or off or it.made_up or it.uncited or it.gap_texts or it.conflict or numbers
        ):
            continue
        lines += ["", f"## {it.id} ({it.run_id})", ""]
        lines += [f"- missed {f.kind}: {f.text}" for f in missed]
        lines += [f"- citation off by {_num(f.t_error)} s: {f.text}" for f in off]
        lines += [f"- made up: {t}" for t in it.made_up]
        lines += [f"- wrong owner/due: {t}" for t in it.wrong_fields]
        lines += [f"- uncited: {t}" for t in it.uncited]
        lines += [f"- text in a gap: {t}" for t in it.gap_texts]
        if it.conflict:
            lines.append(f"- slide conflict: {json.dumps(it.conflict)}")
        lines += [f"- Map number repaired: {r}" for r in it.number_repairs]
        lines += [f"- Map number unverified: {u}" for u in it.unverified_numbers]
        lines += [f"- number in no source: {n}" for n in it.unsupported_numbers]
    return "\n".join(lines) + "\n"


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
    runs = root / DATA_DIR / "runs" / truth.item.id
    media = item_dir / truth.media
    sources = [storage.Source(media, truth.item.media)]
    if truth.deck:
        sources.append(storage.Source(item_dir / truth.deck, "slides"))
    want = sorted(storage.sha256_file(s.path) for s in sources)
    for run_dir in storage.list_runs(runs):
        manifest = schemas.load(run_dir, Manifest)
        if sorted(i.sha256 for i in manifest.inputs) == want:
            return run_dir
        storage.delete_run(run_dir, runs)
    return storage.create_run(
        runs, sources, config, name=truth.item.id, duration_s=truth.duration_s
    )


def run_set(
    name: str,
    *,
    root: Path = Path(),
    items: Sequence[str] = (),
    seed: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    fresh: bool = False,
    models: Any = None,
    log: Callable[[str], None] = print,
    progress: pipeline.ProgressFn | None = None,
) -> Report:
    """Generate, run and score every item of a set. `models` is a ModelManager (shared, so
    resident models stay loaded across items)."""
    eval_set = load_set(name, root)
    specs = [s for s in eval_set.items if not items or s.id in items]
    if not specs:
        raise ValueError(f"no items {list(items)} in set {name!r}")
    overrides = dict(overrides or {})
    reports: list[ItemReport] = []
    config = Config()
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
        reports.append(score_item(truth, run_dir, since))

    errors = [f.t_error for it in reports for f in it.facts if f.t_error is not None]
    return Report(
        created_at=datetime.now().astimezone(),
        set=name,
        git_sha=git_sha(root),
        overrides=overrides,
        models={k: f"{m.repo}@{m.revision[:7]}" for k, m in config.models.items()},
        prompts={p: prompt_hash(p)[:12] for p in ("vision", "map", "map_retry", "reduce")},
        items=reports,
        summary=summarize(reports),
        stages=stage_summary(reports),
        citation_errors=sorted(errors),
    )


def write_report(report: Report, root: Path = Path()) -> Path:
    stem_ = report.created_at.strftime("%Y%m%d-%H%M%S")
    path = root / REPORTS_DIR / f"{stem_}-{report.set}.json"
    atomic.write_text(path, report.model_dump_json(indent=2) + "\n")
    atomic.write_text(path.with_suffix(".md"), render_markdown(report))
    return path


def write_baseline(report: Report, root: Path = Path()) -> Path:
    """Rewrite evals/baseline.json. Commit it only with the reason (§18); not before Phase 6."""
    path = root / BASELINE
    atomic.write_text(path, report.model_dump_json(indent=2) + "\n")
    return path


def load_report(path: Path) -> Report:
    return Report.model_validate_json(path.read_bytes())
