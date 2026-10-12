"""Ask (PLAN §5.7): questions about one run, answered by Command R7B with citations.

Not a pipeline stage and no run file: the page keeps the history in its session state. Ask
reuses Reduce's documents, prompt fitting, citation parsing (dropped indices, uncited claims
kept and flagged) and time refinement, so an answer cites exactly like the notes do.

History is the last 3 turns, answers without citation markup, dropped oldest-first when the
prompt is over the run's `reduce.max_input_tokens`. If the documents alone don't fit (long
meetings), they're ranked for the question with an in-memory SQLite FTS5 table (BM25) and
the top-ranked ones that fit are sent, in meeting order. The answer language is the
question's (lingua), or the run's target language when detection is unsure.
"""

import logging
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from cohr import reduce, schemas
from cohr.config import TARGET_LANGUAGES, Config, language_name, load_prompt
from cohr.fuse import STOPWORDS
from cohr.mapper import fill
from cohr.models import ModelManager
from cohr.schemas import (
    Claim,
    Manifest,
    MapFile,
    Segment,
    SegmentNotes,
    SegmentsFile,
    Slide,
    SlidesFile,
    Transcript,
    TranscriptItem,
)

log = logging.getLogger("cohr")

HISTORY_TURNS = 3
MAX_ANSWER_TOKENS = 1024
# Language detection (NOTES.md, Phase 5b): lingua's top confidence must be at least
# MIN_CONFIDENCE and lead the runner-up by MIN_LEAD, on at least MIN_WORDS words (or any
# CJK text). On short questions lingua guesses: "Priya?" → Indonesian at 0.56, and
# "Wat is het budget?" ties Dutch and English at 0.25.
MIN_CONFIDENCE = 0.2
MIN_LEAD = 0.05
MIN_WORDS = 2

_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_WORD = re.compile(r"[^\W\d_]+")
_TERM = re.compile(r"[^\W_]+")
TRIGRAM = 3  # FTS5's trigram tokenizer matches terms of at least 3 characters
# Over the cap, at most this many top-ranked documents (6c long-en): filling the whole 32K cap
# (~60 documents) took ~100 s per answer and R7B fell into citing document after document.
MAX_RANKED = 12


class AskError(RuntimeError):
    pass


@dataclass(frozen=True)
class Sources:
    """What Ask answers from: the same run files Reduce reads."""

    segments: list[Segment]
    notes: dict[int, SegmentNotes]
    slides: list[Slide]
    items: list[TranscriptItem]


def load_sources(run_dir: Path) -> Sources:
    return Sources(
        segments=schemas.load(run_dir, SegmentsFile).segments,
        notes={n.seg_id: n for n in schemas.load(run_dir, MapFile).notes},
        slides=schemas.load(run_dir, SlidesFile).slides,
        items=schemas.load(run_dir, Transcript).items,
    )


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str  # without citation markup (§5.7: saves tokens)


@dataclass(frozen=True)
class Answer:
    question: str
    claims: list[Claim]
    language: str
    prompt_tokens: int
    turns_dropped: int  # history turns left out to fit the cap
    ranked: list[reduce.Target] | None  # over the cap: the top-ranked documents sent (§5.7)
    dropped_citations: int  # indices outside the documents sent
    raw: str

    @property
    def uncited(self) -> int:
        return sum(1 for c in self.claims if not c.citations)

    def scope(self) -> str | None:
        """ "Answer based on the N most relevant segments" when ranking left some out."""
        if self.ranked is None:
            return None
        segs = sum(1 for kind, _ in self.ranked if kind == "seg")
        slides = len(self.ranked) - segs
        text = f"Answer based on the {segs} most relevant segment{'s' * (segs != 1)}"
        return text + (f" and {slides} slide{'s' * (slides != 1)}" if slides else "")

    def turn(self) -> Turn:
        return Turn(self.question, " ".join(c.text for c in self.claims))


# --- answer language (pure, given a detector) ----------------------------------------------

Detect = Callable[[str], list[tuple[str, float]]]  # text → (code, confidence), best first


@cache
def _detector() -> Any:
    from lingua import IsoCode639_1, LanguageDetectorBuilder

    codes = [getattr(IsoCode639_1, c.upper()) for c in sorted(TARGET_LANGUAGES)]
    return LanguageDetectorBuilder.from_iso_codes_639_1(*codes).build()


def lingua_confidences(text: str) -> list[tuple[str, float]]:
    return [
        (v.language.iso_code_639_1.name.lower(), v.value)
        for v in _detector().compute_language_confidence_values(text)
    ]


def answer_language(question: str, fallback: str, detect: Detect = lingua_confidences) -> str:
    """The question's language when lingua is sure, else `fallback` (the run's target)."""
    if not _CJK.search(question) and len(_WORD.findall(question)) < MIN_WORDS:
        return fallback
    ranked = detect(question)
    if not ranked:
        return fallback
    lang, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return lang if top >= MIN_CONFIDENCE and top - second >= MIN_LEAD else fallback


# --- prompt (pure, given a model) ----------------------------------------------------------


def instruction(question: str, history: Sequence[Turn], language: str) -> str:
    """One user turn: rules, earlier turns, the question, and the answer language last (as in
    Map, §5.5). R7B's own multi-turn format made it call the document tool again on every
    follow-up instead of answering (NOTES.md, Phase 5b), so history is folded in here."""
    block = ""
    if history:
        turns = "\n".join(
            fill(load_prompt("ask_turn"), question=t.question, answer=t.answer).strip()
            for t in history
        )
        block = fill(load_prompt("ask_history"), turns=turns)
    return fill(
        load_prompt("ask"), history=block, question=question, language=language_name(language)
    )


# --- ranking (pure) -----------------------------------------------------------------------


def query_terms(text: str) -> tuple[list[str], bool]:
    """FTS5 terms for a question and whether to use the trigram tokenizer: words (English
    stopwords left out) for space-separated languages; for text with Chinese, Japanese or
    Korean, every 3-character window of each run (trigram can't match shorter terms)."""
    cjk = bool(_CJK.search(text))
    out: list[str] = []
    for run in _TERM.findall(text.casefold()):
        if not cjk:
            if run not in STOPWORDS and (len(run) > 1 or run.isdigit()):
                out.append(run)
            continue
        out += [run[i : i + TRIGRAM] for i in range(len(run) - TRIGRAM + 1)]
    return list(dict.fromkeys(out)), cjk


def rank(query: str, docs: Sequence[dict[str, str]]) -> list[int]:
    """Document indices, best first: BM25 over title + snippet for the query terms OR'ed
    (D27); documents matching no term follow in meeting order."""
    terms, cjk = query_terms(query)
    hits: list[int] = []
    if terms:
        tokenizer = "trigram" if cjk else "unicode61 remove_diacritics 2"
        con = sqlite3.connect(":memory:")
        try:
            con.execute(f"CREATE VIRTUAL TABLE d USING fts5(body, tokenize='{tokenizer}')")
            con.executemany(
                "INSERT INTO d(rowid, body) VALUES (?, ?)",
                [(i + 1, f"{d['title']}\n{d['snippet']}") for i, d in enumerate(docs)],
            )
            match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
            rows = con.execute(
                "SELECT rowid FROM d WHERE d MATCH ? ORDER BY bm25(d), rowid", (match,)
            )
            hits = [r[0] - 1 for r in rows]
        finally:
            con.close()
    found = set(hits)
    return hits + [i for i in range(len(docs)) if i not in found]


# --- prompt (pure, given a model) ----------------------------------------------------------


@dataclass(frozen=True)
class Fit:
    prompt: str
    targets: list[reduce.Target]  # what each document index sent cites
    tokens: int
    turns_dropped: int
    ranked: bool  # only the top-ranked documents were sent


def fit(
    lm: Any,
    question: str,
    history: Sequence[Turn],
    language: str,
    sources: Sources,
    max_tokens: int,
) -> Fit:
    """The prompt within max_tokens. Every document if they fit (history dropped
    oldest-first as needed); else the documents ranked for the question and the last turn
    (a follow-up like "who owns that?" has few terms of its own), the best that fit with
    the history added greedily (at most MAX_RANKED) and sent in meeting order, so indices
    map like Reduce's."""
    history = list(history)[-HISTORY_TURNS:]
    docs, targets = reduce.documents(sources.segments, sources.notes, sources.slides)
    for dropped in range(len(history) + 1):
        prompt = lm.prompt(instruction(question, history[dropped:], language), docs)
        size = lm.count_tokens(prompt)
        if size <= max_tokens:
            return Fit(prompt, targets, size, dropped, ranked=False)
    last = f"{history[-1].question} {history[-1].answer}" if history else ""
    order = rank(f"{question} {last}", docs)
    for dropped in range(len(history) + 1):
        text = instruction(question, history[dropped:], language)
        chosen: list[int] = []
        best: tuple[str, int] | None = None
        for i in order:
            if len(chosen) == MAX_RANKED:
                break
            trial = sorted([*chosen, i])
            prompt = lm.prompt(text, [docs[k] for k in trial])
            size = lm.count_tokens(prompt)
            if size <= max_tokens:
                chosen, best = trial, (prompt, size)
        if best is not None:
            return Fit(best[0], [targets[k] for k in chosen], best[1], dropped, ranked=True)
    raise AskError(
        f"This meeting is too long for Ask: no part of it fits the {max_tokens}-token cap "
        "with the question."
    )


def parse_answer(
    reply: str, targets: Sequence[reduce.Target], refine: reduce.Refine
) -> tuple[list[Claim], int]:
    """One claim per sentence, with Reduce's citation rules; returns the dropped count."""
    out, dropped = [], 0
    for raw in reduce.claims(reduce.strip_wrapper(reply), sentences=True):
        claim, n = reduce.resolve(raw, targets, refine)
        out.append(claim)
        dropped += n
    return out, dropped


def ask(
    lm: Any,
    question: str,
    history: Sequence[Turn],
    sources: Sources,
    config: Config,
    detect: Detect = lingua_confidences,
) -> Answer:
    question = question.strip()
    if not question:
        raise AskError("Ask a question.")
    if not sources.segments and not sources.slides:
        raise AskError("This run has no transcript or slides to answer from.")
    language = answer_language(question, config.target_language, detect)
    fitted = fit(lm, question, history, language, sources, config.reduce.max_input_tokens)
    reply = lm.generate(
        fitted.prompt,
        max_tokens=MAX_ANSWER_TOKENS,
        temperature=config.reduce.temperature,
        seed=config.seed,
    )
    refine = reduce.segment_refiner(
        sources.segments, sources.items, config.reduce.citation_min_score
    )
    claims, dropped = parse_answer(reply, fitted.targets, refine)
    if dropped:
        log.warning("ask: dropped %d citations to unknown documents", dropped)
    log.info(
        "ask: %s, %d claims, %d uncited, %d prompt tokens, %d turns dropped, %s documents",
        language,
        len(claims),
        sum(1 for c in claims if not c.citations),
        fitted.tokens,
        fitted.turns_dropped,
        len(fitted.targets) if fitted.ranked else "all",
    )
    return Answer(
        question=question,
        claims=claims,
        language=language,
        prompt_tokens=fitted.tokens,
        turns_dropped=fitted.turns_dropped,
        ranked=fitted.targets if fitted.ranked else None,
        dropped_citations=dropped,
        raw=reply,
    )


def run(
    models: ModelManager,
    run_dir: Path,
    question: str,
    history: Sequence[Turn],
    sources: Sources | None = None,
    detect: Detect = lingua_confidences,
) -> Answer:
    """Answer with the run's config, under the ModelManager lock like a job (§7). The
    manager's previous config is restored afterwards, so a job that set it keeps it."""
    config = Config.model_validate(schemas.load(run_dir, Manifest).config)
    sources = sources if sources is not None else load_sources(run_dir)
    with models.lock:
        previous = models.config
        models.set_config(config)
        try:
            with models.use("reduce") as lm:
                return ask(lm, question, history, sources, config, detect)
        finally:
            models.set_config(previous)
