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
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from cohr import reduce, schemas
from cohr.config import TARGET_LANGUAGES, Config, language_name, load_prompt
from cohr.fts import CJK, match, query_terms, tokenizer
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

_WORD = re.compile(r"[^\W\d_]+")
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
    # The question's periods and groups the meeting never gives (D52): the answer opens with
    # "The meeting doesn't give …" (`deny_unsaid`).
    unsaid: list[str] = field(default_factory=list)

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
    if not CJK.search(question) and len(_WORD.findall(question)) < MIN_WORDS:
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


def rank(query: str, docs: Sequence[dict[str, str]]) -> list[int]:
    """Document indices, best first: BM25 over title + snippet for the query terms OR'ed
    (D27); documents matching no term follow in meeting order."""
    terms, cjk = query_terms(query)
    hits: list[int] = []
    if terms:
        con = sqlite3.connect(":memory:")
        try:
            con.execute(f"CREATE VIRTUAL TABLE d USING fts5(body, tokenize='{tokenizer(cjk)}')")
            con.executemany(
                "INSERT INTO d(rowid, body) VALUES (?, ?)",
                [(i + 1, f"{d['title']}\n{d['snippet']}") for i, d in enumerate(docs)],
            )
            rows = con.execute(
                "SELECT rowid FROM d WHERE d MATCH ? ORDER BY bm25(d), rowid", (match(terms),)
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


# --- near-miss: a period or group the meeting never gives (pure, 8j, D52) ------------------
#
# "What is the budget for the marketing team?" when the meeting gives only the company's
# budget: R7B copies the qualifier onto the meeting's number or drops it, whatever ask.md
# says (6b; the same 5 of 12 quick near-misses at seeds 0–3, temperature 0). A value
# question's periods and groups are read here, and one the documents never say makes the
# answer open with "The meeting doesn't give …". English questions about English meetings
# only: the check looks for the question's own words in the documents.

_UNIT = r"(?:year|quarter|month|week|day|sprint|release)s?"
_COUNT = r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a few)"
_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)  # fmt: skip
_ORDINALS = {"1": "first", "2": "second", "3": "third", "4": "fourth"}
_PERIODS = re.compile(
    rf"\b{_COUNT}\s+{_UNIT}\s+ago\b"
    rf"|\b(?:next|last|previous|past|coming|this)\s+{_UNIT}\b"
    r"|\b(?:in|for|during)\s+(?:19|20)\d\d\b"
    r"|\bQ[1-4]\b"
    rf"|\b(?:in|for|during)\s+(?:{'|'.join(_MONTHS)})\b",
    re.I,
)
_GROUP = re.compile(
    r"\b(?:for|in|at)\s+(?:the\s+)?((?:[^\W\d_][\w-]*\s+){1,2}?)"
    r"(?:team|department|group|division|region|office|country|unit)\b",
    re.I,
)
_GROUP_STOP = {"the", "a", "an", "our", "their", "its", "this", "that", "whole", "entire"}
# Only these shapes are read: the subject after the verb is what the answer says isn't given.
_VALUE_QUESTION = re.compile(
    r"^\s*(?:what(?:'s|’s|\s+(?:is|was|are|were))"
    r"|how\s+(?:much|many|high|big|long|large|low)\s+(?:is|was|are|were))"
    r"\s+(?P<subject>.+?)\s*\??\s*$",
    re.I,
)
_NUMBER_WORDS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}  # fmt: skip
_SAME_PERIOD: dict[str, str] = {"previous": "last", "past": "last", "coming": "next"}
_DENIAL = re.compile(r"\b(?:not|no|never|none|unknown|cannot)\b|n['’]t\b", re.I)


@dataclass(frozen=True)
class Qualifier:
    phrase: str  # as the question says it: "two years ago", "for the marketing team"
    said: list[str]  # regexes over fold(documents); the qualifier is said if all match


def fold(text: str) -> str:
    """Casefold, number words to digits ("two years" = "2 years"; "a year" = "1 year") and
    one word per period ("previous quarter" = "last quarter")."""
    text = text.casefold()
    text = re.sub(rf"\b(?:a|an)\s+(?={_UNIT}\b)", "1 ", text)
    text = re.sub(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", lambda m: _NUMBER_WORDS[m[1]], text)
    return re.sub(r"\b(" + "|".join(_SAME_PERIOD) + r")\b", lambda m: _SAME_PERIOD[m[1]], text)


def _words(text: str) -> str:
    """A folded phrase as a regex: whole words, any whitespace between."""
    return r"(?<!\w)" + r"\s+".join(map(re.escape, text.split())) + r"(?!\w)"


def qualifiers(question: str) -> list[Qualifier]:
    """The periods and groups a value question names ([] for other questions)."""
    if not _VALUE_QUESTION.match(question):
        return []
    out = []
    for m in _PERIODS.finditer(question):
        phrase = fold(m[0])
        if q := re.fullmatch(r"q([1-4])", phrase):  # "Q3" or "third quarter"
            said = [rf"(?<!\w)q{q[1]}(?!\w)|{_ORDINALS[q[1]]}\s+quarter"]
        elif re.match(r"(?:in|for|during)\s", phrase):  # a year or a month: the word alone
            said = [_words(phrase.split(None, 1)[1])]
        else:
            said = [_words(phrase)]
        out.append(Qualifier(m[0], said))
    for m in _GROUP.finditer(question):
        names = [w for w in fold(m[1]).split() if w not in _GROUP_STOP]
        if names:  # "for the team" names no group
            out.append(Qualifier(m[0], [_words(w) for w in names]))
    return out


def said_text(sources: Sources) -> str:
    """Every document Ask could send (transcript, Map's notes, slides), folded."""
    docs, _ = reduce.documents(sources.segments, sources.notes, sources.slides)
    return fold("\n".join(f"{d['title']}\n{d['snippet']}" for d in docs))


def unsaid_qualifiers(question: str, said: str) -> list[str]:
    """The question's periods and groups that `said` (said_text) never gives."""
    return [q.phrase for q in qualifiers(question) if not all(re.search(p, said) for p in q.said)]


def _without(text: str, phrase: str) -> str:
    """The claim with a copied qualifier taken out ("revenue for the marketing team is" →
    "revenue is"), with the preposition before it."""
    core = re.sub(r"^(?:for|in|at|during)\s+(?:the\s+)?", "", phrase, flags=re.I)
    pattern = r"\s*(?:\b(?:for|in|at|during|of)\s+(?:the\s+)?)?" + _words(core)
    return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(pattern, "", text, flags=re.I)).strip()


def deny_unsaid(question: str, claims: Sequence[Claim], unsaid: Sequence[str]) -> list[Claim]:
    """The answer when the question names periods or groups the meeting never gives: it opens
    with "The meeting doesn't give <subject>." unless it already opens with a denial; a later
    claim that copies a qualifier onto a number loses it (keeping its citations), and a later
    denial of it is dropped (the opening says it). Other claims stay as R7B wrote them."""
    if not unsaid:
        return list(claims)
    m = _VALUE_QUESTION.match(question)
    assert m is not None  # unsaid qualifiers come only from value questions
    out: list[Claim] = []
    if not (claims and _DENIAL.search(claims[0].text)):
        sentence = fill(load_prompt("ask_not_given"), subject=m["subject"]).strip()
        out.append(Claim(text=sentence, citations=[]))
    else:
        out.append(claims[0])
        claims = claims[1:]
    for claim in claims:
        copied = [p for p in unsaid if re.search(_words(fold(p)), fold(claim.text))]
        if copied and _DENIAL.search(claim.text):
            continue
        text = claim.text
        for phrase in copied:
            text = _without(text, phrase)
        out.append(claim if text == claim.text else claim.model_copy(update={"text": text}))
    return out


def answer_from(
    question: str,
    reply: str,
    targets: Sequence[reduce.Target],
    language: str,
    sources: Sources,
    config: Config,
) -> tuple[list[Claim], int, list[str]]:
    """Claims, dropped citation count and unsaid qualifiers from R7B's reply: everything Ask
    does after generating, so a cached reply goes through the same code (evals.CachedAsk)."""
    refine = reduce.segment_refiner(
        sources.segments,
        sources.items,
        config.reduce.citation_min_score,
        language,
        config.source_language,
    )
    claims, dropped = parse_answer(reply, targets, refine)
    unsaid: list[str] = []
    if language == "en" and config.source_language == "en":
        unsaid = unsaid_qualifiers(question, said_text(sources))
    return deny_unsaid(question, claims, unsaid), dropped, unsaid


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
    # An "Other (beta)" target falls back to the notes' English: R7B doesn't write it (6d-1).
    language = answer_language(question, config.notes_language, detect)
    fitted = fit(lm, question, history, language, sources, config.reduce.max_input_tokens)
    reply = lm.generate(
        fitted.prompt,
        max_tokens=MAX_ANSWER_TOKENS,
        temperature=config.reduce.temperature,
        seed=config.seed,
    )
    claims, dropped, unsaid = answer_from(
        question, reply, fitted.targets, language, sources, config
    )
    if dropped:
        log.warning("ask: dropped %d citations to unknown documents", dropped)
    if unsaid:
        log.info("ask: the meeting never gives %s", ", ".join(repr(p) for p in unsaid))
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
        unsaid=unsaid,
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
