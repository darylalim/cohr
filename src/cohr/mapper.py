"""Map (PLAN §5.5): Tiny Aya Global on mlx-lm, one call per segment, JSON in the target language.

The prompt ends with the target-language instruction (Phase 0: with it only at the top, de and ja
notes came back in English). Invalid JSON is retried once; after that the raw text becomes the
summary with `raw: true`, so one bad segment never fails the stage.

The number guard then checks every number Map wrote with digits against the segment's own
transcript and slide text, and repairs or flags the ones that aren't there (§5.5, D20); then
dates and dues are checked against what the segment says (8c, D41).
"""

import json
import re
from pathlib import Path
from typing import Any

from cohr import dues, names, schemas
from cohr.config import language_name, load_prompt
from cohr.fuse import match_score, slide_text
from cohr.numbers import (
    SCALE,
    WRITTEN,
    Reading,
    native_readings,
    near,
    ordinals_in,
    readings,
    source_numbers_in,
    write_like,
)
from cohr.pipeline import Stage, StageContext
from cohr.schemas import (
    MapActionItem,
    MapFile,
    NumberRepair,
    Segment,
    SegmentNotes,
    SegmentsFile,
    SlideConflict,
    SlidesFile,
)

KEYS = ("summary", "key_points", "decisions", "action_items", "open_questions", "terms")
_EMPTY = {"", "none", "null", "n/a", "na", "-", "?", "unknown"}
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


# --- prompt and parsing (pure) -------------------------------------------------------------


def clock(t: float) -> str:
    s = int(t)
    return (
        f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"
    )


def fill(template: str, **values: str) -> str:
    """Substitute the {placeholders} named in `values` in one pass, so text inside a value is
    never substituted (the prompts also contain literal JSON braces, which str.format would
    trip on)."""
    if not values:
        return template
    names = re.compile(r"\{(" + "|".join(map(re.escape, values)) + r")\}")
    return names.sub(lambda m: values[m.group(1)], template)


def prompt_values(seg: Segment, slides: list[str], target_language: str) -> dict[str, str]:
    return {
        "target_language": language_name(target_language),
        "time_range": f"{clock(seg.t0)}–{clock(seg.t1)}",
        "slides": "\n\n".join(slides) or "(none)",
        "transcript": seg.transcript_text,
    }


def parse(text: str) -> dict[str, Any] | None:
    """The outermost {...} in the reply, if it's a JSON object with at least one Map key."""
    m = _JSON_OBJECT.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not any(k in obj for k in KEYS):
        return None
    return obj


def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    return "" if text.casefold() in _EMPTY else text


def _strings(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    return [t for v in items if (t := _str(v))]


def _actions(value: Any) -> list[MapActionItem]:
    out = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, dict):
            task = _str(item.get("task"))
            owner, due = _str(item.get("owner")) or None, _str(item.get("due")) or None
        else:
            task, owner, due = _str(item), None, None
        if task:
            out.append(MapActionItem(owner=owner, task=task, due=due))
    return out


def make_notes(seg_id: int, obj: dict[str, Any] | None, raw: str) -> SegmentNotes:
    if obj is None:
        return SegmentNotes(
            seg_id=seg_id,
            summary=raw.strip(),
            key_points=[],
            decisions=[],
            action_items=[],
            open_questions=[],
            terms=[],
            raw=True,
        )
    summary = obj.get("summary")
    return SegmentNotes(
        seg_id=seg_id,
        summary=" ".join(_strings(summary)) if isinstance(summary, list) else _str(summary),
        key_points=_strings(obj.get("key_points")),
        decisions=_strings(obj.get("decisions")),
        action_items=_actions(obj.get("action_items")),
        open_questions=_strings(obj.get("open_questions")),
        terms=_strings(obj.get("terms")),
    )


# --- number guard (pure, §5.5) -------------------------------------------------------------


def source_numbers(transcript: str, slides: list[str], language: str | None = None) -> set[float]:
    """The numbers said or shown, read the way the meeting's `language` writes them (6d-2)."""
    return set().union(
        source_numbers_in(transcript, language), *(source_numbers_in(s, language) for s in slides)
    )


def source_days(transcript: str, slides: list[str]) -> set[float]:
    """The ordinals said or shown ("the 22nd"): the only repair candidates for a Map ordinal."""
    return set().union(ordinals_in(transcript), *(ordinals_in(s) for s in slides))


def check_numbers(
    text: str, sources: set[float], language: str | None = None, days: set[float] | None = None
) -> tuple[str, list[NumberRepair], list[str]]:
    """Keep a number found in the sources; repair it when exactly one source number is near it
    (one digit edit, or a factor of 10/100/1000), in the written style; otherwise keep it as
    written and flag it. Only the readings that follow the target `language`'s separators
    (`native_readings`) are kept or repaired (6b-2 for repairs, 6d-2 for keeping: English
    "$34,000" isn't kept as 34.0 because "34%" was said), and for an ordinal ("2nd") repair
    candidates come only from `days` when given ("by the 22nd", not "12 seats"). A number
    with a scale suffix ("42.5k") or grouped with plain spaces ("42 500", which could also be two
    numbers) is only kept or flagged."""
    repairs: list[NumberRepair] = []
    unverified: list[str] = []

    def check(m: re.Match[str]) -> str:
        body, written = m.group("body"), m.group(0)
        reads = native_readings(readings(body), language)
        scale = SCALE.get(m.group("scale") or "", 1.0)
        if not reads or any(r.value * scale in sources for r in reads):
            return written
        if " " in body and all(p.isdigit() and float(p) in sources for p in body.split(" ")):
            return written
        near_by: dict[float, Reading] = {}
        if scale == 1.0 and " " not in body:
            for r in reads:
                for s in days if m.group("ordinal") and days is not None else sources:
                    if near(r.value, s):
                        near_by.setdefault(s, r)
        if len(near_by) != 1:
            unverified.append(written)
            return written
        [(value, reading)] = near_by.items()
        fixed = write_like(value, reading)
        repairs.append(NumberRepair(original=body, repaired=fixed))
        return fixed + written[len(body) :]

    return WRITTEN.sub(check, text), repairs, unverified


def guard_numbers(
    notes: SegmentNotes,
    sources: set[float],
    language: str | None = None,
    days: set[float] | None = None,
) -> SegmentNotes:
    """Run `check_numbers` on every Map string except `terms`. Raw (non-JSON) segments are
    left alone: their text isn't notes, so there is nothing to keep consistent."""
    if notes.raw:
        return notes
    repairs: list[NumberRepair] = []
    unverified: list[str] = []

    def fix(text: str) -> str:
        fixed, r, u = check_numbers(text, sources, language, days)
        repairs.extend(r)
        unverified.extend(u)
        return fixed

    def fix_all(texts: list[str]) -> list[str]:
        return [fix(t) for t in texts]

    return notes.model_copy(
        update={
            "summary": fix(notes.summary),
            "key_points": fix_all(notes.key_points),
            "decisions": fix_all(notes.decisions),
            "action_items": [
                a.model_copy(update={"task": fix(a.task), "due": a.due and fix(a.due)})
                for a in notes.action_items
            ],
            "open_questions": fix_all(notes.open_questions),
            "number_repairs": repairs,
            "unverified_numbers": list(dict.fromkeys(unverified)),
        }
    )


# --- dates and dues (pure, §5.5, 8c) --------------------------------------------------------


def guard_dates(
    notes: SegmentNotes, source_text: str, notes_language: str, source_language: str
) -> SegmentNotes:
    """Map's dates and dues against the segment's transcript and slide text (D41). English
    notes of an English meeting: a date the segment doesn't say is repaired when one said
    date of its month is a digit edit away ("November 1" → "November 11"), else flagged
    (`dues.check_dates`); other languages' dates are left to the number guard. Then a due
    the segment doesn't say is dropped (`dues.due_said`, as Reduce does on the final notes,
    D40): Map copies its prompt's example due ("End of next week") onto tasks that have
    none, and Ask's documents showed it next to the real ones (8c a-f61)."""
    if notes.raw:
        return notes
    repairs = list(notes.number_repairs)
    unverified = list(notes.unverified_numbers)
    said_dates = dues.dates_in(source_text)
    english = notes_language == source_language == "en"

    def fix(text: str) -> str:
        if not english:
            return text
        fixed, r, u = dues.check_dates(text, said_dates)
        repairs.extend(NumberRepair(original=a, repaired=b) for a, b in r)
        unverified.extend(u)
        return fixed

    def fix_all(texts: list[str]) -> list[str]:
        return [fix(t) for t in texts]

    dropped: list[str] = []
    actions = []
    for a in notes.action_items:
        a = a.model_copy(update={"task": fix(a.task), "due": a.due and fix(a.due)})
        if a.due and not dues.due_said(a.due, source_text, notes_language, source_language):
            dropped.append(a.due)
            a = a.model_copy(update={"due": None})
        actions.append(a)
    return notes.model_copy(
        update={
            "summary": fix(notes.summary),
            "key_points": fix_all(notes.key_points),
            "decisions": fix_all(notes.decisions),
            "action_items": actions,
            "open_questions": fix_all(notes.open_questions),
            "number_repairs": repairs,
            "unverified_numbers": list(dict.fromkeys(unverified)),
            "dropped_dues": dropped,
        }
    )


# --- owners (pure, §5.5, 8d) ----------------------------------------------------------------

TASK_MATCH_MIN = 0.6  # Reduce's ACTION_MATCH_MIN: a decision a Map task already states
_BY_DUE = re.compile(r"\bby\s+(?P<due>[^.,;!?]+?)\s*[.!]?$", re.I)


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def guard_owners(
    notes: SegmentNotes,
    transcript: str,
    said: set[str],
    notes_language: str,
    source_language: str,
) -> tuple[SegmentNotes, list[str]]:
    """Tasks Map wrote without their owner (D42), English notes only; the changes, for the run
    log. A decision saying a person said in the segment will do something ("Tom will update the
    pricing page by the 2nd of November") becomes that person's action item, unless a Map task
    there already states it; its due only if the segment says it (`dues.due_said`). A task with
    no owner gets one from its own text ("Laura will clean up the shared drive …") or, in an
    English meeting, from the segment's sentence that matches it best, when that sentence gives
    it to one person and names no one else ("Maria, can you prepare the customer survey …?").
    Never from a neighbouring task (map.md's rule). `said`: `names.said_names` of the whole
    meeting, accents folded."""
    if notes.raw or notes_language != "en":
        return notes, []
    changes: list[str] = []
    actions = list(notes.action_items)
    decisions: list[str] = []
    for d in notes.decisions:
        found = names.subject_task(d, said)
        if (
            found is None
            or not names.said_folded(found[0], transcript)
            or any(match_score(d, a.task) >= TASK_MATCH_MIN for a in actions)
        ):
            decisions.append(d)
            continue
        owner, rest = found
        m = _BY_DUE.search(rest)
        due = m["due"] if m else None
        if due and not dues.due_said(due, transcript, notes_language, source_language):
            due = None
        actions.append(MapActionItem(owner=owner, task=_capital(rest), due=due))
        changes.append(f"decision → task: {owner}: {rest}")
    sentences = _SENTENCE_END.split(transcript) if source_language == "en" else []
    for i, a in enumerate(actions):
        if a.owner:
            continue
        owner, task = None, a.task
        if found := names.subject_task(a.task, said):
            if names.said_folded(found[0], transcript):
                owner, task = found[0], _capital(found[1])
        elif sentences:
            # ties: the longer sentence ("Criteria." matches a task's one shared word fully)
            best = max(sentences, key=lambda s: (match_score(a.task, s), len(s)))
            if match_score(a.task, best) >= TASK_MATCH_MIN:
                owner = names.addressee(best, said)
        if owner:
            actions[i] = a.model_copy(update={"owner": owner, "task": task})
            changes.append(f"owner from the segment: {owner}: {task}")
    if not changes:
        return notes, []
    return notes.model_copy(update={"decisions": decisions, "action_items": actions}), changes


# --- speech/slide conflicts (pure, §5.5) --------------------------------------------------

_LABEL_WORD = re.compile(r"[^\W\d_]{3,}")
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


def label_words(text: str) -> set[str]:
    return {w.lower() for w in _LABEL_WORD.findall(text)}


def table_rows(tables_md: list[str]) -> list[list[str]]:
    """Cells per row: Markdown tables, or vision's one-row "label\nvalue" strings (§5.2)."""
    rows: list[list[str]] = []
    for table in tables_md:
        if "|" not in table:
            rows.append([c.strip() for c in table.splitlines() if c.strip()])
            continue
        for line in table.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if any(cells) and not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
    return rows


def slide_conflicts(
    transcript: str, tables_md: list[str], language: str | None = None
) -> list[SlideConflict]:
    """Slide rows whose one value was never said, while the sentences naming the row's label
    (every label word in them) hold exactly one number that is on no slide: the spoken value
    for the same figure (the slide-conflict-en case: "approved budget … $80,000", slide row
    "Approved budget | $18,000"). Neither side is assumed right. Numbers are read the way the
    meeting's `language` writes them (6d-2: Spanish "4.650" on a slide is the 4650 said, not a
    conflict). Japanese text has no spaces, so its label words never match: no conflicts."""

    def nums(text: str) -> set[float]:
        return source_numbers_in(text, language)

    rows = table_rows(tables_md)
    on_slides = set().union(*(nums(c) for row in rows for c in row))
    spoken = nums(transcript)
    sentences = _SENTENCE_END.split(transcript)
    out: list[SlideConflict] = []
    for label, *cells in rows:
        words = label_words(label)
        values = [(c, nums(c)) for c in cells if nums(c)]
        if not words or len(values) != 1 or len(values[0][1]) != 1:
            continue
        written, [value] = values[0]
        if value in spoken:
            continue
        said = set().union(*(nums(s) for s in sentences if words <= label_words(s)))
        if len(said - on_slides) == 1:
            [other] = said - on_slides
            out.append(SlideConflict(label=label, spoken=other, slide=written))
    return out


# The word in the slide-conflict note, by target language (6d-1; English when missing). Like
# render's LABELS: a static table, so the note reads in the notes language.
SLIDE_WORD = {
    "ar": "الشريحة",
    "cs": "snímek",
    "de": "Folie",
    "el": "διαφάνεια",
    "en": "slide",
    "es": "diapositiva",
    "fa": "اسلاید",
    "fr": "diapositive",
    "he": "שקופית",
    "hi": "स्लाइड",
    "id": "slide",
    "it": "diapositiva",
    "ja": "スライド",
    "ko": "슬라이드",
    "nl": "dia",
    "pl": "slajd",
    "pt": "slide",
    "ro": "diapozitiv",
    "ru": "слайд",
    "tr": "slayt",
    "uk": "слайд",
    "vi": "trang chiếu",
    "zh": "幻灯片",
}


# A written number plus a currency or percent sign after it ("80.000 $", "37 %"), so the note
# goes after the sign, as it does after "$80,000".
_WRITTEN_UNIT = re.compile(WRITTEN.pattern + r"(?:\s?[$€£¥₹%])?")


def _written_like(cell: str, reading: Reading, source_language: str | None = None) -> str:
    """The slide cell's number in the separator style of the spoken value as Map wrote it:
    "$18,000" next to German "80.000 $" → "$18.000" (read in German, "18,000" is 18.0). The
    cell is read the way the meeting's language writes it (Spanish "18.000 $" = 18000)."""

    def restyle(m: re.Match[str]) -> str:
        [value] = source_numbers_in(m.group("body"), source_language) or [None]
        if value is None:
            return m.group(0)
        return write_like(value, reading) + m.group(0)[len(m.group("body")) :]

    return WRITTEN.sub(restyle, cell, count=1)


def mark_conflicts(
    text: str,
    conflicts: list[SlideConflict],
    language: str | None = None,
    cross_language: bool = False,
    source_language: str | None = None,
) -> tuple[str, list[SlideConflict]]:
    """After each spoken value of a conflict in a string that names its label, add the slide's
    value: "$80,000" → "$80,000 (slide: $18,000)". The spoken value is never replaced. The note
    is in the target `language` (SLIDE_WORD), and the spoken value is read the way it writes
    numbers. Notes in another language than the meeting's can't name the slide's label
    ("Quartalsbudget" for "Approved budget"), so there the spoken value alone finds the string
    (6d-1)."""
    used: list[SlideConflict] = []
    word = SLIDE_WORD.get(language or "en", SLIDE_WORD["en"])
    for c in conflicts:
        if not cross_language and not label_words(c.label) & label_words(text):
            continue

        def add(m: re.Match[str], c: SlideConflict = c) -> str:
            if m.string[m.end() :].startswith(f" ({word}: "):
                return m.group(0)  # marked already
            scale = SCALE.get(m.group("scale") or "", 1.0)
            for r in native_readings(readings(m.group("body")), language):
                if r.value * scale == c.spoken:
                    slide = _written_like(c.slide, r, source_language)
                    return f"{m.group(0)} ({word}: {slide})"
            return m.group(0)

        marked = _WRITTEN_UNIT.sub(add, text)
        if marked != text:
            text = marked
            used.append(c)
    return text, used


def mark_slide_conflicts(
    notes: SegmentNotes,
    conflicts: list[SlideConflict],
    language: str | None = None,
    cross_language: bool = False,
    source_language: str | None = None,
) -> SegmentNotes:
    """`mark_conflicts` on every Map string but `terms`; raw segments are left alone."""
    if notes.raw or not conflicts:
        return notes
    used: list[SlideConflict] = []

    def mark(text: str) -> str:
        marked, u = mark_conflicts(text, conflicts, language, cross_language, source_language)
        used.extend(c for c in u if c not in used)
        return marked

    return notes.model_copy(
        update={
            "summary": mark(notes.summary),
            "key_points": [mark(t) for t in notes.key_points],
            "decisions": [mark(t) for t in notes.decisions],
            "action_items": [
                a.model_copy(update={"task": mark(a.task)}) for a in notes.action_items
            ],
            "open_questions": [mark(t) for t in notes.open_questions],
            "slide_conflicts": used,
        }
    )


# --- model (MLX) ---------------------------------------------------------------------------


class TinyAya:
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        return generate(
            self.model,
            self.tokenizer,
            text,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


def load(path: Path) -> TinyAya:
    from mlx_lm import load as load_lm

    loaded = load_lm(str(path))  # (model, tokenizer); typed as maybe + config
    return TinyAya(loaded[0], loaded[1])


# --- stage ---------------------------------------------------------------------------------


def run_stage(ctx: StageContext) -> None:
    cfg = ctx.config.map
    budget = ctx.config.fuse.max_segment_tokens
    segments = schemas.load(ctx.run_dir, SegmentsFile).segments
    slides = {s.slide_id: s for s in schemas.load(ctx.run_dir, SlidesFile).slides}
    prompt = load_prompt("map")
    retry = f"{prompt}\n\n{load_prompt('map_retry')}"
    notes: list[SegmentNotes] = []
    retried = failed = 0
    said = {names.fold(n) for n in names.said_names(s.transcript_text for s in segments)}
    if segments:
        with ctx.model("map") as lm:
            for k, seg in enumerate(segments):
                ctx.report(k / len(segments), f"segment {k + 1}/{len(segments)}")
                seg_slides = [slide_text(slides[i]) for i in seg.slide_ids]
                values = prompt_values(seg, seg_slides, ctx.config.notes_language)
                size = lm.count_tokens(values["transcript"] + "".join(seg_slides))
                if size > budget:
                    ctx.log.warning("map: segment %d is %d tokens (budget %d)", k, size, budget)
                obj = None
                raw = ""
                for attempt, template in enumerate((prompt, retry)):
                    retried += attempt
                    raw = lm.generate(
                        fill(template, **values),
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        seed=ctx.config.seed,
                    )
                    if (obj := parse(raw)) is not None:
                        break
                if obj is None:
                    failed += 1
                    ctx.log.warning("map: segment %d kept as raw text: %r", k, raw[:200])
                seg_notes = make_notes(seg.seg_id, obj, raw)
                if cfg.number_guard:
                    sources = source_numbers(
                        seg.transcript_text, seg_slides, ctx.config.source_language
                    )
                    days = source_days(seg.transcript_text, seg_slides)
                    seg_notes = guard_numbers(seg_notes, sources, ctx.config.notes_language, days)
                    seg_notes = guard_dates(
                        seg_notes,
                        "\n".join([seg.transcript_text, *seg_slides]),
                        ctx.config.notes_language,
                        ctx.config.source_language,
                    )
                    for fix in seg_notes.number_repairs:
                        ctx.log.info("map: segment %d %s → %s", k, fix.original, fix.repaired)
                    if seg_notes.dropped_dues:
                        ctx.log.info(
                            "map: segment %d dues not said, dropped: %s",
                            k,
                            ", ".join(seg_notes.dropped_dues),
                        )
                    if seg_notes.unverified_numbers:
                        ctx.log.warning(
                            "map: segment %d numbers in no source: %s",
                            k,
                            ", ".join(seg_notes.unverified_numbers),
                        )
                    tables = [t for i in seg.slide_ids for t in slides[i].tables_md]
                    conflicts = slide_conflicts(
                        seg.transcript_text, tables, ctx.config.source_language
                    )
                    seg_notes = mark_slide_conflicts(
                        seg_notes,
                        conflicts,
                        ctx.config.notes_language,
                        ctx.config.notes_language != ctx.config.source_language,
                        ctx.config.source_language,
                    )
                    for c in seg_notes.slide_conflicts:
                        ctx.log.info(
                            "map: segment %d %s: said %s, slide %s", k, c.label, c.spoken, c.slide
                        )
                seg_notes, owners = guard_owners(
                    seg_notes,
                    seg.transcript_text,
                    said,
                    ctx.config.notes_language,
                    ctx.config.source_language,
                )
                for change in owners:
                    ctx.log.info("map: segment %d %s", k, change)
                notes.append(seg_notes)
    ctx.log.info("map: %d segments, %d retried, %d kept as raw text", len(notes), retried, failed)
    schemas.save(ctx.run_dir, MapFile(target_language=ctx.config.notes_language, notes=notes))


STAGE = Stage(
    name="map",
    run=run_stage,
    outputs=(MapFile.FILENAME,),
    format_version=MapFile.FORMAT_VERSION,
    config_keys=("seed", "target_language", "map", "models.map"),
    prompts=("map", "map_retry"),
    models=("map",),
)
