"""Map (PLAN §5.5): Tiny Aya Global on mlx-lm, one call per segment, JSON in the target language.

The prompt ends with the target-language instruction (Phase 0: with it only at the top, de and ja
notes came back in English). Invalid JSON is retried once; after that the raw text becomes the
summary with `raw: true`, so one bad segment never fails the stage.

The number guard then checks every number Map wrote with digits against the segment's own
transcript and slide text, and repairs or flags the ones that aren't there (§5.5, D20).
"""

import json
import re
from pathlib import Path
from typing import Any

from cohr import schemas
from cohr.config import language_name, load_prompt
from cohr.fuse import slide_text
from cohr.numbers import (
    SCALE,
    WRITTEN,
    Reading,
    native_readings,
    near,
    numbers_in,
    ordinals_in,
    readings,
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


def source_numbers(transcript: str, slides: list[str]) -> set[float]:
    return set().union(numbers_in(transcript), *(numbers_in(s) for s in slides))


def source_days(transcript: str, slides: list[str]) -> set[float]:
    """The ordinals said or shown ("the 22nd"): the only repair candidates for a Map ordinal."""
    return set().union(ordinals_in(transcript), *(ordinals_in(s) for s in slides))


def check_numbers(
    text: str, sources: set[float], language: str | None = None, days: set[float] | None = None
) -> tuple[str, list[NumberRepair], list[str]]:
    """Keep a number found in the sources; repair it when exactly one source number is near it
    (one digit edit, or a factor of 10/100/1000), in the written style; otherwise keep it as
    written and flag it. Repair candidates come only from the readings that follow the target
    `language`'s separators (`native_readings`), and for an ordinal ("2nd") only from `days`
    when given ("by the 22nd", not "12 seats"); keeping a number accepts any reading. A number
    with a scale suffix ("42.5k") or grouped with plain spaces ("42 500", which could also be two
    numbers) is only kept or flagged."""
    repairs: list[NumberRepair] = []
    unverified: list[str] = []

    def check(m: re.Match[str]) -> str:
        body, written = m.group("body"), m.group(0)
        reads = readings(body)
        scale = SCALE.get(m.group("scale") or "", 1.0)
        if not reads or any(r.value * scale in sources for r in reads):
            return written
        if " " in body and all(p.isdigit() and float(p) in sources for p in body.split(" ")):
            return written
        near_by: dict[float, Reading] = {}
        if scale == 1.0 and " " not in body:
            for r in native_readings(reads, language):
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


def slide_conflicts(transcript: str, tables_md: list[str]) -> list[SlideConflict]:
    """Slide rows whose one value was never said, while the sentences naming the row's label
    (every label word in them) hold exactly one number that is on no slide: the spoken value
    for the same figure (the slide-conflict-en case: "approved budget … $80,000", slide row
    "Approved budget | $18,000"). Neither side is assumed right."""
    rows = table_rows(tables_md)
    on_slides = set().union(*(numbers_in(c) for row in rows for c in row))
    spoken = numbers_in(transcript)
    sentences = _SENTENCE_END.split(transcript)
    out: list[SlideConflict] = []
    for label, *cells in rows:
        words = label_words(label)
        values = [(c, numbers_in(c)) for c in cells if numbers_in(c)]
        if not words or len(values) != 1 or len(values[0][1]) != 1:
            continue
        written, [value] = values[0]
        if value in spoken:
            continue
        said = set().union(*(numbers_in(s) for s in sentences if words <= label_words(s)))
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


def _written_like(cell: str, reading: Reading) -> str:
    """The slide cell's number in the separator style of the spoken value as Map wrote it:
    "$18,000" next to German "80.000 $" → "$18.000" (read in German, "18,000" is 18.0)."""

    def restyle(m: re.Match[str]) -> str:
        [value] = numbers_in(m.group("body")) or [None]
        if value is None:
            return m.group(0)
        return write_like(value, reading) + m.group(0)[len(m.group("body")) :]

    return WRITTEN.sub(restyle, cell, count=1)


def mark_conflicts(
    text: str,
    conflicts: list[SlideConflict],
    language: str | None = None,
    cross_language: bool = False,
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
                    return f"{m.group(0)} ({word}: {_written_like(c.slide, r)})"
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
) -> SegmentNotes:
    """`mark_conflicts` on every Map string but `terms`; raw segments are left alone."""
    if notes.raw or not conflicts:
        return notes
    used: list[SlideConflict] = []

    def mark(text: str) -> str:
        marked, u = mark_conflicts(text, conflicts, language, cross_language)
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
                    sources = source_numbers(seg.transcript_text, seg_slides)
                    days = source_days(seg.transcript_text, seg_slides)
                    seg_notes = guard_numbers(seg_notes, sources, ctx.config.notes_language, days)
                    for fix in seg_notes.number_repairs:
                        ctx.log.info("map: segment %d %s → %s", k, fix.original, fix.repaired)
                    if seg_notes.unverified_numbers:
                        ctx.log.warning(
                            "map: segment %d numbers in no source: %s",
                            k,
                            ", ".join(seg_notes.unverified_numbers),
                        )
                    tables = [t for i in seg.slide_ids for t in slides[i].tables_md]
                    conflicts = slide_conflicts(seg.transcript_text, tables)
                    seg_notes = mark_slide_conflicts(
                        seg_notes,
                        conflicts,
                        ctx.config.notes_language,
                        ctx.config.notes_language != ctx.config.source_language,
                    )
                    for c in seg_notes.slide_conflicts:
                        ctx.log.info(
                            "map: segment %d %s: said %s, slide %s", k, c.label, c.spoken, c.slide
                        )
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
