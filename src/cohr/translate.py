"""Translate (PLAN §5.6, 6d-1): English notes → an "Other (beta)" target language.

For a target outside R7B's 23 languages, Map and Reduce write English (`Config.notes_language`)
and Tiny Aya translates notes.json section by section into notes_translated.json. Citations
never pass through the model: each claim goes out as one numbered line ("<3> text"), the
number is the only marker, and the translated text goes back on the claim it came from, so
every claim keeps its citations, in the same section and order. Section markers are never
sent; the "[owner: …] [due: …]" labels are asked to stay in English (Tiny Aya translates them
anyway, so any two leading bracket labels are read back). A translated line must keep the
value of every number in its source line (Tiny Aya wrote "$4,500" for "$42,500" and "3.4%" for
"34%", 6d-1). A
reply that doesn't give back every marker once, or changes a number, is retried one claim at
a time; a claim that still fails keeps its English text (counted in `untranslated`, logged).
Nothing is written for R7B's languages.
"""

import re
from collections.abc import Callable, Sequence

from cohr import schemas
from cohr.config import language_name, load_prompt
from cohr.mapper import fill
from cohr.numbers import SCALE, WRITTEN, native_readings, readings
from cohr.pipeline import Stage, StageContext
from cohr.reduce import SECTIONS
from cohr.schemas import ActionItem, Claim, NotesDoc, NotesSections, TranslatedNotes

_LINE = re.compile(r"^\s*<(\d+)>[ \t]*(.*?)\s*$", re.M)
# "[owner: …] [due: …] task", label words in any language ("[mmiliki: …] [tarehe: …]")
_LABELS = re.compile(r"^\[[^\]:]*:\s*([^\]]*)\]\s*\[[^\]:]*:\s*([^\]]*)\]\s*(.*)$", re.S)
_SPECIAL = re.compile(r"<\|[A-Z_]+\|>")  # chat-template tokens left in a reply
_UNKNOWN = {"", "?"}

type Generate = Callable[[str], str]
type LogFn = Callable[[str], None]


def claim_line(claim: Claim) -> str:
    """One claim as the text sent for translation: an action item with its labels."""
    if isinstance(claim, ActionItem):
        return f"[owner: {claim.owner or '?'}] [due: {claim.due or '?'}] {claim.text}"
    return claim.text


def numbered_lines(claims: Sequence[Claim], start: int = 1) -> str:
    return "\n".join(f"<{start + k}> {claim_line(c)}" for k, c in enumerate(claims))


def _values(text: str, language: str | None) -> list[set[float]]:
    out = []
    for m in WRITTEN.finditer(text):
        scale = SCALE.get(m.group("scale") or "", 1.0)
        reads = native_readings(readings(m.group("body")), language)
        out.append({r.value * scale for r in reads})
    return [v for v in out if v]


def same_numbers(source: str, translated: str, source_language: str) -> bool:
    """Every number of the source line (read the source language's way) is in the translated
    line, which has no other: "$42,500" → "42.500 $" is the same number (a reading fits),
    "$4,500" or "4.25" isn't, and "34%" → "3.4%" isn't either."""
    want = _values(source, source_language)
    have = _values(translated, None)  # any separator style
    if len(want) != len(have):
        return False
    for values in want:
        hit = next((k for k, h in enumerate(have) if values & h), None)
        if hit is None:
            return False
        have.pop(hit)
    return True


def parse_lines(reply: str, ids: Sequence[int]) -> dict[int, str] | None:
    """{marker: text} when the reply gives back every marker in `ids` exactly once, with text,
    and no other marker; otherwise None (the caller retries)."""
    found = [(int(m.group(1)), m.group(2)) for m in _LINE.finditer(_SPECIAL.sub("", reply))]
    numbers = [n for n, _ in found]
    if sorted(numbers) != sorted(ids) or any(not text for _, text in found):
        return None
    return dict(found)


def with_text(claim: Claim, line: str) -> Claim:
    """The claim with its translated text; citations untouched. An action item's labels are
    read back: the due date may be translated, the owner (a name) is always kept."""
    if not isinstance(claim, ActionItem):
        return claim.model_copy(update={"text": line})
    m = _LABELS.match(line)
    if m is None:  # labels dropped: the whole line is the task
        return claim.model_copy(update={"text": line})
    due = m.group(2).strip()
    return claim.model_copy(
        update={
            "text": m.group(3).strip() or claim.text,
            "due": claim.due if claim.due is None or due in _UNKNOWN else due,
        }
    )


def translate_sections(
    notes: NotesDoc, target: str, generate: Generate, log: LogFn
) -> TranslatedNotes:
    """Pure apart from `generate(prompt) -> reply`: every section, then single-claim retries.
    `target` is a language code."""
    template = load_prompt("translate")

    def ask(claims: Sequence[Claim], start: int) -> dict[int, str]:
        """Translated lines that kept every number, by marker."""
        reply = generate(fill(template, **names, lines=numbered_lines(claims, start)))
        raw.append(reply)
        ids = range(start, start + len(claims))
        texts = parse_lines(reply, ids) or {}
        sources = dict(zip(ids, (claim_line(c) for c in claims), strict=True))
        return {k: t for k, t in texts.items() if same_numbers(sources[k], t, notes.language)}

    names = {
        "source_language": language_name(notes.language),
        "target_language": language_name(target),
    }
    out: dict[str, list[Claim]] = {}
    raw: list[str] = []
    untranslated = 0
    for name in SECTIONS:
        claims: list[Claim] = list(getattr(notes.sections, name))
        if not claims:
            out[name] = []
            continue
        texts = ask(claims, 1)
        done: list[Claim] = []
        for k, claim in enumerate(claims, start=1):
            if k not in texts:  # one claim at a time, same marker
                texts.update(ask([claim], k))
            if k in texts:
                done.append(with_text(claim, texts[k]))
            else:
                untranslated += 1
                log(f"translate: {name} item {k} kept in {names['source_language']}")
                done.append(claim)
        out[name] = done
    return TranslatedNotes(
        sections=NotesSections.model_validate(out),
        language=target,
        source_language=notes.language,
        untranslated=untranslated,
        raw=raw,
    )


def run_stage(ctx: StageContext) -> None:
    path = ctx.run_dir / TranslatedNotes.FILENAME
    if not ctx.config.translated:
        path.unlink(missing_ok=True)  # the target changed back to one of R7B's languages
        return
    cfg = ctx.config.translate
    notes = schemas.load(ctx.run_dir, NotesDoc)
    target = ctx.config.target_language
    with ctx.model("map") as lm:

        def generate(prompt: str) -> str:
            ctx.check_cancel()
            return lm.generate(
                prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature, seed=ctx.config.seed
            )

        doc = translate_sections(notes, target, generate, ctx.log.warning)
    ctx.log.info(
        "translate: %s → %s, %d replies, %d claims kept in %s",
        notes.language,
        target,
        len(doc.raw),
        doc.untranslated,
        notes.language,
    )
    schemas.save(ctx.run_dir, doc)


STAGE = Stage(
    name="translate",
    run=run_stage,
    outputs=(),  # notes_translated.json exists only for "Other (beta)" targets
    format_version=TranslatedNotes.FORMAT_VERSION,
    config_keys=("seed", "target_language", "translate", "models.map"),
    prompts=("translate",),
    models=("map",),
)
