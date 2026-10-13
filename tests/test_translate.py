"""PLAN §5.6 "Other (beta)" translation (6d-1): numbered lines out, text back on its claim,
citations never touched."""

import re
from pathlib import Path

from cohr import render, schemas, translate
from cohr.config import BETA_LANGUAGES, TARGET_LANGUAGES, Config, load_prompt
from cohr.schemas import ActionItem, Citation, Claim, NotesDoc, NotesSections, TranslatedNotes
from tests.conftest import ctx_for, fake_manager, make_run

SEG0, SEG1, SLIDE2 = (
    Citation(kind="seg", id=0, t=12.0),
    Citation(kind="seg", id=1, t=301.5),
    Citation(kind="slide", id=2),
)


def _notes() -> NotesDoc:
    return NotesDoc(
        sections=NotesSections(
            summary=[Claim(text="The team met.", citations=[SEG0])],
            # one claim from two cited spans, with two segment citations and a slide
            key_points=[
                Claim(text="Signups rose to 4,650 and churn fell.", citations=[SEG0, SEG1, SLIDE2]),
                Claim(text="Not sourced.", citations=[]),
            ],
            decisions=[],
            action_items=[
                ActionItem(text="Record the video", citations=[SEG1], owner="Tom", due="Monday"),
                ActionItem(text="Fix the build", citations=[SEG0]),
            ],
            open_questions=[],
        ),
        passes=1,
        language="en",
        raw="",
    )


class FakeAya:
    """Translates by upper-casing the text after each marker (and a due date). A request of
    several lines whose first text is in `broken` loses its markers; a claim whose text is in
    `never` never comes back; `digits` maps a source number to the one written instead."""

    def __init__(
        self,
        broken: tuple[str, ...] = (),
        never: tuple[str, ...] = (),
        digits: dict[str, str] | None = None,
    ):
        self.broken, self.never, self.digits = broken, never, digits or {}
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        lines = re.findall(r"^<(\d+)> (.*)$", prompt, re.M)
        if lines[0][1] in self.broken and len(lines) > 1:
            return "Here is the translation:\n" + lines[0][1].upper()  # markers lost
        out = []
        for n, text in lines:
            if text in self.never:
                return "Sorry."
            for old, new in self.digits.items():
                text = text.replace(old, new)
            m = re.match(r"(\[owner: [^\]]*\] \[due: )([^\]]*)(\] )(.*)", text)
            due = m.group(2) if m and m.group(2) == "?" else (m.group(2).upper() if m else "")
            out.append(
                f"<{n}> "
                + (f"{m.group(1)}{due}{m.group(3)}{m.group(4).upper()}" if m else text.upper())
            )
        return "\n".join(out)

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
        return self(prompt)


def _citations(doc: NotesDoc | TranslatedNotes) -> list[list[list[Citation]]]:
    return [[c.citations for c in getattr(doc.sections, s)] for s in translate.SECTIONS]


def test_beta_languages_extend_r7b_without_overlap() -> None:
    assert not set(BETA_LANGUAGES) & set(TARGET_LANGUAGES)
    assert len(BETA_LANGUAGES) + len(TARGET_LANGUAGES) == 67  # the Tiny Aya model card
    assert Config(target_language="sw").notes_language == "en"
    assert Config(target_language="de").notes_language == "de"
    assert Config(target_language="sw").translated and not Config(target_language="de").translated


def test_lines_out_and_back() -> None:
    lines = translate.numbered_lines(_notes().sections.action_items, start=3)
    assert lines == (
        "<3> [owner: Tom] [due: Monday] Record the video\n<4> [owner: ?] [due: ?] Fix the build"
    )
    assert translate.parse_lines("<2> b\n<1> a", [1, 2]) == {1: "a", 2: "b"}
    assert translate.parse_lines("<1> a", [1, 2]) is None  # one missing
    assert translate.parse_lines("<1> a\n<1> a\n<2> b", [1, 2]) is None  # repeated
    assert translate.parse_lines("<1> a\n<2> b\n<3> c", [1, 2]) is None  # one added
    assert translate.parse_lines("<1>\n<2> b", [1, 2]) is None  # empty text


def test_translation_keeps_every_citation_and_label() -> None:
    aya = FakeAya()
    out = translate.translate_sections(_notes(), "sw", aya, lambda _m: None)
    assert len(aya.prompts) == 3  # one per non-empty section: no retry on a good reply
    assert _citations(out) == _citations(_notes())
    assert out.sections.key_points[0].text == "SIGNUPS ROSE TO 4,650 AND CHURN FELL."
    tom, fix = out.sections.action_items
    assert (tom.text, tom.owner, tom.due) == ("RECORD THE VIDEO", "Tom", "MONDAY")
    assert (fix.owner, fix.due) == (None, None)  # "?" stays unknown
    assert out.language == "sw" and out.source_language == "en" and out.untranslated == 0
    prompt = aya.prompts[2]
    assert "\n<1> [owner: Tom] [due: Monday] Record the video\n" in prompt
    assert "## " not in prompt  # section markers are never sent
    assert prompt.rstrip().endswith("translated into Swahili.")
    assert load_prompt("translate").rstrip().endswith("translated into {target_language}.")


def test_a_bad_reply_is_retried_one_claim_at_a_time() -> None:
    logged: list[str] = []
    first = _notes().sections.key_points[0].text
    aya = FakeAya(broken=(first,), never=("Not sourced.",))
    out = translate.translate_sections(_notes(), "sw", aya, logged.append)
    # key_points: one section reply, then one per claim; claim 2 never comes back
    assert len(aya.prompts) == 1 + (1 + 2) + 1
    assert "<2> Not sourced." in aya.prompts[3]  # the retry keeps the claim's own marker
    assert out.sections.key_points[0].text == "SIGNUPS ROSE TO 4,650 AND CHURN FELL."
    assert out.sections.key_points[1].text == "Not sourced."  # kept in English
    assert out.untranslated == 1 and logged == ["translate: key_points item 2 kept in English"]
    assert _citations(out) == _citations(_notes())


def _run(tmp_path: Path, target: str) -> Path:
    config = Config(target_language=target)
    run_dir = make_run(tmp_path, config=config)
    schemas.save(run_dir, _notes())
    return run_dir


def test_stage_writes_translated_notes_for_a_beta_target(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "sw")
    manager = fake_manager(Config(), tmp_path, map=FakeAya())
    translate.run_stage(ctx_for(run_dir, "translate", manager))
    doc = schemas.load(run_dir, TranslatedNotes)
    assert doc.language == "sw" and _citations(doc) == _citations(_notes())
    assert render.shown_notes(run_dir) == doc
    assert "THE TEAM MET. [0:12]" in render.render(doc, [], [], "portable")


def test_stage_does_nothing_for_r7b_languages(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "de")
    (run_dir / TranslatedNotes.FILENAME).write_text("stale")  # from an earlier beta target
    manager = fake_manager(Config(), tmp_path, map=FakeAya())
    translate.run_stage(ctx_for(run_dir, "translate", manager))
    assert not (run_dir / TranslatedNotes.FILENAME).exists()
    assert manager.loaded() == []
    assert render.shown_notes(run_dir) == _notes()


def test_a_line_that_changes_a_number_is_not_kept() -> None:
    same = translate.same_numbers
    assert same("$42,500 and 0.74, in Q3", "42.500 $ na 0,74, Q3", "en")  # style may change
    assert same("up from 233 to 390", "390, kutoka 233", "en")  # order may change
    # 6d-1, real Tiny Aya replies: a dropped digit, a decimal point added, a number lost
    assert not same("Budget: $42,500 approved", "Bajeti: $4,500 imepitishwa", "en")
    assert not same("34% week-2 return rate", "3.4% kiwango cha kurudi", "en")
    assert not same("390 open tickets, 233 last week", "tikiti 390 zimefunguliwa", "en")
    # The line is retried alone, then kept in English; its sibling is still translated.
    logged: list[str] = []
    aya = FakeAya(digits={"4,650": "4,500"})
    out = translate.translate_sections(_notes(), "sw", aya, logged.append)
    assert out.sections.key_points[0].text == "Signups rose to 4,650 and churn fell."
    assert out.sections.key_points[1].text == "NOT SOURCED."
    assert out.untranslated == 1 and logged == ["translate: key_points item 1 kept in English"]
    kept = translate.translate_sections(_notes(), "sw", FakeAya(digits={"4,650": "4.650"}), print)
    assert kept.untranslated == 0  # a German-style separator is the same number
    assert kept.sections.key_points[0].text == "SIGNUPS ROSE TO 4.650 AND CHURN FELL."


def test_reply_noise_and_translated_labels() -> None:
    # Tiny Aya ends its reply with the template's end token and translates the labels.
    reply = "\n<1> [mmiliki: Tom] [tarehe: Jumatatu] Rekodi video<|END_RESPONSE|>"
    assert translate.parse_lines(reply, [1]) == {
        1: "[mmiliki: Tom] [tarehe: Jumatatu] Rekodi video"
    }
    tom = _notes().sections.action_items[0]
    back = translate.with_text(tom, "[mmiliki: Thomas] [tarehe: Jumatatu] Rekodi video")
    assert isinstance(back, ActionItem)
    assert (back.text, back.owner, back.due) == ("Rekodi video", "Tom", "Jumatatu")
    assert back.citations == tom.citations
