"""PLAN §5.6: localized headings, portable vs in-app citations, Sources appendix."""

from pathlib import Path

from cohr import render, schemas
from cohr.config import TARGET_LANGUAGES
from cohr.reduce import SECTIONS
from cohr.schemas import (
    ActionItem,
    Citation,
    Claim,
    NotesDoc,
    NotesSections,
    Segment,
    SegmentsFile,
    Slide,
    SlidesFile,
)
from tests.conftest import ctx_for, make_run

SEGMENTS = [
    Segment(
        seg_id=0,
        t0=0.0,
        t1=300.0,
        chunk_ids=[0],
        transcript_text="Welcome everyone. " * 20,
        slide_ids=[0, 1],
    ),
    Segment(seg_id=1, t0=300.0, t1=410.0, chunk_ids=[1], transcript_text="Short.", slide_ids=[]),
]
SLIDES = [
    Slide(
        slide_id=k,
        frame_id=k,
        t=None,
        t_source="none",
        title=title,
        text_md="Approved: $14,500",
        tables_md=[],
        chart_notes="",
        visual_notes="",
    )
    for k, title in enumerate(["Q4 Budget", "Plan", ""])
]


def _notes(language: str = "en") -> NotesDoc:
    seg0 = Citation(kind="seg", id=0, t=767.4)
    return NotesDoc(
        sections=NotesSections(
            summary=[
                Claim(text="Budget approved.", citations=[seg0, Citation(kind="slide", id=0)]),
                Claim(text="Not sourced.", citations=[]),
            ],
            key_points=[Claim(text="Slide only", citations=[Citation(kind="slide", id=2)])],
            decisions=[],
            action_items=[
                ActionItem(
                    text="Screenshots",
                    citations=[Citation(kind="seg", id=1, t=None)],
                    owner="Priya",
                    due="20 October",
                ),
                ActionItem(text="Close tickets", citations=[seg0]),
            ],
            open_questions=[],
        ),
        passes=1,
        language=language,
        raw="",
    )


def test_labels_cover_every_target_language() -> None:
    assert set(render.LABELS) == set(TARGET_LANGUAGES)
    for words in render.LABELS.values():
        assert all(words[k] for k in (*SECTIONS, "sources", "owner", "due", "uncited"))
    assert render.labels("xx") == render.LABELS["en"]


def test_portable_markdown() -> None:
    md = render.render(_notes(), SEGMENTS, SLIDES, "portable")
    assert md.startswith(
        "## Summary\n\nBudget approved. [12:47] [slide 1] Not sourced. [uncited]\n"
    )
    assert "## Key points\n\n- Slide only [slide 3]\n" in md
    assert "## Decisions\n\n—\n" in md
    assert "- Screenshots (owner: Priya; due: 20 October) [5:00]\n" in md  # t None → seg start
    assert "- Close tickets [12:47]\n" in md
    assert "](#" not in md  # no in-app anchors in the download
    sources = md.split("## Sources\n\n", 1)[1]
    lines = sources.strip().splitlines()
    assert lines[0].startswith("- [0:00–5:00] · slide 1, slide 2: Welcome everyone.")
    assert lines[0].endswith(" …") and len(lines[0]) < 260
    assert lines[1] == "- [5:00–6:50]: Short."
    assert lines[2:] == ["- [slide 1]: Q4 Budget", "- [slide 3]: Approved: $14,500"]


def test_app_markdown_links_to_sources() -> None:
    md = render.render(_notes(), SEGMENTS, SLIDES, "app")
    assert "Budget approved. [12:47](#seg-0) [slide 1](#slide-0)" in md
    assert ":gray[Not sourced.] :gray-badge[uncited]" in md
    assert "## Sources" not in md


def test_app_markdown_badges_and_escapes() -> None:
    notes = _notes()
    notes.sections.decisions.append(Claim(text="Cap at $14,500 [final]", citations=[]))
    md = render.render(notes, SEGMENTS, SLIDES, "app")
    assert "## Action items :orange-badge[2]\n" in md
    assert "## Decisions :green-badge[1]\n" in md
    assert "## Open questions\n" in md  # empty: no badge
    assert "## Summary\n" in md
    assert ":gray[Cap at \\$14,500 \\[final\\]]" in md
    portable = render.render(notes, SEGMENTS, SLIDES, "portable")
    assert "- Cap at $14,500 [final] [uncited]" in portable  # the download stays plain


def test_localized_headings_from_the_table() -> None:
    md = render.render(_notes("de"), SEGMENTS, SLIDES)
    assert md.startswith("## Zusammenfassung\n")
    assert "## Aufgaben" in md and "## Quellen" in md
    assert "(verantwortlich: Priya; fällig: 20 October)" in md
    assert "Not sourced. [ohne Quelle]" in md
    assert "## SUMMARY" not in md


def test_stage_writes_notes_md(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    schemas.save(run_dir, SlidesFile(slides=SLIDES))
    schemas.save(run_dir, SegmentsFile(segments=SEGMENTS, unaligned_slide_ids=[2]))
    schemas.save(run_dir, _notes())
    render.run_stage(ctx_for(run_dir, "render"))
    text = (run_dir / "notes.md").read_text(encoding="utf-8")
    assert text == render.render(_notes(), SEGMENTS, SLIDES, "portable")


def test_every_citation_survives_both_modes() -> None:
    notes = _notes()
    cited = [
        c for name in SECTIONS for claim in getattr(notes.sections, name) for c in claim.citations
    ]
    by_seg = {s.seg_id: s for s in SEGMENTS}
    portable = render.render(notes, SEGMENTS, SLIDES, "portable").split("## Sources")[0]
    app = render.render(notes, SEGMENTS, SLIDES, "app")
    for mode, md in (("portable", portable), ("app", app)):
        labels = [render.citation_md(c, by_seg, mode) for c in cited]
        assert sum(md.count(lbl) for lbl in set(labels)) == len(labels)
