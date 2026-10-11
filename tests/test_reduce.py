"""PLAN §5.6: documents, section markers, citation rules, time refinement, action items."""

from pathlib import Path

import pytest

from cohr import reduce, schemas
from cohr.config import Config, ReduceConfig
from cohr.schemas import (
    MapActionItem,
    MapFile,
    NotesDoc,
    Segment,
    SegmentNotes,
    SegmentsFile,
    Slide,
    SlidesFile,
    Transcript,
    TranscriptItem,
)
from tests.conftest import ctx_for, fake_manager, make_run

# Shaped like the Phase 0 reply: wrapper tokens, marker on the wrapper's line, [i,j] lists.
REPLY = (
    "<|START_RESPONSE|>## SUMMARY\n"
    "<co>The team approved the 14,500 budget.</co: 0:[0,2]> "
    "The beta launches <co>on 3 November</co: 0:[1]>. An extra sentence with no source.\n"
    "## KEY_POINTS\n"
    "- <co>Retention rose to 41%</co: 0:[3]>\n"
    "- Mentioned <co>twice</co: 0:[0]> and <co>again</co: 0:[0]>\n"
    "## DECISIONS\n"
    "- <co>Launch the beta on 3 November</co: 0:[1]>\n"
    "- Bad index <co>here</co: 0:[9]>\n"
    "## ACTION_ITEMS\n"
    "- [owner: Priya] [due: 20 October] <co>Prepare the store screenshots</co: 0:[1]>\n"
    "- <co>Tom: release notes by end of next week</co: 0:[1]>\n"
    "- [owner: ?] [due: ?] <co>Close duplicate tickets</co: 0:[0]>\n"
    "## OPEN_QUESTIONS\n"
    "- none\n"
    "<|END_RESPONSE|>"
)
TARGETS: list[reduce.Target] = [("seg", 0), ("seg", 1), ("slide", 0), ("slide", 2)]


def _items() -> list[TranscriptItem]:
    return [
        TranscriptItem(chunk_id=0, t0=0.0, t1=20.0, text="Welcome everyone to the Atlas sync."),
        TranscriptItem(chunk_id=1, t0=20.0, t1=40.0, text="We approved the 14,500 budget today."),
        TranscriptItem(chunk_id=2, t0=300.0, t1=320.0, text="Beta launch is on 3 November."),
        TranscriptItem(chunk_id=3, t0=320.0, t1=340.0, text="Tom writes release notes."),
    ]


def _segments() -> list[Segment]:
    items = _items()
    return [
        Segment(
            seg_id=k,
            t0=items[2 * k].t0,
            t1=items[2 * k + 1].t1,
            chunk_ids=[2 * k, 2 * k + 1],
            transcript_text="\n".join(i.text for i in items[2 * k : 2 * k + 2]),
            slide_ids=[0] if k == 0 else [],
        )
        for k in range(2)
    ]


def _map() -> dict[int, SegmentNotes]:
    def notes(seg_id: int, actions: list[MapActionItem]) -> SegmentNotes:
        return SegmentNotes(
            seg_id=seg_id,
            summary=f"summary {seg_id}",
            key_points=["kp"],
            decisions=[],
            action_items=actions,
            open_questions=[],
            terms=[],
        )

    return {
        0: notes(0, []),
        1: notes(1, [MapActionItem(owner="Tom", task="release notes", due="end of next week")]),
    }


def _slide(sid: int, title: str = "", text: str = "") -> Slide:
    return Slide(
        slide_id=sid,
        frame_id=sid,
        t=None,
        t_source="none",
        title=title,
        text_md=text,
        tables_md=[],
        chart_notes="",
        visual_notes="",
    )


def _refine() -> reduce.Refine:
    return reduce.segment_refiner(_segments(), _items(), 0.1)


# --- documents -----------------------------------------------------------------------------


def test_documents_cover_every_segment_and_slide() -> None:
    slides = [_slide(0, "Q4 Budget", "Approved: $14,500"), _slide(1, "", "Hiring")]
    docs, targets = reduce.documents(_segments(), _map(), slides)
    assert targets == [("seg", 0), ("seg", 1), ("slide", 0), ("slide", 1)]
    assert [d["title"] for d in docs] == [
        "0:00–0:40 · slides 1",
        "5:00–5:40",
        "slide 1: Q4 Budget",
        "slide 2",
    ]
    assert docs[1]["snippet"].startswith("Summary: summary 1\nKey points:\n- kp\nAction items:")
    assert "[owner: Tom] [due: end of next week] release notes" in docs[1]["snippet"]
    assert docs[1]["snippet"].endswith(
        "Transcript: Beta launch is on 3 November.\nTom writes release notes."
    )
    assert docs[2]["snippet"] == "Q4 Budget\nApproved: $14,500"


def test_documents_trim_transcripts_and_tolerate_missing_map_notes() -> None:
    docs, _ = reduce.documents(_segments(), {}, [], max_transcript_chars=20)
    assert docs[0]["snippet"] == "Transcript: Welcome everyone to …"


# --- markers -------------------------------------------------------------------------------


def test_strip_wrapper() -> None:
    assert reduce.strip_wrapper("x<|START_RESPONSE|>## SUMMARY\nhi<|END_RESPONSE|>y") == (
        "## SUMMARY\nhi"
    )
    assert reduce.strip_wrapper("plain <|END_OF_TURN_TOKEN|>") == "plain "


def test_split_sections_on_fixed_markers() -> None:
    split = reduce.split_sections(reduce.strip_wrapper(REPLY))
    assert split.missing == [] and split.unknown == []
    assert split.sections["open_questions"] == "- none"
    assert split.sections["decisions"].startswith("- <co>Launch")


def test_localized_headings_are_never_parsed() -> None:
    text = "Intro line\n## ZUSAMMENFASSUNG\nDas Budget.\n## decisions\n- Ja\n## Key Points:\n- k"
    split = reduce.split_sections(text)
    assert split.unknown == ["ZUSAMMENFASSUNG"]
    assert split.missing == ["summary", "action_items", "open_questions"]
    assert split.sections["summary"] == "Intro line\nDas Budget."
    assert split.sections["decisions"] == "- Ja"
    assert split.sections["key_points"] == "- k"  # case and spaces normalized, still English


# --- claims and citations ------------------------------------------------------------------


def test_doc_indices() -> None:
    assert reduce.doc_indices("0:[1,2]") == [1, 2]
    assert reduce.doc_indices(" 0:[3]") == [3]
    assert reduce.doc_indices("0:[1],0:[4]") == [1, 4]
    assert reduce.doc_indices("0:5") == [5]


def test_summary_splits_into_sentences_and_keeps_spans_whole() -> None:
    text = reduce.split_sections(reduce.strip_wrapper(REPLY)).sections["summary"]
    claims = reduce.claims(text, sentences=True)
    assert [c.text for c in claims] == [
        "The team approved the 14,500 budget.",
        "The beta launches on 3 November.",
        "An extra sentence with no source.",
    ]
    assert claims[0].spans == [("The team approved the 14,500 budget.", [0, 2])]
    assert claims[2].spans == []


def test_cjk_sentences_split_without_spaces() -> None:
    claims = reduce.claims("<co>予算を承認。</co: 0:[0]>ベータは11月3日。", sentences=True)
    assert [(c.text, len(c.spans)) for c in claims] == [
        ("予算を承認。", 1),
        ("ベータは11月3日。", 0),
    ]


def test_list_items_with_continuation_lines() -> None:
    claims = reduce.claims(
        "- first <co>a</co: 0:[0]>\n  continued\n* second\n1. third\n- none", sentences=False
    )
    assert [c.text for c in claims] == ["first a continued", "second", "third"]


def test_resolve_drops_bad_indices_and_dedupes() -> None:
    raw = reduce.RawClaim("x", [("budget", [0, 9]), ("again", [0, 2]), ("more", [4])])
    claim, dropped = reduce.resolve(raw, TARGETS, lambda _sid, _span: 1.0)
    assert dropped == 2  # 9 and 4 are past the 4 documents
    assert [(c.kind, c.id, c.t) for c in claim.citations] == [("seg", 0, 1.0), ("slide", 0, None)]


def test_markers_wrapped_in_citations_still_split() -> None:
    reply = (
        "## <co>SUMMARY</co: 0:[0]>\nIntro.\n"
        "## <co>DECISIONS</co: 0:[0]>\n- <co>Ship it.</co: 0:[0]>"
    )
    split = reduce.split_sections(reply)
    assert split.sections["decisions"] == "- <co>Ship it.</co: 0:[0]>"
    assert split.sections["summary"] == "Intro." and split.unknown == []


def test_refine_time_picks_the_best_chunk_or_falls_back() -> None:
    items = _items()[:2]
    assert reduce.refine_time("approved the 14,500 budget", items, 0.0, 0.1) == 20.0
    assert reduce.refine_time("completely unrelated words", items, 7.0, 0.1) == 7.0
    assert reduce.refine_time("", items, 7.0, 0.1) == 7.0


def test_refine_time_moves_to_the_sentence_inside_a_long_chunk() -> None:
    text = "Welcome back everyone. The approved budget is 59,500 dollars. Thanks."
    chunk = TranscriptItem(chunk_id=0, t0=100.0, t1=130.0, text=text)
    t = reduce.refine_time("Approved budget: $59,500", [chunk], 0.0, 0.1)
    assert t == round(100.0 + 30.0 * text.index("The approved") / len(text), 2)
    assert reduce.refine_time("welcome back everyone", [chunk], 0.0, 0.1) == 100.0
    assert reduce.sentence_offset({"budget"}, "No match here") == 0.0


def test_action_item_format_and_map_fallback() -> None:
    notes = _map()
    claim, _ = reduce.resolve(
        reduce.RawClaim("[owner: Priya] [due: 20 October] Screenshots", [("s", [1])]),
        TARGETS,
        _refine(),
    )
    item = reduce.action_item(claim, notes)
    assert (item.text, item.owner, item.due) == ("Screenshots", "Priya", "20 October")

    unknown = reduce.action_item(
        claim.model_copy(update={"text": "[owner: ?] [due: ?] Close tickets"}), notes
    )
    assert (unknown.text, unknown.owner, unknown.due) == ("Close tickets", None, None)

    fallback = reduce.action_item(
        claim.model_copy(update={"text": "Tom: release notes by end of next week"}), notes
    )
    assert (fallback.owner, fallback.due) == ("Tom", "end of next week")

    no_match = reduce.action_item(claim.model_copy(update={"text": "Book a room"}), notes)
    assert (no_match.owner, no_match.due) == (None, None)

    uncited = reduce.action_item(
        claim.model_copy(update={"text": "release notes", "citations": []}), notes
    )
    assert uncited.owner is None  # only cited segments' action items are candidates


def test_build_notes_end_to_end() -> None:
    doc, stats = reduce.build_notes(REPLY, TARGETS, _refine(), _map(), "en")
    s = doc.sections
    assert doc.raw == REPLY and doc.passes == 1 and doc.language == "en"
    budget = s.summary[0].citations
    assert [(c.kind, c.id) for c in budget] == [("seg", 0), ("slide", 0)]
    assert budget[0].t == 20.0  # refined onto the chunk that says it
    assert s.summary[1].citations[0].t == 300.0
    assert s.summary[2].citations == []  # uncited: kept, never given a citation
    assert [c.text for c in s.key_points] == ["Retention rose to 41%", "Mentioned twice and again"]
    assert [(c.kind, c.id) for c in s.key_points[0].citations] == [("slide", 2)]
    assert len(s.key_points[1].citations) == 1
    assert s.decisions[1].text == "Bad index here" and s.decisions[1].citations == []
    assert [(a.text, a.owner, a.due) for a in s.action_items] == [
        ("Prepare the store screenshots", "Priya", "20 October"),
        ("Tom: release notes by end of next week", "Tom", "end of next week"),
        ("Close duplicate tickets", None, None),
    ]
    assert s.open_questions == []
    assert (stats.claims, stats.uncited, stats.dropped) == (10, 2, 1)
    assert stats.missing_markers == [] and stats.unknown_markers == []


def test_build_notes_without_markers_keeps_everything_in_summary() -> None:
    doc, stats = reduce.build_notes(
        "Just text <co>cited</co: 0:[0]>.", TARGETS, _refine(), {}, "de"
    )
    assert [c.text for c in doc.sections.summary] == ["Just text cited."]
    assert stats.missing_markers == list(reduce.SECTIONS)


# --- stage ---------------------------------------------------------------------------------


class FakeR7B:
    def __init__(self, reply: str, tokens_per_char: float = 0.0):
        self.reply = reply
        self.tokens_per_char = tokens_per_char
        self.instructions: list[str] = []
        self.docs: list[list[dict[str, str]]] = []
        self.calls: list[dict[str, float]] = []

    def prompt(self, instruction: str, docs: list[dict[str, str]]) -> str:
        self.instructions.append(instruction)
        self.docs.append(docs)
        return instruction + "".join(d["snippet"] for d in docs)

    def count_tokens(self, text: str) -> int:
        return int(len(text) * self.tokens_per_char)

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
        self.calls.append({"max_tokens": max_tokens, "temperature": temperature, "seed": seed})
        return self.reply


def _run(tmp_path: Path, config: Config, slides: list[Slide] | None = None) -> Path:
    run_dir = make_run(tmp_path, config=config)
    slides = (
        [_slide(0, "Q4 Budget"), _slide(1, "Plan"), _slide(2, "Retention")]
        if slides is None
        else slides
    )
    schemas.save(run_dir, SlidesFile(slides=slides))
    schemas.save(run_dir, SegmentsFile(segments=_segments(), unaligned_slide_ids=[]))
    schemas.save(
        run_dir, MapFile(target_language=config.target_language, notes=list(_map().values()))
    )
    schemas.save(run_dir, Transcript(language="en", runtime="mlx-speech", items=_items()))
    return run_dir


def test_stage_writes_notes(tmp_path: Path) -> None:
    config = Config(seed=3, target_language="ja")
    run_dir = _run(tmp_path, config)
    lm = FakeR7B(REPLY.replace("0:[3]", "0:[4]"))  # slide 2 is document 4 here
    reduce.run_stage(ctx_for(run_dir, "reduce", fake_manager(Config(), tmp_path, reduce=lm)))

    doc = schemas.load(run_dir, NotesDoc)
    assert doc.language == "ja"
    assert [(c.kind, c.id) for c in doc.sections.key_points[0].citations] == [("slide", 2)]
    assert lm.calls == [{"max_tokens": 2048, "temperature": 0.0, "seed": 3}]
    assert "Write meeting notes in Japanese" in lm.instructions[0]
    assert lm.instructions[0].rstrip().endswith("keeping the five English section markers.")
    assert len(lm.docs[0]) == 5


def test_stage_trims_transcripts_to_fit(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    long = Config(reduce=ReduceConfig(max_input_tokens=3000))
    run_dir = _run(tmp_path, long, slides=[])
    segs = [s.model_copy(update={"transcript_text": "word " * 400}) for s in _segments()]
    schemas.save(run_dir, SegmentsFile(segments=segs, unaligned_slide_ids=[]))
    lm = FakeR7B(REPLY, tokens_per_char=1.0)
    reduce.run_stage(ctx_for(run_dir, "reduce", fake_manager(long, tmp_path, reduce=lm)))
    assert len(lm.docs) > 1
    assert all(len(d["snippet"]) < 2000 for d in lm.docs[-1])
    assert "transcripts trimmed" in caplog.text


def test_stage_fails_when_trimming_cannot_fit(tmp_path: Path) -> None:
    tiny = Config(reduce=ReduceConfig(max_input_tokens=10))
    run_dir = _run(tmp_path, tiny)
    lm = FakeR7B(REPLY, tokens_per_char=1.0)
    with pytest.raises(reduce.ReduceTooLong, match="two-pass"):
        reduce.run_stage(ctx_for(run_dir, "reduce", fake_manager(tiny, tmp_path, reduce=lm)))
    assert lm.calls == []


def test_stage_with_nothing_to_summarize_loads_no_model(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, Config(), slides=[])
    schemas.save(run_dir, SegmentsFile(segments=[], unaligned_slide_ids=[]))
    manager = fake_manager(Config(), tmp_path, reduce=FakeR7B(""))
    reduce.run_stage(ctx_for(run_dir, "reduce", manager))
    assert schemas.load(run_dir, NotesDoc).sections.summary == []
    assert manager.loaded() == []
