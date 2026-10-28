"""PLAN §5.6: documents, section markers, citation rules, time refinement, action items;
§4 two-pass reduce."""

import logging
from pathlib import Path

import pytest

from cohr import reduce, schemas
from cohr.config import Config, ReduceConfig
from cohr.schemas import (
    ActionItem,
    Citation,
    Claim,
    MapActionItem,
    MapFile,
    NotesDoc,
    NotesSections,
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
    return reduce.segment_refiner(_segments(), _items(), 0.1, "en", "en")


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


# Cross-language jump times (D39): clean-hi and xling-ja shaped, chunks of one segment.
EN_CHUNKS = [
    TranscriptItem(chunk_id=0, t0=0.0, t1=30.0, text="Welcome back. Omar will draft the plan."),
    TranscriptItem(
        chunk_id=1,
        t0=30.0,
        t1=60.0,
        text="When we talked to customers, retention came up. "
        "Maria, can you prepare the survey? The pipeline has 75 candidates.",
    ),
    TranscriptItem(
        chunk_id=2, t0=60.0, t1=90.0, text="Action item for Tom record the video by Monday."
    ),
]
JA_CHUNKS = [
    TranscriptItem(chunk_id=0, t0=0.0, t1=20.0, text="では、始めましょう。リテンションの話です。"),
    TranscriptItem(
        chunk_id=1,
        t0=20.0,
        t1=40.0,
        text="トライアル登録に移りましょう。今月の新規トライアル登録数は、4650です。",
    ),
    TranscriptItem(chunk_id=2, t0=40.0, t1=60.0, text="トムさんへのアクションアイテムです。"),
]


def _at(chunk: TranscriptItem, sentence: str) -> float:
    return round(chunk.t0 + (chunk.t1 - chunk.t0) * chunk.text.index(sentence) / len(chunk.text), 2)


def test_refine_time_anchors_a_hindi_claim_by_its_owner_and_number() -> None:
    def refine(span: str) -> float:
        return reduce.refine_time(span, EN_CHUNKS, 0.0, 0.1, "hi", "en")

    # no shared words: the owner, in the meeting's spelling for the whole segment (मारिया
    # alone in chunk 0's text would match Omar, the only m-r name there)
    survey = "[owner: मारिया] [due: अगले सप्ताह का अंत] ग्राहक सर्वेक्षण तैयार करें"
    assert refine(survey) == _at(EN_CHUNKS[1], "Maria, can")
    # Tom said as a word in chunk 2, not the "tom" of "customers" in chunk 1
    assert refine("[owner: टॉम] [due: सोमवार] ऑनबोर्डिंग वीडियो रिकॉर्ड करें") == 60.0
    assert refine("पाइपलाइन में कैंडिडेट्स की संख्या ७५ है") == _at(EN_CHUNKS[1], "The pipeline")
    # no anchor, or one the segment doesn't say: the segment's start (§5.6)
    assert refine("नया लोगो वेबसाइट पर उपयोग किया जाएगा।") == 0.0
    assert refine("[owner: ?] [due: ?] बजट 42,500 है") == 0.0


def test_refine_time_anchors_an_english_claim_on_a_japanese_meeting() -> None:
    def refine(span: str) -> float:
        return reduce.refine_time(span, JA_CHUNKS, 0.0, 0.1, "en", "ja")

    claim = "Trial registration numbers are higher this month (4,650) than last month (2,600)."
    assert refine(claim) == _at(JA_CHUNKS[1], "今月の")
    assert refine("[owner: Tom] [due: Monday] Create the onboarding video by Monday.") == 40.0


def test_refine_time_anchors_only_when_words_dont_match() -> None:
    # English on English: today's time exactly, though 75 and Omar point elsewhere
    span = "Omar will draft the plan; the pipeline has 75 candidates"
    want = reduce.refine_time(span, EN_CHUNKS, 0.0, 0.1)
    assert want == _at(EN_CHUNKS[0], "Omar")
    assert reduce.refine_time(span, EN_CHUNKS, 0.0, 0.1, "hi", "en") == want
    # a number said in two chunks: the chunk the words match wins
    chunks = [
        TranscriptItem(chunk_id=0, t0=0.0, t1=30.0, text="Last year we had 75 people."),
        TranscriptItem(chunk_id=1, t0=30.0, t1=60.0, text="The pipeline has 75 candidates."),
    ]
    assert reduce.refine_time("Pipeline: 75 candidates", chunks, 0.0, 0.1, "en", "en") == 30.0
    # word overlap under min_score still breaks a tie between anchored chunks
    assert reduce.refine_time("pipeline 75 क ख ग घ ङ च छ ज झ", chunks, 9.0, 0.5, "hi", "en") == 30.0


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
    """Replies in order (the last one repeats); tokens = characters × tokens_per_char;
    `finish` is mlx-lm's finish reason for every reply."""

    def __init__(self, reply: str | list[str], tokens_per_char: float = 0.0, finish: str = "stop"):
        self.replies = [reply] if isinstance(reply, str) else list(reply)
        self.finish = finish
        self.tokens_per_char = tokens_per_char
        self.instructions: list[str] = []
        self.docs: list[list[dict[str, str]]] = []
        self.calls: list[dict[str, float]] = []
        self.prompts: list[str] = []

    def prompt(self, instruction: str, docs: list[dict[str, str]]) -> str:
        self.instructions.append(instruction)
        self.docs.append(docs)
        return instruction + "".join(d["snippet"] for d in docs)

    def count_tokens(self, text: str) -> int:
        return int(len(text) * self.tokens_per_char)

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
        self.calls.append({"max_tokens": max_tokens, "temperature": temperature, "seed": seed})
        self.prompts.append(prompt)
        return self.replies[min(len(self.calls), len(self.replies)) - 1]

    def complete(
        self, prompt: str, *, max_tokens: int, temperature: float, seed: int
    ) -> reduce.Reply:
        text = self.generate(prompt, max_tokens=max_tokens, temperature=temperature, seed=seed)
        return reduce.Reply(text, self.finish)


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
    assert lm.calls == [{"max_tokens": 8192, "temperature": 0.0, "seed": 3}]
    assert "Write meeting notes in Japanese" in lm.instructions[0]
    assert lm.instructions[0].rstrip().endswith("keeping the five English section markers.")
    assert len(lm.docs[0]) == 5


def test_stage_starts_the_reply_with_the_first_marker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 6d-1: R7B translated the markers in de/ar notes; the reply is started with "## SUMMARY"
    # and the model writes on from there, so its reply has no first marker of its own.
    run_dir = _run(tmp_path, Config(target_language="de"))
    lm = FakeR7B(REPLY.removeprefix(reduce.REPLY_START))
    reduce.run_stage(ctx_for(run_dir, "reduce", fake_manager(Config(), tmp_path, reduce=lm)))
    assert lm.prompts[0].endswith(reduce.REPLY_START)
    doc = schemas.load(run_dir, NotesDoc)
    assert doc.raw.startswith(reduce.REPLY_START)
    assert doc.sections.summary[0].text == "The team approved the 14,500 budget."
    assert len(doc.sections.decisions) == 2
    assert "missing section markers" not in caplog.text


def test_stage_trims_a_group_over_the_cap(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    long = Config(reduce=ReduceConfig(max_input_tokens=6000))
    run_dir = _run(tmp_path, long, slides=[])
    segs = [s.model_copy(update={"transcript_text": "word " * 1000}) for s in _segments()]
    schemas.save(run_dir, SegmentsFile(segments=segs, unaligned_slide_ids=[]))
    lm = FakeR7B(REPLY, tokens_per_char=1.0)
    reduce.run_stage(ctx_for(run_dir, "reduce", fake_manager(long, tmp_path, reduce=lm)))
    assert len(lm.calls) == 3  # one segment per group, each trimmed, then pass 2
    assert "transcripts trimmed" in caplog.text
    assert schemas.load(run_dir, NotesDoc).passes == 2


def test_stage_fails_when_trimming_cannot_fit(tmp_path: Path) -> None:
    tiny = Config(reduce=ReduceConfig(max_input_tokens=10))
    run_dir = _run(tmp_path, tiny)
    lm = FakeR7B(REPLY, tokens_per_char=1.0)
    with pytest.raises(reduce.ReduceTooLong, match="over the 10 cap"):
        reduce.run_stage(ctx_for(run_dir, "reduce", fake_manager(tiny, tmp_path, reduce=lm)))
    assert lm.calls == []


def test_stage_with_nothing_to_summarize_loads_no_model(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, Config(), slides=[])
    schemas.save(run_dir, SegmentsFile(segments=[], unaligned_slide_ids=[]))
    manager = fake_manager(Config(), tmp_path, reduce=FakeR7B(""))
    reduce.run_stage(ctx_for(run_dir, "reduce", manager))
    assert schemas.load(run_dir, NotesDoc).sections.summary == []
    assert manager.loaded() == []


# --- two-pass (§4) -------------------------------------------------------------------------


class PartsR7B(FakeR7B):
    """100 tokens per segment document, so the cap decides how many segments share a group."""

    def count_tokens(self, text: str) -> int:
        return 100 * text.count("Transcript:")


PASS1 = [
    # group 0: seg 0 and slide 0 → documents [seg 0, slide 0]
    "## SUMMARY\n<co>The team approved the 14,500 budget.</co: 0:[0,1]>\n"
    "## KEY_POINTS\n- <co>Welcome to the Atlas sync</co: 0:[0]>\n- An unsupported remark\n"
    "## DECISIONS\n- none\n## ACTION_ITEMS\n- none\n## OPEN_QUESTIONS\n- none",
    # group 1: seg 1 → documents [seg 1]
    "## SUMMARY\n<co>Beta launch is on 3 November.</co: 0:[0]>\n## KEY_POINTS\n- none\n"
    "## DECISIONS\n- <co>Launch the beta on 3 November</co: 0:[0]>\n"
    "## ACTION_ITEMS\n- [owner: Tom] [due: ?] <co>Write release notes</co: 0:[0]>\n"
    "## OPEN_QUESTIONS\n- none",
]
# Pass-2 documents: [group 0, group 1, slide 1 (aligned to no segment)].
PASS2 = (
    "## SUMMARY\n<co>The team approved the 14,500 budget [seg-0] [slide-1]</co: 0:[0]>. "
    "<co>The beta launches on 3 November</co: 0:[1]>. Nothing cited here.\n"
    "## KEY_POINTS\n- <co>An unsupported remark</co: 0:[0]>\n- <co>The plan slide</co: 0:[2]>\n"
    "- <co>Something vague entirely</co: 0:[1]>\n- Bad <co>index</co: 0:[9]>\n"
    "- <co>Ghost part [seg-7]</co: 0:[1]>\n"
    "## DECISIONS\n- <co>Launch the beta on 3 November [seg-1]</co: 0:[1]>\n"
    "## ACTION_ITEMS\n- [owner: Tom] [due: ?] <co>Write release notes</co: 0:[1]>\n"
    "## OPEN_QUESTIONS\n- none"
)


def _two_pass(lm: FakeR7B, max_tokens: int = 150) -> tuple[NotesDoc, reduce.ReduceStats]:
    progress: list[str] = []
    return reduce.two_pass(
        lm,
        ("pass 1", "pass 2"),
        _segments(),
        _map(),
        [_slide(0, "Q4 Budget")],
        [_slide(1, "Plan")],
        _refine(),
        max_tokens,
        lambda prompt: lm.generate(prompt, max_tokens=1, temperature=0.0, seed=0),
        lambda _f, msg: progress.append(msg),
        "en",
    )


def _cites(claim: Claim) -> list[tuple[str, int]]:
    return [(c.kind, c.id) for c in claim.citations]


def test_groups_fill_the_budget_with_consecutive_segments() -> None:
    segs = [*_segments(), _segments()[1].model_copy(update={"seg_id": 2})]
    slides = [_slide(0, "Q4 Budget")]
    lm = PartsR7B("")
    groups = reduce.make_groups(lm, "x", segs, _map(), slides, budget=200)
    assert [[s.seg_id for s in g.segments] for g in groups] == [[0, 1], [2]]
    assert [[s.slide_id for s in g.slides] for g in groups] == [[0], []]
    # a segment over the budget on its own is still a group of one
    alone = reduce.make_groups(lm, "x", segs, _map(), slides, budget=50)
    assert [[s.seg_id for s in g.segments] for g in alone] == [[0], [1], [2]]


def test_group_snippet_writes_citation_markers() -> None:
    doc, _ = reduce.build_notes(PASS1[1], [("seg", 1)], _refine(), _map(), "en")
    snippet = reduce.group_snippet(doc)
    assert snippet.splitlines() == [
        "Summary: Beta launch is on 3 November. [seg-1]",
        "Decisions:",
        "- Launch the beta on 3 November [seg-1]",
        "Action items:",
        "- [owner: Tom] [due: ?] Write release notes [seg-1]",
    ]
    slide_doc, _ = reduce.build_notes(PASS1[0], [("seg", 0), ("slide", 0)], _refine(), {}, "en")
    lines = reduce.group_snippet(slide_doc).splitlines()
    assert lines[0] == "Summary: The team approved the 14,500 budget. [seg-0] [slide-1]"
    assert "- An unsupported remark" in lines  # uncited in pass 1: no marker


def test_span_targets_markers_then_claim_then_group() -> None:
    known: set[reduce.Target] = {("seg", 0), ("seg", 1), ("slide", 0)}
    group = reduce.PassOne(
        [0, 1],
        [
            Claim(text="Budget approved at 14,500", citations=[Citation(kind="seg", id=1)]),
            Claim(text="Someone mentioned the weather", citations=[]),
        ],
    )
    assert reduce.span_targets("Budget [seg-0] [slide-1] [seg-9]", group, known) == (
        [("seg", 0), ("slide", 0)],
        "marker",
        1,
    )
    assert reduce.span_targets("approved 14,500", group, known) == ([("seg", 1)], "claim", 0)
    assert reduce.span_targets("the weather", group, known) == ([], "uncited", 0)
    assert reduce.span_targets("unrelated words", group, known) == (
        [("seg", 0), ("seg", 1)],
        "group",
        0,
    )
    # only unknown markers: counted as dropped, then resolved like a span without markers
    assert reduce.span_targets("unrelated [seg-9]", group, known) == (
        [("seg", 0), ("seg", 1)],
        "group",
        1,
    )


def test_two_pass_citations_survive_both_passes() -> None:
    lm = PartsR7B([*PASS1, PASS2])
    doc, stats = _two_pass(lm)
    assert len(lm.calls) == 3
    assert [d["title"] for d in lm.docs[-1]] == [
        "0:00–0:40 · slides 1",
        "5:00–5:40",
        "slide 2: Plan",
    ]
    assert "[seg-0] [slide-1]" in lm.docs[-1][0]["snippet"]
    assert doc.passes == 2 and doc.raw == PASS2
    assert doc.two_pass is not None
    assert [(g.seg_ids, g.slide_ids, g.raw) for g in doc.two_pass.groups] == [
        ([0], [0], PASS1[0]),
        ([1], [], PASS1[1]),
    ]
    assert doc.two_pass.resolved == {"marker": 2, "claim": 2, "uncited": 1, "group": 2}
    s = doc.sections
    # markers in the span; never shown in the text
    assert s.summary[0].text == "The team approved the 14,500 budget."
    assert _cites(s.summary[0]) == [("seg", 0), ("slide", 0)]
    # refined against the resolved segment, with the pass-2 span
    assert s.summary[0].citations[0].t == 20.0
    # no marker: the best-matching pass-1 claim
    assert _cites(s.summary[1]) == [("seg", 1)] and s.summary[1].citations[0].t == 300.0
    assert _cites(s.summary[2]) == []  # uncited in pass 2 stays uncited
    assert _cites(s.key_points[0]) == []  # matched an uncited pass-1 claim: nothing invented
    assert _cites(s.key_points[1]) == [("slide", 1)]  # a slide document cites itself
    assert _cites(s.key_points[2]) == [("seg", 1)]  # no match: every segment of the group
    assert _cites(s.key_points[3]) == []  # out-of-range document index
    assert _cites(s.key_points[4]) == [("seg", 1)]  # unknown marker dropped, group fallback
    assert _cites(s.decisions[0]) == [("seg", 1)]
    assert (s.action_items[0].owner, s.action_items[0].due) == ("Tom", None)
    assert _cites(s.action_items[0]) == [("seg", 1)]
    assert (stats.groups, stats.dropped, stats.pass1_uncited) == (2, 2, 1)
    assert stats.uncited == 3


class BigPartsR7B(PartsR7B):
    def count_tokens(self, text: str) -> int:
        return super().count_tokens(text) + 1000 * text.count("pass 2")


def test_two_pass_fails_when_the_part_summaries_are_over_the_cap() -> None:
    with pytest.raises(reduce.ReduceTooLong, match="2 part summaries are 1000 tokens"):
        _two_pass(BigPartsR7B([*PASS1, PASS2]))


def test_stage_runs_two_passes_over_the_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="cohr")
    capped = Config(reduce=ReduceConfig(max_input_tokens=150))
    run_dir = _run(tmp_path, capped)
    lm = PartsR7B([*PASS1, PASS2])
    reduce.run_stage(ctx_for(run_dir, "reduce", fake_manager(capped, tmp_path, reduce=lm)))
    doc = schemas.load(run_dir, NotesDoc)
    assert doc.passes == 2 and doc.two_pass is not None and len(doc.two_pass.groups) == 2
    assert lm.instructions[-1].startswith("Write meeting notes in English for a long meeting")
    assert lm.instructions[-2].startswith("Write notes in English for one part of a long meeting")
    assert [c["max_tokens"] for c in lm.calls] == [ReduceConfig().max_tokens] * 3
    assert [d["title"] for d in lm.docs[-1]][2:] == ["slide 2: Plan", "slide 3: Retention"]
    assert "two passes, 2 groups" in caplog.text


def _seg_notes(
    decisions: list[str], tasks: list[str], owner: str | None = "Tom"
) -> dict[int, SegmentNotes]:
    actions = [MapActionItem(owner=owner, task=t, due="Friday") for t in tasks]
    return {
        0: SegmentNotes(
            seg_id=0,
            summary="",
            key_points=[],
            decisions=decisions,
            action_items=actions,
            open_questions=[],
            terms=[],
        )
    }


SEG0 = [Citation(kind="seg", id=0)]


@pytest.mark.parametrize(
    ("text", "citations", "decisions", "tasks", "moved"),
    [
        # AMI ES2004b: Map's task, filed as a decision by Reduce
        ("The project will prepare a questionnaire", SEG0, [], ["Prepare a questionnaire"], True),
        # xling-es: Map lists the task under both: a task
        (
            "Laura will send the contract",
            SEG0,
            ["Laura will send the contract"],
            ["Send the contract"],
            True,
        ),
        # must not fire: a Map decision matches it better than the task
        (
            "Prepare the questionnaire for users",
            SEG0,
            ["Prepare the questionnaire for users"],
            ["Prepare a questionnaire about remote controls"],
            False,
        ),
        # must not fire: shares a word with a task, under ACTION_MATCH_MIN
        ("Launch the beta on 3 November", SEG0, [], ["Write the beta release notes"], False),
        ("Prepare a questionnaire", [], [], ["Prepare a questionnaire"], False),  # uncited
        ("Prepare a questionnaire", [Citation(kind="slide", id=0)], [], ["Prepare a q"], False),
    ],
)
def test_filed_task(
    text: str, citations: list[Citation], decisions: list[str], tasks: list[str], moved: bool
) -> None:
    claim = Claim(text=text, citations=citations)
    assert reduce.filed_task(claim, _seg_notes(decisions, tasks)) is moved


def test_a_tie_without_an_owner_stays_a_decision() -> None:
    """AMI TS3003c: Map wrote a decided design as a task and a decision, with no owner."""
    claim = Claim(text="Design the single curved side remote control", citations=SEG0)
    design = ["Design the single curved side remote control"]
    assert not reduce.filed_task(claim, _seg_notes(design, design, owner=None))
    assert reduce.filed_task(claim, _seg_notes(design, design))  # an owner: a task


def test_a_filed_task_moves_to_action_items_with_its_citations() -> None:
    reply = (
        "## SUMMARY\nShort.\n## KEY_POINTS\n- none\n## DECISIONS\n"
        "- <co>Keep the price at 25 euros</co: 0:[0]>\n"
        "- <co>The project will prepare a questionnaire</co: 0:[0]>\n"
        "## ACTION_ITEMS\n- [owner: Ana] [due: ?] <co>Book the room</co: 0:[0]>\n"
        "## OPEN_QUESTIONS\n- none\n<|END_RESPONSE|>"
    )
    doc, stats = reduce.build_notes(
        reply,
        [("seg", 0)],
        lambda seg, span: 1.0,
        _seg_notes([], ["Prepare a questionnaire"]),
        "en",
    )
    assert [c.text for c in doc.sections.decisions] == ["Keep the price at 25 euros"]
    moved = doc.sections.action_items[-1]
    assert (moved.text, moved.owner, moved.due) == (
        "The project will prepare a questionnaire",
        "Tom",
        "Friday",
    )
    assert [(c.kind, c.id) for c in moved.citations] == [("seg", 0)]
    assert doc.sections.action_items[0].owner == "Ana"  # the model's own items come first
    assert (stats.moved, stats.claims) == (1, 4)


def test_a_moved_task_already_listed_adds_its_citations() -> None:
    listed = ActionItem(text="Send the contract to legal", citations=SEG0, owner="Laura")
    items = [listed]
    reduce.add_task(
        items,
        ActionItem(
            text="Laura will send the contract to legal",
            citations=[Citation(kind="seg", id=0), Citation(kind="seg", id=2)],
        ),
    )
    assert len(items) == 1 and items[0].owner == "Laura"
    assert [(c.kind, c.id) for c in items[0].citations] == [("seg", 0), ("seg", 2)]
    reduce.add_task(items, ActionItem(text="Book the room", citations=SEG0))
    assert [i.text for i in items] == ["Send the contract to legal", "Book the room"]


@pytest.mark.parametrize(
    ("text", "finish", "cut"),
    [
        ("## KEY_POINTS\n- <co>Retention rose to 41%</co: 0:[", "length", True),  # AMI IS1009c
        ("## KEY_POINTS\n- none\n##", "length", True),  # AMI IS1009b: a bare marker
        # closed with its last token at the cap: complete
        ("## OPEN_QUESTIONS\n- none\n<|END_RESPONSE|>", "length", False),
        ("## OPEN_QUESTIONS\n- none\n<|END_RESPONSE|>", "stop", False),
        ("## OPEN_QUESTIONS\n- none\n", "stop", False),  # end of turn: not the cap
        ("## OPEN_QUESTIONS\n- none\n", None, False),
    ],
)
def test_cut_off(text: str, finish: str | None, cut: bool) -> None:
    assert reduce.cut_off(reduce.Reply(text, finish)) is cut


# Cut by the cap mid-section (6e-1, AMI TS3003c): no DECISIONS, ACTION_ITEMS, OPEN_QUESTIONS.
CUT_REPLY = REPLY.split("## DECISIONS")[0] + "- <co>Retention rose"


def test_stage_flags_a_cut_reply(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    run_dir = _run(tmp_path, Config())
    lm = FakeR7B(CUT_REPLY.replace("0:[3]", "0:[4]"), finish="length")
    reduce.run_stage(ctx_for(run_dir, "reduce", fake_manager(Config(), tmp_path, reduce=lm)))
    doc = schemas.load(run_dir, NotesDoc)
    assert doc.truncated == 1
    assert doc.sections.decisions == [] and doc.sections.action_items == []
    assert "1 reply cut at reduce.max_tokens = 8192" in caplog.text
    # The cut claim is kept as written, uncited: nothing is completed or invented (§5.6).
    assert [c.text for c in doc.sections.key_points][-1] == "Retention rose"


def test_stage_keeps_a_reply_closed_at_the_cap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Must not fire: <|END_RESPONSE|> was the last token the cap allowed."""
    run_dir = _run(tmp_path, Config())
    lm = FakeR7B(REPLY.replace("0:[3]", "0:[4]"), finish="length")
    reduce.run_stage(ctx_for(run_dir, "reduce", fake_manager(Config(), tmp_path, reduce=lm)))
    assert schemas.load(run_dir, NotesDoc).truncated == 0
    assert "cut at reduce.max_tokens" not in caplog.text


def test_two_pass_counts_cut_replies(tmp_path: Path) -> None:
    capped = Config(reduce=ReduceConfig(max_input_tokens=150))
    run_dir = _run(tmp_path, capped)
    lm = PartsR7B([*PASS1, PASS2], finish="length")  # none closes with <|END_RESPONSE|>
    reduce.run_stage(ctx_for(run_dir, "reduce", fake_manager(capped, tmp_path, reduce=lm)))
    assert schemas.load(run_dir, NotesDoc).truncated == 3


def test_stage_under_the_cap_is_one_pass(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, Config())
    lm = PartsR7B(REPLY.replace("0:[3]", "0:[4]"))
    reduce.run_stage(ctx_for(run_dir, "reduce", fake_manager(Config(), tmp_path, reduce=lm)))
    doc = schemas.load(run_dir, NotesDoc)
    assert (doc.passes, doc.two_pass, len(lm.calls)) == (1, None, 1)


def test_restore_owners_uses_the_meetings_spelling() -> None:
    doc = NotesDoc(
        sections=NotesSections(
            summary=[],
            key_points=[],
            decisions=[],
            action_items=[
                ActionItem(text="अनुबंध भेजें", citations=SEG0, owner="लॉरा", due="शुक्रवार"),
                ActionItem(text="वीडियो", citations=SEG0, owner="प्रिया"),  # not said: kept
                ActionItem(text="सर्वे", citations=SEG0),
            ],
            open_questions=[],
        ),
        passes=1,
        language="hi",
        raw="",
    )
    fixed, changed = reduce.restore_owners(doc, ["Laura, send the contract by Friday."])
    assert [a.owner for a in fixed.sections.action_items] == ["Laura", "प्रिया", None]
    assert fixed.sections.action_items[0].due == "शुक्रवार"  # only the owner changes
    assert changed == ["लॉरा → Laura"]


def test_refine_time_without_chunks_or_overlap_keeps_the_segment_start() -> None:
    assert reduce.refine_time("budget 75", [], 5.0, 0.1, "hi", "en") == 5.0
    assert reduce.refine_time("nothing shared", EN_CHUNKS, 5.0, 0.0) == 5.0


def _due_doc(items: list[ActionItem], language: str = "en") -> NotesDoc:
    return NotesDoc(
        sections=NotesSections(
            summary=[], key_points=[], decisions=[], action_items=items, open_questions=[]
        ),
        passes=1,
        language=language,
        raw="",
    )


DUE_SEGMENTS = [
    Segment(
        seg_id=0,
        t0=0.0,
        t1=60.0,
        chunk_ids=[0],
        transcript_text="Tom, finish the incident post mortem by the 17th of November. "
        "Laura configures the error alerts by Friday.",
        slide_ids=[],
    ),
    Segment(
        seg_id=1,
        t0=60.0,
        t1=120.0,
        chunk_ids=[1],
        transcript_text="The dashboard for revenue was updated on Monday. "
        "Grace, summarise the competitor research by 11 November.",
        slide_ids=[],
    ),
]
SEG1 = [Citation(kind="seg", id=1)]


def test_guard_dues_drops_an_unsaid_due_and_keeps_a_said_one() -> None:
    doc = _due_doc(
        [
            ActionItem(
                text="Finish the post mortem", citations=SEG0, owner="Tom", due="17 November"
            ),
            ActionItem(text="Configure error alerts", citations=SEG0, owner="Laura", due="Friday"),
            ActionItem(text="Update revenue dashboard", citations=SEG1, due="end of next week"),
            ActionItem(text="Summarise research", citations=SEG1, owner="Grace", due="1 November"),
        ]
    )
    fixed, dropped = reduce.guard_dues(doc, DUE_SEGMENTS, [], "en")
    assert [a.due for a in fixed.sections.action_items] == ["17 November", "Friday", None, None]
    assert [a.owner for a in fixed.sections.action_items] == ["Tom", "Laura", None, "Grace"]
    assert dropped == [
        "Update revenue dashboard: end of next week",
        "Summarise research: 1 November",
    ]


def test_guard_dues_must_not_fire() -> None:
    slide = _slide(0, "Next steps", "Send the contract — due 20 October")
    doc = _due_doc(
        [
            # Said in other words: "by the 17th of November" for "the seventeenth of Nov".
            ActionItem(text="Post mortem", citations=SEG0, due="the seventeenth of Nov"),
            # Said on the cited slide, not in a transcript.
            ActionItem(
                text="Send the contract", citations=[Citation(kind="slide", id=0)], due="20 Oct"
            ),
            # Uncited: nothing to check it against.
            ActionItem(text="Book the venue", citations=[], due="end of next week"),
        ]
    )
    fixed, dropped = reduce.guard_dues(doc, DUE_SEGMENTS, [slide], "en")
    assert fixed == doc and dropped == []


def test_guard_dues_reads_only_digits_across_languages() -> None:
    doc = _due_doc(
        [
            # A Hindi weekday on an English meeting: unreadable, kept.
            ActionItem(text="अलर्ट कॉन्फ़िगर करें", citations=SEG0, due="शुक्रवार"),
            ActionItem(text="पोस्ट मॉर्टम", citations=SEG0, due="17 नवंबर"),
            ActionItem(text="शोध का सारांश", citations=SEG1, due="1 नवंबर"),
        ],
        language="hi",
    )
    fixed, dropped = reduce.guard_dues(doc, DUE_SEGMENTS, [], "en")
    assert [a.due for a in fixed.sections.action_items] == ["शुक्रवार", "17 नवंबर", None]
    assert dropped == ["शोध का सारांश: 1 नवंबर"]


def test_move_vague_tasks_to_key_points() -> None:
    vague = ActionItem(text="Continue monitoring retention metrics.", citations=SEG1)
    kept = [
        # An owner-less task that's real (long-en: planted open question, due "next time").
        ActionItem(text="Run the user conference online.", citations=SEG0, due="next time"),
        ActionItem(text="Prepare a questionnaire for the next meeting.", citations=SEG0),
        ActionItem(text="Ensure the alerts fire.", citations=SEG0, owner="Laura"),
    ]
    fixed, moved = reduce.move_vague_tasks(_due_doc([vague, *kept]))
    assert fixed.sections.action_items == kept
    assert fixed.sections.key_points == [Claim(text=vague.text, citations=SEG1)]
    assert moved == [vague.text]
    other = _due_doc([vague], language="de")
    assert reduce.move_vague_tasks(other) == (other, [])


def test_placeholder_dues_are_unknown() -> None:
    claim = Claim(
        text="[owner: Marketing] [due: Not specified] Look into requirements", citations=SEG0
    )
    item = reduce.action_item(claim, {})
    assert (item.owner, item.due) == ("Marketing", None)
