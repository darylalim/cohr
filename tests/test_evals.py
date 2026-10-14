"""Eval scoring and generator (PLAN §18 Tests). No models, no `say`, no ffmpeg."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from cohr import evals, schemas
from cohr.config import Config
from cohr.evals import (
    NUMBER_LABELS,
    ItemReport,
    ItemSpec,
    Report,
    Truth,
    TruthFact,
    TruthSlide,
    action_fit,
    citation_check,
    compare,
    coverage,
    same_due,
    wer_text,
)
from cohr.numbers import digits, numbers_in
from cohr.schemas import (
    ActionItem,
    Chunk,
    ChunksFile,
    Citation,
    Claim,
    Manifest,
    MapFile,
    NotesDoc,
    NotesGroup,
    NotesSections,
    NumberRepair,
    SegmentNotes,
    Slide,
    SlidesFile,
    Span,
    StageState,
    Transcript,
    TranscriptItem,
    TranslatedNotes,
    TwoPass,
)
from evals.gen import build as gen_build
from evals.gen.script import (
    DECISIONS,
    EN,
    LANGS,
    MORE_DECISIONS,
    MORE_DISCUSSION,
    MORE_OPEN_QUESTIONS,
    MORE_TASKS,
    OPEN_QUESTIONS,
    TASKS,
    TEEN_TO_TY,
    Script,
    Variant,
    generate,
)
from tests.conftest import make_run

ROOT = Path(__file__).resolve().parents[1]


# --- normalization -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("$14,500", {14500}),
        ("fourteen thousand five hundred dollars", {14500}),
        ("14.5k", {14500}),
        ("$70 thousand", {70000}),
        ("38%", {38}),
        ("thirty eight percent", {38}),
        ("the twentieth of October", {20}),
        ("on the 3rd", {3}),
        ("twenty first", {21}),
        ("one hundred and five tickets", {105}),
        ("Revenue rose to 1.2 million", {1200000}),
        ("no numbers here", set()),
    ],
)
def test_numbers_normalized_to_digits(text: str, values: set[int]) -> None:
    assert numbers_in(text) == {float(v) for v in values}


def test_thousand_alone_is_not_a_number() -> None:
    assert numbers_in("a thousand thanks") == set()
    assert digits("thousands of users") == "thousands of users"


def test_wer_text_makes_spoken_and_written_forms_equal() -> None:
    spoken = "The budget is fourteen thousand five hundred dollars, up thirty eight percent."
    written = "the budget is $14,500, up 38%."
    assert wer_text(spoken) == wer_text(written)


def test_asr_error_in_spanish_and_japanese() -> None:
    # 6d-2: Transcribe writes the script's plain "42500" as "42.500" or "42,500" and "34 por
    # ciento" as "34%" (say → Transcribe round trip); neither is an error.
    script = "El presupuesto es de 42500 dólares, frente al 34 por ciento."
    heard = "El presupuesto es de 42.500 dólares, frente al 34%."
    assert evals.asr_error(script, heard, "es") == ("wer", 0.0)
    assert evals.asr_error(script, heard.replace("42.500", "42,500"), "es") == ("wer", 0.0)
    kind, err = evals.asr_error(script, heard.replace("presupuesto", "precio"), "es")
    assert kind == "wer" and err == pytest.approx(1 / 11)  # "por ciento" is two words
    ja = "承認された予算は42500ドルです。再訪率は34パーセントです。"
    assert evals.asr_error(ja, "承認された予算は42,500ドルです 再訪率は34%です", "ja") == (
        "cer",
        0.0,
    )
    kind, err = evals.asr_error(ja, ja.replace("承認", "使用"), "ja")
    assert kind == "cer" and 0 < err < 0.1
    # English is the old WER, unchanged.
    assert evals.asr_error("It is $42,500.", "it is 42500 dollars", "en") == ("wer", 0.0)


@pytest.mark.parametrize(
    ("stated", "planted", "same"),
    [
        ("the twentieth of October", "October 20", True),
        ("Oct 20", "October 20", True),
        ("20 October", "October 20", True),
        ("by the end of next week", "end of next week", True),
        ("Friday", "Friday", True),
        ("by Friday", "end of next week", False),
        ("end of the month", "end of next week", False),
        ("October 21", "October 20", False),
    ],
)
def test_due_dates(stated: str, planted: str, same: bool) -> None:
    assert same_due(stated, planted) is same


def test_coverage_uses_content_tokens_and_stems() -> None:
    fact = "move the weekly sync to Thursdays"
    assert coverage(fact, "The weekly syncs moved to Thursday.") == 1.0
    assert coverage(fact, "Weekly meeting stays on Tuesdays.") < evals.TEXT_MATCH_MIN


# --- fact matching -------------------------------------------------------------------------


def fact(kind: str = "action", **kw: object) -> TruthFact:
    base: dict[str, object] = {
        "id": "f1",
        "kind": kind,
        "text": "update the pricing page",
        "t0": 100.0,
        "t1": 104.0,
    }
    if kind == "action":
        base |= {"owner": "Priya", "due": "October 20"}
    return TruthFact.model_validate(base | kw)


def item(
    text: str, owner: str | None = None, due: str | None = None, citations: list | None = None
) -> ActionItem:
    return ActionItem(text=text, citations=citations or [], owner=owner, due=due)


def test_action_owner_and_due() -> None:
    f = fact()
    assert action_fit(item("Update the pricing page", "Priya", "20 October"), f) == "match"
    assert action_fit(item("Update the pricing page", "Priya", None), f) == "match"
    assert action_fit(item("Priya: update the pricing page"), f) == "match"  # owner in text
    assert action_fit(item("Update the pricing page"), f) == "partial"  # no owner stated
    plain = Claim(text="Update the pricing page by Oct 20.", citations=[])
    assert action_fit(plain, f) == "partial"
    assert action_fit(item("Update the pricing page", "Tom", None), f) == "wrong"
    # The Phase 4a Map error: someone else's due date on the right task.
    assert action_fit(item("Update the pricing page", "Priya", "end of next week"), f) == "wrong"
    assert action_fit(item("Book the venue", "Priya", "October 20"), f) == "none"


def test_cross_language_matches_exact_fields_only() -> None:
    f = fact()
    de = item("Preisseite aktualisieren", "Priya", "20. Oktober")
    assert action_fit(de, f, cross_language=True) == "match"
    assert action_fit(item("Preisseite", "Priya", "21. Oktober"), f, True) == "wrong"
    assert action_fit(item("Preisseite", "Tom", "20. Oktober"), f, True) == "none"
    number = fact("number", text="Open tickets: 409", value=409.0)
    assert evals.fact_fit(Claim(text="409 offene Tickets", citations=[]), number, True) == "match"
    decision = fact("decision", text="drop the legacy export feature")
    claim = Claim(text="Den alten Export einstellen", citations=[])
    assert evals.fact_fit(claim, decision, True) == "none"  # judge only (4b-2)


def test_owner_match_ignores_accents_and_case() -> None:
    # 6d-2: Spanish Transcribe wrote "María" for Maria; Japanese "OMAR" for Omar.
    f = fact(owner="Maria")
    assert action_fit(item("Preparar la encuesta", "María", None), f, True) == "match"
    assert action_fit(item("Prepare the survey (owner: MARIA)"), f, True) == "match"
    assert action_fit(item("Prepare the survey", "Mario", None), f, True) == "none"
    assert evals.fold(" Ómar ") == "omar"


def test_cross_language_numbers_are_read_in_the_notes_language() -> None:
    # 6d-1: a German "3.550" is the planted 3,550, not 3.55 (scored as missed before).
    number = fact("number", text="New trial signups: 3,550", value=3550.0)
    claim = Claim(text="Neue Trial-Signups: 3.550 (dieses Monat)", citations=[])
    assert evals.fact_fit(claim, number, True) == "none"  # read as English
    assert evals.fact_fit(claim, number, True, "de") == "match"
    f = fact()  # due 20 October
    assert action_fit(item("Preisseite", "Priya", "20. Oktober"), f, True, "de") == "match"
    assert action_fit(item("Preisseite", "Priya", "21. Oktober"), f, True, "de") == "wrong"


def test_unsupported_numbers_in_another_language() -> None:
    transcript = Transcript(
        language="en",
        runtime="mlx-speech",
        items=[
            TranscriptItem(chunk_id=0, t0=0, t1=5, text="The budget is $42,500, signups 3,550.")
        ],
    )
    de = [Claim(text="Budget 42.500 $, Signups 3.550, Kosten 9.900 $", citations=[])]
    assert evals.unsupported_numbers(de, transcript, [], "de") == ["9900"]
    assert evals.unsupported_numbers(de, transcript, []) == ["3.55", "9.9", "42.5"]  # English read
    ar = [Claim(text="الميزانية 42,500 دولار", citations=[])]  # also 42.5: one reading is enough
    assert evals.unsupported_numbers(ar, transcript, [], "ar") == []


def test_citation_tolerance_and_slides() -> None:
    f = fact("number", text="Open tickets: 409", value=409.0, slide=2)
    near = Claim(text="x", citations=[Citation(kind="seg", id=0, t=115.0)])
    far = Claim(text="x", citations=[Citation(kind="seg", id=0, t=115.5)])
    slide = Claim(text="x", citations=[Citation(kind="slide", id=7)])
    assert citation_check(near, f, {}) == (True, 15.0, False)
    assert citation_check(far, f, {})[0] is False
    assert citation_check(slide, f, {2: 7}) == (True, None, True)
    assert citation_check(slide, f, {2: 6})[0] is False
    both = Claim(text="x", citations=[*far.citations, Citation(kind="seg", id=1, t=98.0)])
    assert citation_check(both, f, {})[1] == -2.0  # nearest error, signed


def notes(**sections: list[Claim]) -> NotesDoc:
    empty: dict[str, list] = {s: [] for s in evals.SECTIONS}
    return NotesDoc(
        sections=NotesSections.model_validate(empty | sections),
        passes=1,
        language="en",
        raw="",
    )


def cite(t: float) -> list[Citation]:
    return [Citation(kind="seg", id=0, t=t)]


def test_score_facts_and_made_up_items() -> None:
    facts = [
        fact("decision", id="d1", text="drop the legacy export feature", t0=50.0),
        fact(id="a1", t0=100.0),
        fact("number", id="n1", text="Open tickets: 409", value=409.0, t0=10.0, slide=0),
        fact("decision", id="d2", text="hire a contract designer", t0=200.0),
    ]
    doc = notes(
        key_points=[Claim(text="There are 409 open tickets.", citations=cite(40.0))],
        decisions=[
            Claim(text="The legacy export feature will be dropped.", citations=cite(45.0)),
            Claim(text="Move the office to Berlin.", citations=cite(60.0)),
        ],
        action_items=[
            item("Update the pricing page", "Priya", "October 20", citations=cite(101.0)),
            item("Write the release notes", "Tom", "Friday", citations=cite(120.0)),
            item("Update the pricing page", "Priya", "Friday", citations=cite(101.0)),
            item("Update the pricing page by October 20", citations=cite(101.0)),
        ],
    )
    results = {r.id: r for r in evals.score_facts(facts, doc, {0: 3})}
    assert results["d1"].found and results["d1"].citation_ok and results["d1"].t_error == -5.0
    assert results["a1"].found and results["a1"].section == "action_items"
    assert results["n1"].found and results["n1"].citation_ok is False  # 30 s off, no slide
    assert not results["d2"].found
    made, wrong = evals.made_up(facts, doc)
    assert made == [
        "Move the office to Berlin.",
        "Write the release notes",
        "Update the pricing page",
    ]
    assert wrong == ["Update the pricing page"]


def test_open_questions_have_their_own_recall_and_made_up_count() -> None:
    facts = [
        fact("open_question", id="q1", text="outsource the translation work", t0=80.0),
        fact("decision", id="d1", text="drop the legacy export feature", t0=50.0),
    ]
    doc = notes(
        decisions=[Claim(text="Drop the legacy export feature.", citations=cite(50.0))],
        open_questions=[
            Claim(text="Should the translation work be outsourced?", citations=cite(82.0)),
            Claim(text="Is there seasonality in the retention figures?", citations=cite(90.0)),
        ],
    )
    results = {r.id: r for r in evals.score_facts(facts, doc, {})}
    assert results["q1"].found and results["q1"].section == "open_questions"
    assert evals.made_up_questions(facts, doc) == ["Is there seasonality in the retention figures?"]
    assert evals.made_up_questions(facts, doc, cross_language=True) == []
    # A planted question only counts in the Open questions section.
    elsewhere = notes(key_points=[Claim(text="Outsource the translation work.", citations=[])])
    assert not evals.score_facts(facts, elsewhere, {})[0].found


# --- stage metrics -------------------------------------------------------------------------


def test_speech_f1() -> None:
    truth = [Span(t0=1.0, t1=3.0)]
    assert evals.speech_f1(truth, [Span(t0=1.0, t1=3.0)], 5.0) == 1.0
    assert evals.speech_f1(truth, [Span(t0=2.0, t1=4.0)], 5.0) == pytest.approx(0.5)
    assert evals.speech_f1(truth, [], 5.0) == 0.0


def test_gap_text_counts_chunks_mostly_inside_gaps() -> None:
    t = Transcript(
        language="en",
        runtime="mlx-speech",
        items=[
            TranscriptItem(chunk_id=0, t0=0.0, t1=10.0, text="real speech here"),
            TranscriptItem(chunk_id=1, t0=19.0, t1=25.0, text="la la la"),
            TranscriptItem(chunk_id=2, t0=29.0, t1=35.0, text="more speech"),
        ],
    )
    words, texts = evals.gap_text(t, [Span(t0=18.0, t1=30.0)])
    assert (words, texts) == (3, ["la la la"])


def slide(slide_id: int, title: str, tables: list[str], t: float | None = None) -> Slide:
    return Slide(
        slide_id=slide_id,
        frame_id=slide_id,
        t=t,
        t_source="aligned" if t is not None else "none",
        title=title,
        text_md="",
        tables_md=tables,
        chart_notes="",
        visual_notes="",
    )


def test_slide_matching_and_metrics() -> None:
    truth = [
        TruthSlide(slide_id=0, title="Retention", cells=["Metric", "Value", "83%"], t0=0, t1=50),
        TruthSlide(slide_id=1, title="Site Speed", cells=["Page load (ms)", "640"], t0=50, t1=99),
    ]
    out = [
        slide(0, "**Site Speed**", ["| Page load (ms) | 640 |"], t=55.0),
        slide(1, "Retention", ["| Metric | Value |\n|---|---|\n| Week 2 | 83 % |"], t=60.0),
    ]
    ids = evals.match_slides(truth, out)
    assert ids == {0: 1, 1: 0}
    m = evals.slide_metrics(truth, out, ids)
    assert m["slides_found"] == 1.0 and m["slide_title_exact"] == 1.0
    assert m["slide_cell_accuracy"] == 1.0 and m["slide_number_recall"] == 1.0
    chunks = [Span(t0=55.0, t1=70.0), Span(t0=60.0, t1=80.0)]
    # Slide 1 ("Site Speed") lands on a chunk inside its stretch; "Retention" is too late.
    assert evals.alignment_accuracy(truth, out, ids, chunks) == 0.5


# --- reports and the baseline gate ---------------------------------------------------------


def report(**summary: float) -> Report:
    base = {"fact_recall": 0.9, "citation_accuracy": 0.95, "made_up_items": 0}
    base |= {"uncited_claims": 0, "gap_words_per_min": 0.0}
    return Report(
        created_at=datetime(2026, 10, 11, tzinfo=UTC),
        set="quick",
        git_sha="abc",
        models={},
        prompts={},
        items=[],
        summary=base | summary,
        stages={"map": {"s_per_audio_hour": 100.0, "peak_mem_gb": 9.0}},
    )


def test_baseline_thresholds() -> None:
    old = report()
    assert compare(report(fact_recall=0.85), old) == []  # −5 points: within −5.9 (D16)
    assert compare(report(fact_recall=0.841), old) == []  # exactly −5.9
    assert "fact_recall" in compare(report(fact_recall=0.84), old)[0]
    assert compare(report(citation_accuracy=0.881), old) == []  # D25: −6.9 allowed
    assert "citation_accuracy" in compare(report(citation_accuracy=0.88), old)[0]
    assert compare(report(made_up_items=2), old) == []  # D16: +2 allowed (noise 1)
    assert "made_up_items" in compare(report(made_up_items=3), old)[0]
    ask = report(ask_accuracy=0.98, ask_citation_accuracy=1.0, ask_near_miss=7 / 12)
    assert (
        compare(report(ask_accuracy=0.93, ask_citation_accuracy=0.97, ask_near_miss=5 / 12), ask)
        == []
    )  # D24: near-miss may lose 2 of 12
    assert "ask_near_miss" in compare(report(ask_near_miss=4 / 12), ask)[0]
    assert compare(report(made_up_open_questions=2), report(made_up_open_questions=0)) == []
    assert (
        "made_up_open_questions"
        in compare(report(made_up_open_questions=3), report(made_up_open_questions=0))[0]
    )
    regressions = compare(report(ask_accuracy=0.92, ask_citation_accuracy=0.96), ask)
    assert ["ask_accuracy" in regressions[0], "ask_citation_accuracy" in regressions[1]] == [
        True,
        True,
    ]
    assert "uncited_claims" in compare(report(uncited_claims=2), old)[0]
    assert "gap_words_per_min" in compare(report(gap_words_per_min=0.5), old)[0]
    slow = report()
    slow.stages["map"]["s_per_audio_hour"] = 126.0
    assert "map" in compare(slow, old)[0]
    slow.stages["map"]["s_per_audio_hour"] = 124.0
    assert compare(slow, old) == []


def test_error_histogram_buckets() -> None:
    hist = evals.error_histogram([0.0, -3.0, 5.0, 12.0, -20.0, 90.0], (5, 15, 30, 60))
    assert hist == [
        ("0–5 s", 3),
        ("5–15 s", 1),
        ("15–30 s", 1),
        ("30–60 s", 0),
        ("> 60 s", 1),
    ]


def test_overrides() -> None:
    assert evals.parse_override("fuse.window_s=120") == ("fuse.window_s", 120)
    assert evals.parse_override("vad.chunker=silence") == ("vad.chunker", "silence")
    config = evals.apply_overrides(Config(), {"fuse.window_s": 120, "reduce.temperature": 0.1})
    assert config.fuse.window_s == 120.0 and config.reduce.temperature == 0.1
    with pytest.raises(KeyError):
        evals.apply_overrides(Config(), {"fuse.windw_s": 1})


def test_quick_set_loads() -> None:
    quick = evals.load_set("quick", ROOT)
    assert [i.id for i in quick.items] == [
        "clean-en",
        "noisy-en",
        "pdf-slides-en",
        "slide-conflict-en",
    ]
    assert len({i.seed for i in quick.items}) == 4


# --- generator -----------------------------------------------------------------------------


def test_script_invariants() -> None:
    v = Variant(conflict=True)
    s = generate(5, v)
    kinds = [f.kind for f in s.facts]
    assert kinds.count("decision") == v.decisions and kinds.count("action") == v.actions
    assert kinds.count("number") == v.topics and len(s.slides) == v.topics
    owners = [f.owner for f in s.facts if f.kind == "action"]
    assert len(set(owners)) == len(owners)
    values = [f.value for f in s.facts if f.value is not None]
    assert len(set(values)) == len(values) and all(x > 31 for x in values)
    (conflict,) = [f for f in s.facts if f.misspoken is not None]
    assert conflict.value is not None and conflict.misspoken is not None
    assert TEEN_TO_TY[int(conflict.value) // 1000] * 1000 == conflict.misspoken
    spoken = s.lines[conflict.line].text
    assert f"{int(conflict.misspoken):,}" in spoken and f"{int(conflict.value):,}" not in spoken
    for f in s.facts:  # every fact is spoken on its line
        line = s.lines[f.line].text
        assert (f.owner or "") in line and (f.kind == "number" or f.text in line)
    assert len(s.gaps_after) == len(v.gaps)
    questions = [f for f in s.facts if f.kind == "open_question"]
    assert 1 <= len(questions) <= 2 and len({f.topic for f in questions}) == len(questions)
    assert all(s.lines[f.line].text.endswith(("?", "open.", "yet.")) for f in questions)


def test_open_questions_leave_the_other_facts_as_they_were() -> None:
    def other(s: Script) -> list[tuple]:
        return [
            (f.kind, f.text, f.owner, f.due, f.value) for f in s.facts if f.kind != "open_question"
        ]

    for seed in (1, 2, 3, 4):
        with_q, without = generate(seed, Variant()), generate(seed, Variant(open_questions=(0, 0)))
        assert other(with_q) == other(without)
        assert len(with_q.lines) > len(without.lines)
    # Planted questions share no content word with a decision or a task (a match is unambiguous).
    for q in OPEN_QUESTIONS:
        assert not numbers_in(q)
        for text in DECISIONS + TASKS:
            assert not evals.content(q) & evals.content(text), (q, text)


def fake_say(voice: str, text: str) -> np.ndarray:
    """Deterministic stand-in for `say`: length from the text, tone from the voice."""
    n = int(len(text) * 0.06 * gen_build.SR)
    return (0.1 * np.sin(np.arange(n) * (0.05 + len(voice) / 100))).astype(np.float32)


def test_generator_is_deterministic(tmp_path: Path) -> None:
    spec = ItemSpec(id="t", seed=3, media="audio", slides="pdf", gaps=["noise"])
    a = gen_build.build(spec, tmp_path / "a", fake_say, ROOT)
    b = gen_build.build(spec, tmp_path / "b", fake_say, ROOT)
    assert (tmp_path / "a" / "truth.json").read_bytes() == (
        tmp_path / "b" / "truth.json"
    ).read_bytes()
    assert a == b
    assert (tmp_path / "a" / "deck.pdf").is_file() and (tmp_path / "a" / "meeting.wav").is_file()
    c = gen_build.build(spec.model_copy(update={"seed": 4}), tmp_path / "c", fake_say, ROOT)
    assert c.script_text != a.script_text
    # Gaps sit between lines, facts sit on their lines, slides cover their topics.
    (gap,) = a.gap_spans
    assert all(line.t1 <= gap.t0 or line.t0 >= gap.t1 for line in a.lines)
    for f in a.facts:
        assert any(line.t0 == f.t0 and line.t1 == f.t1 for line in a.lines)
    assert [s.t0 for s in a.slides] == sorted(s.t0 for s in a.slides)


# --- score_item on a run built by hand ------------------------------------------------------


def test_score_item_end_to_end(tmp_path: Path) -> None:
    spec = ItemSpec(id="t", seed=1, media="audio", slides="pdf", gaps=["silence"])
    truth = Truth(
        item=spec,
        gen_hash="x",
        language="en",
        duration_s=60.0,
        media="meeting.wav",
        deck="deck.pdf",
        script_text="The number of open tickets is 409. Priya will update the pricing page by "
        "Friday.",
        lines=[],
        facts=[
            fact("number", id="n1", text="Open tickets: 409", value=409.0, t0=2.0, slide=0),
            fact(id="a1", t0=20.0, due="Friday"),
        ],
        speech_spans=[Span(t0=0.0, t1=30.0)],
        gap_spans=[Span(t0=30.0, t1=45.0)],
        slides=[TruthSlide(slide_id=0, title="Support Queue", cells=["409"], t0=0, t1=30)],
    )
    run_dir = make_run(tmp_path)
    schemas.save(
        run_dir,
        ChunksFile(chunker="vad", chunks=[Chunk(chunk_id=0, t0=0.0, t1=15.0)] + [], gaps=[]),
    )
    schemas.save(
        run_dir,
        Transcript(
            language="en",
            runtime="mlx-speech",
            items=[
                TranscriptItem(
                    chunk_id=0,
                    t0=0.0,
                    t1=15.0,
                    text="The number of open tickets is 409. Priya will update the pricing "
                    "page by Friday.",
                )
            ],
        ),
    )
    schemas.save(
        run_dir, SlidesFile(slides=[slide(0, "Support Queue", ["| Open tickets | 409 |"], 0.0)])
    )
    schemas.save(
        run_dir,
        MapFile(
            target_language="en",
            notes=[
                SegmentNotes(
                    seg_id=0,
                    summary="409 open tickets.",
                    key_points=[],
                    decisions=[],
                    action_items=[],
                    open_questions=[],
                    terms=[],
                    number_repairs=[NumberRepair(original="4,090", repaired="409")],
                    unverified_numbers=["2024"],
                )
            ],
        ),
    )
    schemas.save(
        run_dir,
        notes(
            key_points=[
                Claim(text="There are 409 open tickets.", citations=cite(0.0)),
                Claim(text="Up from 3,100 in 2024, 1.61 million users.", citations=cite(0.0)),
            ],
            action_items=[item("Update the pricing page", "Priya", "Friday", citations=cite(0.0))],
        ),
    )
    manifest = schemas.load(run_dir, Manifest)
    start = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)
    manifest.stages["map"] = StageState(
        status="done",
        started_at=start,
        finished_at=start + timedelta(seconds=6),
        peak_mem_bytes=9_000_000_000,
    )
    schemas.save(run_dir, manifest)

    rep = evals.score_item(truth, run_dir, since=start - timedelta(seconds=1))
    m = rep.metrics
    assert m["fact_recall"] == 1.0 and m["citation_accuracy"] == 0.5  # action cited 20 s early
    assert m["made_up_items"] == 0 and m["uncited_claims"] == 0
    assert m["wer"] == 0.0 and m["gap_words"] == 0 and m["output_language_ok"] == 1
    assert m["slide_alignment"] == 1.0
    assert m["map_numbers_repaired"] == 1 and m["map_numbers_unverified"] == 1
    assert rep.number_repairs == ["4,090 → 409"] and rep.unverified_numbers == ["2024"]
    assert m["unsupported_numbers"] == 3
    assert rep.unsupported_numbers == ["2024", "3100", "1610000"]  # plain digits, no 1.61e+06
    assert m["translation_citations_same"] is None and m["untranslated_claims"] is None
    # "Other (beta)" (6d-1): the translated notes are scored, against the English citations.
    english = schemas.load(run_dir, NotesDoc)

    def translated(sections: NotesSections) -> None:
        doc = TranslatedNotes(
            sections=sections, language="sw", source_language="en", untranslated=1, raw=[]
        )
        schemas.save(run_dir, doc)

    translated(english.sections)
    m2 = evals.score_item(truth, run_dir).metrics
    assert m2["translation_citations_same"] == 1 and m2["untranslated_claims"] == 1
    assert m2["fact_recall"] == 1.0
    first, second = english.sections.key_points
    swapped = [first.model_copy(update={"citations": []}), second]
    translated(english.sections.model_copy(update={"key_points": swapped}))
    assert evals.score_item(truth, run_dir).metrics["translation_citations_same"] == 0
    assert rep.stages["map"] == evals.StageCost(
        seconds=6.0, s_per_audio_hour=360.0, peak_mem_gb=9.0, reused=False
    )

    full = Report(
        created_at=start,
        set="quick",
        git_sha=None,
        models={},
        prompts={},
        items=[rep],
        summary=evals.summarize([rep]),
        stages=evals.stage_summary([rep]),
        citation_errors=[-20.0, -2.0],
    )
    assert full.summary["fact_recall"] == 1.0
    assert full.summary["recall_number"] == 1.0 and full.summary["recall_decision"] is None
    assert full.summary["map_numbers_repaired"] == 1 and full.summary["unsupported_numbers"] == 3
    md = evals.render_markdown(full)
    assert "citation off by -20 s" in md and "| map | 360 | 9 |" in md
    assert "Map number repaired: 4,090 → 409" in md and "number in no source: 3100" in md
    assert m["reduce_groups"] == 0 and m["pass2_group"] == 0  # one pass

    two = schemas.load(run_dir, NotesDoc).model_copy(
        update={
            "passes": 2,
            "two_pass": TwoPass(
                groups=[NotesGroup(seg_ids=[0], slide_ids=[0], raw="")] * 2,
                resolved={"marker": 3, "group": 1},
            ),
        }
    )
    schemas.save(run_dir, two)
    m2 = evals.score_item(truth, run_dir).metrics
    assert (m2["reduce_groups"], m2["pass2_marker"], m2["pass2_claim"], m2["pass2_group"]) == (
        2,
        3,
        0,
        1,
    )
    assert evals.summarize([rep, evals.score_item(truth, run_dir)])["pass2_marker"] == 3
    path = evals.write_report(full, tmp_path)
    assert evals.load_report(path) == full
    assert json.loads(path.read_text())["summary"]["facts"] == 2
    assert isinstance(full.items[0], ItemReport)


def test_noise_over_seeds() -> None:
    runs = [report(fact_recall=r, made_up_items=m) for r, m in ((0.9, 3), (0.86, 5), (0.88, 4))]
    runs[1].stages["map"]["s_per_audio_hour"] = 110.0
    n = evals.noise(runs)
    assert n["fact_recall"] == 0.04 and n["made_up_items"] == 2.0
    assert n["citation_accuracy"] == 0.0 and n["stage:map"] == 0.1
    runs[2].summary["ask_accuracy"] = 0.5  # missing from some runs: no noise figure
    assert "ask_accuracy" not in evals.noise(runs)


def test_gate_tables_drive_compare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(evals.MAX_DROP, "ask_accuracy", 0.1)
    monkeypatch.setitem(evals.MAX_RISE, "made_up_items", 2)
    old = report(ask_accuracy=0.8, made_up_items=3)
    assert compare(report(ask_accuracy=0.7, made_up_items=5), old) == []
    out = compare(report(ask_accuracy=0.69, made_up_items=6), old)
    assert len(out) == 2 and "ask_accuracy" in out[0] and "made_up_items" in out[1]


# --- long meetings (6c) --------------------------------------------------------------------

# sha256 of each quick item's script (project, lines, facts, slides, gaps) before 6c's
# long-meeting generator: quick media must not change.
QUICK_SCRIPTS = {
    "clean-en": "c756efafea95384e",
    "noisy-en": "828ee1ad2db3580c",
    "pdf-slides-en": "8d2d7c332774f379",
    "slide-conflict-en": "8797da481efee227",
}


def _fingerprint(s: Script) -> str:
    text = repr((s.project, s.lines, s.facts, s.slides, s.gaps_after))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def test_quick_scripts_are_unchanged_by_the_long_generator() -> None:
    for spec in evals.load_set("quick", ROOT).items:
        assert _fingerprint(generate(spec.seed, gen_build.variant(spec))) == QUICK_SCRIPTS[spec.id]


def test_lang_items_reuse_the_quick_scripts() -> None:
    # 6d-1: only the notes language changes, so lang and quick results compare item to item.
    twins = {"conflict-de": "slide-conflict-en", "clean-ar": "clean-en", "clean-sw": "clean-en"}
    twins |= {"clean-hi": "clean-en", "clean-bn": "clean-en"}  # 6d-2, D18
    lang = evals.load_set("lang", ROOT)
    targets = {i.id: (i.language, i.target) for i in lang.items}
    assert targets == {
        "conflict-de": ("en", "de"),
        "clean-ar": ("en", "ar"),
        "clean-sw": ("en", "sw"),
        "clean-hi": ("en", "hi"),
        "clean-bn": ("en", "bn"),
        "xling-es": ("es", "en"),
        "xling-ja": ("ja", "en"),
    }
    for spec in lang.items:
        if spec.id not in twins:
            continue
        assert (
            _fingerprint(generate(spec.seed, gen_build.variant(spec)))
            == QUICK_SCRIPTS[twins[spec.id]]
        )


def test_planted_texts_stay_distinguishable() -> None:
    texts = [*DECISIONS, *MORE_DECISIONS, *TASKS, *MORE_TASKS, *OPEN_QUESTIONS]
    texts += MORE_OPEN_QUESTIONS
    assert len(set(texts)) == len(texts)
    sets = [evals.content(t) for t in texts]
    for i, a in enumerate(sets):
        for j in range(i):
            b = sets[j]
            assert len(a & b) / min(len(a), len(b)) < 0.6, (texts[i], texts[j])
    for q in (*OPEN_QUESTIONS, *MORE_OPEN_QUESTIONS):
        assert not numbers_in(q)
        for text in (*DECISIONS, *MORE_DECISIONS, *TASKS, *MORE_TASKS):
            assert not evals.content(q) & evals.content(text), (q, text)
    assert not any(numbers_in(t) for t in MORE_DISCUSSION)
    # Long-meeting filler states nothing Map could list as a decision, task or question.
    cues = ("?", "let's", "we should", "i'd like", "happy to", "will ", "keep an eye", "asked")
    assert not [t for t in MORE_DISCUSSION if any(c in t.casefold() for c in cues)]


@pytest.mark.parametrize("code", ["es", "ja"])
def test_language_tables_line_up_with_english(code: str) -> None:
    # 6d-2: the same seed draws the same facts in every language, so the lists line up.
    lang = LANGS[code]
    for name in ("decisions", "tasks", "open_questions", "weekdays", "relative_dues", "topics"):
        assert len(getattr(lang, name)) == len(getattr(EN, name)), name
    for name in ("decision_lines", "action_lines", "open_question_lines", "topic_intros"):
        assert len(getattr(lang, name)) == len(getattr(EN, name)), name
    assert [(t.unit, t.lo, t.hi, t.step) for t in lang.topics] == [
        (t.unit, t.lo, t.hi, t.step) for t in EN.topics
    ]
    assert len(set(lang.voices)) == 3


@pytest.mark.parametrize("code", ["es", "ja"])
def test_planted_texts_stay_distinguishable_per_language(code: str) -> None:
    lang = LANGS[code]
    texts = [*lang.decisions, *lang.tasks, *lang.open_questions]
    assert len(set(texts)) == len(texts)
    sets = [evals.content(t) for t in texts]
    for i, a in enumerate(sets):
        for j in range(i):
            assert len(a & sets[j]) / min(len(a), len(sets[j])) < 0.6, (texts[i], texts[j])
    assert not any(numbers_in(t) for t in texts)


@pytest.mark.parametrize("code", ["es", "ja"])
def test_script_in_another_language_plants_the_english_seed_facts(code: str) -> None:
    en, other = generate(1, Variant()), generate(1, Variant(language=code))
    assert len(other.lines) == len(en.lines) and other.project == en.project
    assert [(f.kind, f.line, f.owner, f.due, f.value) for f in other.facts] == [
        (f.kind, f.line, f.owner, f.due, f.value) for f in en.facts
    ]
    voices = set(LANGS[code].voices)
    assert {line.voice for line in other.lines} <= voices
    # The words are the language's own: no English template is left.
    english = {line.text for line in en.lines}
    assert not english & {line.text for line in other.lines}
    for f in other.facts:
        if f.kind == "action":
            assert f.owner and f.owner in other.lines[f.line].text  # Latin name, as said
        if f.kind == "number":
            assert str(int(f.value or 0)) in other.lines[f.line].text  # plain digits for `say`


def test_long_meeting_facts_are_unique() -> None:
    spec = evals.load_set("long", ROOT).items[0]
    script = generate(spec.seed, gen_build.variant(spec))
    kinds = Counter(f.kind for f in script.facts)
    assert kinds == {"number": 40, "decision": 20, "action": 20, "open_question": 8}
    assert len(script.slides) == 40
    assert len({s.title for s in script.slides}) == 40  # region in every title
    texts = [f.text for f in script.facts if f.kind != "number"]
    assert len(set(texts)) == len(texts)
    labels = [f.text.rsplit(": ", 1)[0] for f in script.facts if f.kind == "number"]
    assert len(set(labels)) == len(labels)  # "Open tickets (North)": one per region
    values = [f.value for f in script.facts if f.kind == "number"]
    assert len(set(values)) == len(values) and min(v for v in values if v) > 31
    assert any("for the North region is" in line.text for line in script.lines)


def test_long_meeting_ask_sample() -> None:
    spec = evals.load_set("long", ROOT).items[0]
    script = generate(spec.seed, gen_build.variant(spec))
    facts = [
        TruthFact(
            id=f.id, kind=f.kind, text=f.text, owner=f.owner, due=f.due, value=f.value, t0=0, t1=1
        )
        for f in script.facts
    ]
    qs = evals.ask_questions(Truth.model_construct(facts=facts))
    assert Counter(q.kind for q in qs) == {
        "number": 12,
        "action": 6,
        "followup": 1,
        "de_alone": 1,
        "de_history": 1,
        "near_miss": 3,
    }
    assert qs[0].question.endswith("for the North region?")
    actions = [q.question for q in qs if q.kind == "action"]
    assert all(" need to do by " in q for q in actions)  # owners repeat in long-en
    assert len(set(actions)) == len(actions)
    assert "in der Region" in next(q.question for q in qs if q.kind == "de_alone")
    assert evals.sample(facts[:3], 12) == facts[:3]


def test_question_labels_with_and_without_a_region() -> None:
    def fact(text: str) -> TruthFact:
        return TruthFact(id="f1", kind="number", text=text, t0=0, t1=1)

    assert evals._label(fact("Page load (ms): 740")) == NUMBER_LABELS["Page load (ms)"]
    assert evals._label(fact("Page load (ms) (West): 740"))[0] == (
        "the page load time for the West region"
    )


def test_each_set_has_its_own_baseline(tmp_path: Path) -> None:
    assert evals.baseline_path("quick", tmp_path) == tmp_path / "evals" / "baseline.json"
    assert evals.baseline_path("long", tmp_path) == tmp_path / "evals" / "baseline-long.json"
    (tmp_path / "evals").mkdir()
    written = evals.write_baseline(report().model_copy(update={"set": "long"}), tmp_path)
    assert written.name == "baseline-long.json"
    assert "long" not in evals.GATED_SETS and "quick" in evals.GATED_SETS
