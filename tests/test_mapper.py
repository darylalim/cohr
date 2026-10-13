"""PLAN §5.5: Map prompt filling, JSON coercion, one retry, invalid-JSON fallback, number guard."""

from pathlib import Path

import pytest

from cohr import mapper, schemas
from cohr.config import Config, MapConfig, load_prompt
from cohr.schemas import (
    MapActionItem,
    MapFile,
    NumberRepair,
    Segment,
    SegmentNotes,
    SegmentsFile,
    Slide,
    SlidesFile,
)
from tests.conftest import ctx_for, fake_manager, make_run

GOOD = (
    "```json\n"
    '{"summary": "Budget approved.", "key_points": ["14,500 approved", ""], '
    '"decisions": ["Launch beta on 3 November"], '
    '"action_items": [{"owner": "Priya", "task": "Screenshots", "due": "20 October"}, '
    '{"owner": "null", "task": "Release notes", "due": "end of next week"}, "Close duplicates", '
    '{"owner": "Tom", "task": ""}], '
    '"open_questions": ["Offline mode?"], "terms": "Atlas"}\n```'
)


def test_clock() -> None:
    assert mapper.clock(0) == "0:00"
    assert mapper.clock(754.9) == "12:34"
    assert mapper.clock(3725) == "1:02:05"


def test_fill_is_one_pass_and_keeps_json_braces() -> None:
    template = load_prompt("map")
    seg = Segment(seg_id=0, t0=60, t1=360, chunk_ids=[0], transcript_text="{slides}", slide_ids=[])
    text = mapper.fill(template, **mapper.prompt_values(seg, [], "de"))
    assert '{"summary": string' in text  # literal JSON schema survives
    assert "Time range: 1:00–6:00" in text
    assert "Transcript:\n{slides}" in text  # a value is never substituted again
    assert "Slides shown:\n(none)" in text
    assert text.rstrip().splitlines()[-1] == (
        "Write every JSON string value in German. Return only the JSON object."
    )


def test_retry_prompt_still_ends_with_the_target_language() -> None:
    retry = mapper.fill(load_prompt("map_retry"), target_language="Japanese")
    assert retry.rstrip().endswith("write every string value in Japanese.")


def test_parse_and_coerce() -> None:
    notes = mapper.make_notes(3, mapper.parse(GOOD), GOOD)
    assert notes.seg_id == 3 and notes.raw is False
    assert notes.key_points == ["14,500 approved"]
    assert notes.terms == ["Atlas"]
    assert [(a.owner, a.task, a.due) for a in notes.action_items] == [
        ("Priya", "Screenshots", "20 October"),
        (None, "Release notes", "end of next week"),
        (None, "Close duplicates", None),
    ]
    assert mapper.make_notes(0, {"summary": ["a", "b"]}, "").summary == "a b"


def test_parse_rejects_non_map_replies() -> None:
    assert mapper.parse("no json") is None
    assert mapper.parse('{"summary": ') is None
    assert mapper.parse('{"title": "slide"}') is None
    assert mapper.parse("[1]") is None


def test_raw_fallback_keeps_the_text() -> None:
    notes = mapper.make_notes(1, None, "  Das Budget wurde genehmigt.  ")
    assert notes.raw is True
    assert notes.summary == "Das Budget wurde genehmigt."
    assert notes.decisions == [] and notes.action_items == []


class FakeLm:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.seeds: list[int] = []
        self.temperatures: list[float] = []

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
        self.prompts.append(prompt)
        self.seeds.append(seed)
        self.temperatures.append(temperature)
        return self.replies.pop(0)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _slide(sid: int, title: str) -> Slide:
    return Slide(
        slide_id=sid,
        frame_id=sid,
        t=None,
        t_source="none",
        title=title,
        text_md="Approved: $14,500",
        tables_md=[],
        chart_notes="",
        visual_notes="",
    )


def _run(tmp_path: Path, segments: list[Segment]) -> Path:
    run_dir = make_run(tmp_path, config=Config(seed=5, target_language="ja"))
    schemas.save(run_dir, SlidesFile(slides=[_slide(0, "Q4 Budget")]))
    schemas.save(run_dir, SegmentsFile(segments=segments, unaligned_slide_ids=[]))
    return run_dir


def _seg(seg_id: int, slide_ids: list[int]) -> Segment:
    return Segment(
        seg_id=seg_id,
        t0=seg_id * 300.0,
        t1=seg_id * 300.0 + 290,
        chunk_ids=[seg_id],
        transcript_text=f"transcript {seg_id}",
        slide_ids=slide_ids,
    )


def test_stage_retries_once_then_falls_back(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, [_seg(0, [0]), _seg(1, [])])
    lm = FakeLm(["oops", GOOD, "still not json", "nope"])
    ctx = ctx_for(run_dir, "map", fake_manager(Config(), tmp_path, map=lm))

    mapper.run_stage(ctx)

    out = schemas.load(run_dir, MapFile)
    assert out.target_language == "ja"
    assert [(n.seg_id, n.raw) for n in out.notes] == [(0, False), (1, True)]
    assert out.notes[1].summary == "nope"
    assert len(lm.prompts) == 4
    assert lm.seeds == [5] * 4 and lm.temperatures == [0.3] * 4
    first, retry = lm.prompts[:2]
    assert "Q4 Budget\nApproved: $14,500" in first and "transcript 0" in first
    assert first.rstrip().endswith(
        "Write every JSON string value in Japanese. Return only the JSON object."
    )
    assert retry.startswith(first.rstrip()) and "not valid JSON" in retry
    assert retry.rstrip().endswith("in Japanese.")
    assert "Slides shown:\n(none)" in lm.prompts[2]


def test_stage_with_no_segments_loads_no_model(tmp_path: Path) -> None:
    run_dir = _run(tmp_path, [])
    manager = fake_manager(Config(), tmp_path, map=FakeLm([]))
    mapper.run_stage(ctx_for(run_dir, "map", manager))
    assert schemas.load(run_dir, MapFile).notes == []
    assert manager.loaded() == []


# --- number guard ---------------------------------------------------------------------------


def _check(text: str, source: str) -> tuple[str, list[NumberRepair], list[str]]:
    return mapper.check_numbers(text, mapper.source_numbers(source, []))


@pytest.mark.parametrize(
    ("text", "source", "fixed"),
    [  # the 4b-1 cases: the transcript was right, Map changed one digit or a factor of 10
        ("Budget of $420,500 approved.", "a budget of $42,500", "Budget of $42,500 approved."),
        ("570,000 monthly users", "57,000 monthly users", "57,000 monthly users"),
        ("3,360 open tickets", "336 open tickets", "336 open tickets"),
        ("Due Nov 2nd.", "by the 22nd of November", "Due Nov 22nd."),
        ("Spend $595,000", "$59,500", "Spend $59,500"),
        ("Cap at $1,180", "$11,800", "Cap at $11,800"),
        ("Churn fell to 340%", "churn fell to 34%", "Churn fell to 34%"),
        ("Budget 420.500 €", "42.500 Euro", "Budget 42.500 €"),  # thousands dot stays a dot
        ("4,6500 signups", "4,650 signups", "4650 signups"),  # malformed: decimal comma ×1000
        ("vs. 4,246 last week", "336 now, 426 last week", "vs. 426 last week"),
    ],
)
def test_guard_repairs_one_close_source_number(text: str, source: str, fixed: str) -> None:
    out, repairs, unverified = _check(text, source)
    assert out == fixed
    assert len(repairs) == 1 and unverified == []


def test_guard_records_the_repair_as_written() -> None:
    _, repairs, _ = _check("$420,500", "$42,500")
    assert repairs == [NumberRepair(original="420,500", repaired="42,500")]


def test_guard_keeps_numbers_in_the_sources() -> None:
    text = "$42,500 for 57,000 users (42.5k), up 34% on the 22nd, 1.5 million"
    assert _check(text, "42,500 dollars, 57000 users, 34%, the 22nd, 1.5 million") == (
        text,
        [],
        [],
    )
    assert _check("forty-two items", "") == ("forty-two items", [], [])  # words aren't checked


def test_guard_flags_a_number_two_edits_away() -> None:
    assert _check("$43,600 approved", "$42,500 approved") == ("$43,600 approved", [], ["43,600"])


def test_guard_flags_two_equally_close_candidates() -> None:
    assert _check("450 seats", "45 or 451 seats") == ("450 seats", [], ["450"])
    assert _check("by the 2nd", "the 22nd, one item") == ("by the 2nd", [], ["2nd"])  # 22 or 1


def test_guard_repairs_only_from_the_target_languages_reading() -> None:
    # 6b-2: Map wrote "$770,000" for the spoken $77,500; its decimal-comma reading 770.0 is one
    # digit from 740 (another figure), so without a language it was "repaired" to $740.
    sources = {77500.0, 740.0}
    assert mapper.check_numbers("$770,000", sources) == (
        "$740",
        [NumberRepair(original="770,000", repaired="740")],
        [],
    )
    assert mapper.check_numbers("$770,000", sources, "en") == ("$770,000", [], ["770,000"])
    assert mapper.check_numbers("770.000 €", sources, "de") == ("770.000 €", [], ["770.000"])
    # the D20 case still repairs: in English "$425,000" is 425000, ×10 from 42500
    assert mapper.check_numbers("$425,000", {42500.0}, "en")[0] == "$42,500"
    # a language not in the table keeps both readings
    assert mapper.check_numbers("$770,000", sources, "es")[0] == "$740"


def test_guard_falls_back_to_every_reading_when_none_is_native() -> None:
    # "3,36" isn't an English number; only the decimal-comma 3.36 reads it (×100 → 336)
    assert mapper.check_numbers("3,36 open tickets", {336.0}, "en")[0] == "336 open tickets"


def test_guard_numbers_passes_the_language_on() -> None:
    notes = SegmentNotes(
        seg_id=0,
        summary="Budget $770,000.",
        key_points=["Page load 740 ms"],
        decisions=[],
        action_items=[],
        open_questions=[],
        terms=[],
    )
    out = mapper.guard_numbers(notes, {77500.0, 740.0}, "en")
    assert out.summary == "Budget $770,000." and out.unverified_numbers == ["770,000"]


def test_guard_repairs_an_ordinal_only_from_the_days_said() -> None:
    # 6b-2: "by the 22nd of November" → Map "November 2nd" on every seed. Against every source
    # number 2 is one edit from 22, 12 and 1, so it was flagged; among the days said only 22 is.
    source = "Nadia, can you prepare the survey by the 22nd of November? 12 seats, 1 team."
    sources, days = mapper.source_numbers(source, []), mapper.source_days(source, [])
    assert mapper.check_numbers("by November 2nd", sources)[2] == ["2nd"]
    assert mapper.check_numbers("by November 2nd", sources, "en", days) == (
        "by November 22nd",
        [NumberRepair(original="2", repaired="22")],
        [],
    )
    # a day near no spoken day stays flagged; a plain number still uses every source
    assert mapper.check_numbers("by the 5th", sources, "en", days)[2] == ["5th"]
    assert mapper.check_numbers("120 seats", sources, "en", days)[0] == "12 seats"


def test_source_days_include_slides() -> None:
    assert mapper.source_days("due the twenty-second", ["Launch: 3rd of May"]) == {22.0, 3.0}


def test_guard_flags_an_invented_iso_year() -> None:
    text, repairs, unverified = _check("Due 2024-11-22", "by the 22nd of November")
    assert text == "Due 2024-11-22" and repairs == []
    assert "2024" in unverified


def test_guard_only_flags_suffixed_and_space_grouped_numbers() -> None:
    assert _check("42.6k users", "42,500 users") == ("42.6k users", [], ["42.6k"])
    assert _check("4 250 €", "42500 Euro") == ("4 250 €", [], ["4 250"])
    assert _check("2 300 users", "2 teams of 300 users")[2] == []  # two source numbers


def test_guard_numbers_walks_every_field_but_terms() -> None:
    notes = SegmentNotes(
        seg_id=0,
        summary="Budget $420,500.",
        key_points=["3,360 tickets", "Q3 plan"],
        decisions=["Cap at 3,360"],
        action_items=[MapActionItem(owner="Elena", task="Close 3,360 tickets", due="Nov 2nd")],
        open_questions=["Why 999?"],
        terms=["R2 3,360"],
    )
    out = mapper.guard_numbers(notes, mapper.source_numbers("$42,500, 336 tickets, the 22nd", []))
    assert out.summary == "Budget $42,500."
    assert out.key_points == ["336 tickets", "Q3 plan"]
    assert out.decisions == ["Cap at 336"]
    assert out.action_items == [
        MapActionItem(owner="Elena", task="Close 336 tickets", due="Nov 22nd")
    ]
    assert out.open_questions == ["Why 999?"]
    assert out.terms == ["R2 3,360"]
    assert [(r.original, r.repaired) for r in out.number_repairs] == [
        ("420,500", "42,500"),
        ("3,360", "336"),
        ("3,360", "336"),
        ("3,360", "336"),
        ("2", "22"),
    ]
    assert out.unverified_numbers == ["999"]


def test_guard_leaves_raw_segments_alone() -> None:
    raw = mapper.make_notes(0, None, "Budget $420,500, not JSON")
    assert mapper.guard_numbers(raw, {42500.0}) == raw


GUARDED = (
    '{"summary": "Budget $420,500 approved.", "key_points": ["14,500 on the slide"], '
    '"decisions": [], "action_items": [], "open_questions": ["What about 2024?"], "terms": []}'
)


def _guard_run(tmp_path: Path, config: Config) -> MapFile:
    run_dir = make_run(tmp_path, config=config)
    schemas.save(run_dir, SlidesFile(slides=[_slide(0, "Budget")]))
    seg = _seg(0, [0]).model_copy(update={"transcript_text": "We approved $42,500 today."})
    schemas.save(run_dir, SegmentsFile(segments=[seg], unaligned_slide_ids=[]))
    lm = FakeLm([GUARDED])
    mapper.run_stage(ctx_for(run_dir, "map", fake_manager(config, tmp_path, map=lm)))
    return schemas.load(run_dir, MapFile)


def test_stage_applies_the_guard_with_transcript_and_slides(tmp_path: Path) -> None:
    notes = _guard_run(tmp_path, Config()).notes[0]
    assert notes.summary == "Budget $42,500 approved."
    assert notes.key_points == ["14,500 on the slide"]  # from the attached slide's text
    assert notes.number_repairs == [NumberRepair(original="420,500", repaired="42,500")]
    assert notes.unverified_numbers == ["2024"]


def test_stage_guard_can_be_switched_off(tmp_path: Path) -> None:
    notes = _guard_run(tmp_path, Config(map=MapConfig(number_guard=False))).notes[0]
    assert notes.summary == "Budget $420,500 approved."
    assert notes.number_repairs == [] and notes.unverified_numbers == []
