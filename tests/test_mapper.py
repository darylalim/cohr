"""PLAN §5.5: Map prompt filling, JSON coercion, one retry, invalid-JSON fallback, number guard."""

import logging
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
    SlideConflict,
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


def test_guard_reads_the_sources_in_the_meetings_language() -> None:
    # 6d-2 step 0, xling-es: "11.800 dólares" was read as 11.8, so Map's "$11.80" was kept;
    # "4.650" on the slide gave 4.65, a second candidate for Map's "465".
    es = mapper.source_numbers("La factura es de 11.800 dólares.", ["4.650", "354.000 $"], "es")
    assert {11800.0, 4650.0, 354000.0} <= es and not {11.8, 4.65, 354.0} & es
    assert mapper.check_numbers("Cloud: $11.80", es, "en")[0] == "Cloud: $11800"
    assert mapper.check_numbers("465 signups", es, "en")[0] == "4650 signups"
    # xling-ja: "35万4000ドル" / "13万4000ドル" were read as 35, 4000 and 13; Map's "$13,000"
    # and "$34,000" were kept through their decimal-comma readings (13 and 34 from "34%").
    ja = mapper.source_numbers(
        "月間経常収益は35万4000ドル。先月は13万4000ドル。再訪率は34%。", [], "ja"
    )
    assert {354000.0, 134000.0, 34.0} <= ja and 35.0 not in ja
    assert mapper.check_numbers("$13,000 last month", ja, "en")[0] == "$134,000 last month"
    # two candidates (354000, 134000): flagged, not guessed
    assert mapper.check_numbers("$34,000", ja, "en") == ("$34,000", [], ["34,000"])


def test_guard_drops_a_candidate_the_claim_states_elsewhere() -> None:
    # 8g: "$34,000, up from $13,000" with 354,000 and 134,000 said. The 13,000 repair states
    # 134,000, so of 34,000's two candidates only 354,000 is left.
    sources = {354000.0, 134000.0}
    text = "Monthly recurring revenue is $34,000, up from $13,000 last month."
    assert mapper.check_numbers(text, sources, "en") == (
        "Monthly recurring revenue is $354,000, up from $134,000 last month.",
        [
            NumberRepair(original="34,000", repaired="354,000"),
            NumberRepair(original="13,000", repaired="134,000"),
        ],
        [],
    )
    # a value kept as written counts too
    assert mapper.check_numbers("$34,000, up from $134,000", sources, "en")[0] == (
        "$354,000, up from $134,000"
    )


def test_guard_still_flags_when_no_other_number_settles_the_candidates() -> None:
    sources = {354000.0, 134000.0}
    assert mapper.check_numbers("$34,000", sources, "en") == ("$34,000", [], ["34,000"])
    # a claim that says one value twice is kept as written
    text = "$134,000, the same $134,000 as last month"
    assert mapper.check_numbers(text, sources, "en") == (text, [], [])
    # another ambiguous number doesn't settle anything
    assert mapper.check_numbers("$34,000 and $34,000", sources, "en")[2] == ["34,000"] * 2
    # xling-ja f9 as said: "34%" too, a third candidate (×1000), so two are left: flagged
    ja = {354000.0, 134000.0, 34.0}
    assert mapper.check_numbers("$34,000, up from $134,000", ja, "en")[2] == ["34,000"]


def test_guard_does_not_prefer_a_digit_edit_over_a_factor() -> None:
    # pdf-slides-en at seed 1: "6,900 candidates" for the 69 said (×100); $8,900 (the cloud
    # bill) is one edit away. Two candidates: flagged, as before 8g.
    sources = mapper.source_numbers("69 candidates. The monthly cloud bill is $8,900.", [], "en")
    text = "6,900 candidates in pipeline vs. 86 last month."
    assert mapper.check_numbers(text, sources, "en")[2] == ["6,900", "86"]


def test_guard_never_repairs_a_one_or_two_digit_number() -> None:
    # 8g, AMI IS1009b: "forty percent ... and the rest we make regular" → Map "40% large
    # buttons, 60% regular buttons"; 60 was "repaired" to the 40 said, one edit away.
    assert _check("40% large, 60% regular", "forty percent large, the rest regular") == (
        "40% large, 60% regular",
        [],
        ["60"],
    )
    assert _check("6 seats", "60 seats")[2] == ["6"]  # nor by a factor of 10
    # three digits still repair, and a day still repairs from the days said
    assert _check("120 seats", "12 seats")[0] == "12 seats"
    assert _check("Due Nov 2nd.", "by the 22nd of November")[0] == "Due Nov 22nd."


def test_slide_conflicts_read_the_meetings_language() -> None:
    # 6d-2 step 1, xling-es: slide "4.650" read the English way (4.65) wasn't the 4650 said,
    # so the notes got "4,650 (slide: 4.65)".
    said = "El número de registros de prueba nuevos este mes es de 4650."
    rows = ["| Métrica | Valor |\n|---|---|\n| Registros de prueba nuevos | 4.650 |"]
    assert mapper.slide_conflicts(said, rows) != []  # the English reading: a false conflict
    assert mapper.slide_conflicts(said, rows, "es") == []
    # a real Spanish conflict keeps the slide value whole in the note
    conflict = said.replace("4650", "4050")
    [c] = mapper.slide_conflicts(conflict, rows, "es")
    text, _ = mapper.mark_conflicts("New signups: 4,050", [c], "en", True, "es")
    assert text == "New signups: 4,050 (slide: 4,650)"


def test_guard_keeps_only_the_target_languages_reading() -> None:
    # 6d-2: keeping follows the same reading as repairing. "$34,000" in English is 34000, not
    # the 34 of "34%"; a language not in DECIMAL_MARK still keeps either reading.
    sources = {34.0, 354000.0, 134000.0}  # xling-ja's segment: "34%", 35万4000, 13万4000
    assert mapper.check_numbers("$34,000", sources, "en") == ("$34,000", [], ["34,000"])
    # Before, any reading kept it (34.0). Alone, 34 is a factor-1000 candidate (D20's rule),
    # so "$34,000" with only "34%" said would now be repaired to "$34": watch the eval's repairs.
    assert mapper.check_numbers("$42,500", {42500.0}, "en") == ("$42,500", [], [])
    assert mapper.check_numbers("42.500 $", {42500.0}, "de") == ("42.500 $", [], [])
    assert mapper.check_numbers("42,500", {42.5}, "ar") == ("42,500", [], [])
    # an English meeting's sources are read as before
    text = "the 22nd, $42,500 and fourteen thousand"
    assert mapper.source_numbers(text, [], "en") == mapper.source_numbers(text, [])


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


# --- speech/slide conflicts -------------------------------------------------------------------

_SAID = (
    "Let's move on to quarterly budget. The approved budget for next quarter is $80,000. "
    "That's compared with $77,500 last quarter. Can you say that again?"
)
_ROWS = ["Metric\nValue", "Approved budget\n$18,000", "Last quarter\n$77,500"]  # vision's rows
_MARKDOWN = [
    "| Metric | Value |\n| --- | --- |\n| Approved budget | $18,000 |\n| Last quarter | $77,500 |"
]
_BUDGET = SlideConflict(label="Approved budget", spoken=80000.0, slide="$18,000")


@pytest.mark.parametrize("tables", [_ROWS, _MARKDOWN])
def test_slide_conflict_found_in_both_table_forms(tables: list[str]) -> None:
    assert mapper.slide_conflicts(_SAID, tables) == [_BUDGET]


def test_table_rows_skip_markdown_rules() -> None:
    assert mapper.table_rows(_MARKDOWN) == [
        ["Metric", "Value"],
        ["Approved budget", "$18,000"],
        ["Last quarter", "$77,500"],
    ]


def test_no_conflict_when_the_slide_value_is_said_or_the_label_isnt() -> None:
    assert mapper.slide_conflicts(_SAID.replace("$80,000", "$18,000"), _ROWS) == []
    assert mapper.slide_conflicts("The budget is $80,000.", _ROWS) == []  # "approved" unsaid
    two = "The approved budget is $80,000, maybe $90,000."
    assert mapper.slide_conflicts(two, _ROWS) == []  # two spoken candidates


def test_mark_keeps_the_spoken_value_and_adds_the_slides() -> None:
    text, used = mapper.mark_conflicts("Quarterly budget approved at $80,000.", [_BUDGET])
    assert text == "Quarterly budget approved at $80,000 (slide: $18,000)." and used == [_BUDGET]
    # another string with the same number but not the label; an already marked string
    assert mapper.mark_conflicts("Hired 80,000 users", [_BUDGET]) == ("Hired 80,000 users", [])
    assert mapper.mark_conflicts(text, [_BUDGET]) == (text, [])


def test_mark_in_the_notes_language() -> None:
    # 6d-1: German Map strings write "80.000 $" and don't name the English label; the note is
    # in German and the slide value in the spoken value's style ("18,000" read in German is 18).
    de = "Quartalsbudget: 80.000 $ im Vergleich zu 77.500 $."
    text, used = mapper.mark_conflicts(de, [_BUDGET], "de", cross_language=True)
    assert text == "Quartalsbudget: 80.000 $ (Folie: $18.000) im Vergleich zu 77.500 $."
    assert used == [_BUDGET]
    assert mapper.mark_conflicts(text, [_BUDGET], "de", True) == (text, [])  # marked already
    # Same-language notes still need the label: the value alone must not mark them.
    assert mapper.mark_conflicts("Hired 80,000 users", [_BUDGET], "en") == (
        "Hired 80,000 users",
        [],
    )
    assert mapper.mark_conflicts(de, [_BUDGET], "de") == (de, [])  # label not named
    # Read the German way, "80,000" is 80: not the spoken value.
    assert mapper.mark_conflicts("Budget 80,000 $", [_BUDGET], "de", True)[1] == []
    # A language without a word in the table gets the English one.
    sw, _ = mapper.mark_conflicts("Bajeti $80,000", [_BUDGET], "sw", True)
    assert sw == "Bajeti $80,000 (slide: $18,000)"


def test_mark_slide_conflicts_walks_the_notes() -> None:
    notes = SegmentNotes(
        seg_id=0,
        summary="Budget of $80,000 approved.",
        key_points=["Approved budget: $80,000", "Page load 740 ms"],
        decisions=[],
        action_items=[MapActionItem(owner="Ann", task="Spend the $80,000 budget", due=None)],
        open_questions=[],
        terms=["budget 80,000"],
    )
    out = mapper.mark_slide_conflicts(notes, [_BUDGET])
    assert out.summary == "Budget of $80,000 (slide: $18,000) approved."
    assert out.key_points == ["Approved budget: $80,000 (slide: $18,000)", "Page load 740 ms"]
    assert out.action_items[0].task == "Spend the $80,000 (slide: $18,000) budget"
    assert out.terms == ["budget 80,000"]
    assert out.slide_conflicts == [_BUDGET]
    assert mapper.mark_slide_conflicts(notes, []) == notes


# --- dates and dues (8c, D41) ---------------------------------------------------------------


def _dated(**fields: object) -> SegmentNotes:
    base: dict[str, object] = {
        "seg_id": 0,
        "summary": "",
        "key_points": [],
        "decisions": [],
        "action_items": [],
        "open_questions": [],
        "terms": [],
    }
    return SegmentNotes.model_validate(base | fields)


GRACE = "Action item for Grace: summarize the competitor research by 11 November. Makes sense."


def test_guard_dates_repairs_map_dates_everywhere() -> None:
    # long-en a-f82: the number guard kept "November 1" because the segment says 1 elsewhere.
    notes = _dated(
        decisions=["Summarize competitor research by November 1."],
        action_items=[
            MapActionItem(owner="Grace", task="Summarize research by November 1.", due="November 1")
        ],
    )
    out = mapper.guard_dates(notes, GRACE + " A month ago, 1 team.", "en", "en")
    assert out.decisions == ["Summarize competitor research by November 11."]
    assert out.action_items == [
        MapActionItem(owner="Grace", task="Summarize research by November 11.", due="November 11")
    ]
    assert out.number_repairs == [NumberRepair(original="November 1", repaired="November 11")] * 3
    assert out.dropped_dues == [] and out.unverified_numbers == []


def test_guard_dates_drops_a_due_the_segment_doesnt_say() -> None:
    # Map's example due on a task with none (8b, 8c a-f61).
    notes = _dated(
        action_items=[
            MapActionItem(owner=None, task="Update revenue dashboard", due="End of next week"),
            MapActionItem(owner="Grace", task="Summarize research", due="the 11th of November"),
        ]
    )
    out = mapper.guard_dates(notes, GRACE, "en", "en")
    assert [a.due for a in out.action_items] == [None, "the 11th of November"]
    assert out.dropped_dues == ["End of next week"]


def test_guard_dates_must_not_fire_across_languages() -> None:
    # Hindi notes of an English meeting: the due's digits are said; dates aren't read in Hindi.
    hindi = MapActionItem(owner="ग्रेस", task="1 नवंबर तक शोध", due="11 नवंबर")
    out = mapper.guard_dates(_dated(action_items=[hindi]), GRACE, "hi", "en")
    assert out.action_items == [hindi] and out.number_repairs == [] and out.dropped_dues == []
    # English notes of a Spanish meeting: dates are left to the number guard.
    english = MapActionItem(owner="Grace", task="Research by November 1", due=None)
    out = mapper.guard_dates(_dated(action_items=[english]), "para el 11 de noviembre", "en", "es")
    assert out.action_items == [english] and out.number_repairs == []


def test_guard_dates_leaves_raw_segments_alone() -> None:
    notes = _dated(summary="not json by November 1", raw=True)
    assert mapper.guard_dates(notes, GRACE, "en", "en") == notes


# --- owners (8d, D42) -----------------------------------------------------------------------

SAID = {"Tom", "Maria", "Laura", "Grace", "Victor"}
PRICING = "Tom will update the pricing page by the 2nd of November. Let me pull up the next slide."


def test_guard_owners_makes_a_persons_decision_their_task() -> None:
    # long-en f22: Map wrote Tom's task only as a decision; Reduce lost it.
    tom = "Tom will update the pricing page by the 2nd of November."
    notes = _dated(decisions=[tom, "Replace the wiki with Notion."])
    out, changes = mapper.guard_owners(notes, PRICING, SAID, "en", "en")
    assert out.decisions == ["Replace the wiki with Notion."]
    assert out.action_items == [
        MapActionItem(
            owner="Tom",
            task="Update the pricing page by the 2nd of November.",
            due="the 2nd of November",
        )
    ]
    assert changes == ["decision → task: Tom: update the pricing page by the 2nd of November."]
    # a due the segment doesn't say is left out (D40)
    out, _ = mapper.guard_owners(notes, "Tom will update the pricing page.", SAID, "en", "en")
    assert out.action_items[0].due is None


def test_guard_owners_reads_the_name_with_accents_and_in_another_language() -> None:
    # xling-es f8: English notes of a Spanish meeting that says "Víctor".
    notes = _dated(decisions=["Victor will write the notes for the next version."])
    said = "Víctor, ¿puedes escribir las notas de la versión?"
    out, _ = mapper.guard_owners(notes, said, SAID, "en", "es")
    assert [(a.owner, a.task) for a in out.action_items] == [
        ("Victor", "Write the notes for the next version.")
    ]


@pytest.mark.parametrize(
    "decision",
    [
        "The remote control will incorporate corporate image.",  # AMI: no person
        "Design will incorporate fashion trends.",  # AMI: not a said name
        "We will drop the legacy export feature.",
        "Laura will send the contract to legal by Friday.",  # a Map task states it (D33)
        "Grace will review the audit.",  # Grace isn't said in this segment
    ],
)
def test_guard_owners_must_not_move(decision: str) -> None:
    task = MapActionItem(owner="Laura", task="Send the contract to legal by Friday", due="Friday")
    notes = _dated(decisions=[decision], action_items=[task])
    text = "Laura will send the contract to legal by Friday. We will drop the export."
    assert mapper.guard_owners(notes, text, SAID, "en", "en") == (notes, [])


def test_guard_owners_fills_an_owner_from_the_task_or_its_sentence() -> None:
    # long-en f31: the owner inside the task; clean-sw f11: the sentence addresses Maria.
    laura = MapActionItem(task="Laura will clean up the shared drive by the end of the month.")
    survey = MapActionItem(
        task="Prepare customer survey by end of next week", due="end of next week"
    )
    text = (
        "Laura will clean up the shared drive by the end of the month. Okay, sounds good. "
        "Maria, can you prepare the customer survey by the end of next week? Okay, will do."
    )
    out, changes = mapper.guard_owners(_dated(action_items=[laura, survey]), text, SAID, "en", "en")
    assert [(a.owner, a.task) for a in out.action_items] == [
        ("Laura", "Clean up the shared drive by the end of the month."),
        ("Maria", survey.task),
    ]
    assert len(changes) == 2


@pytest.mark.parametrize(
    ("text", "notes_language", "source_language"),
    [
        ("Tom, can you ask Maria to prepare the customer survey by Friday?", "en", "en"),
        ("We should prepare the customer survey by the end of the week.", "en", "en"),
        ("The customer survey dashboard was updated on Monday.", "en", "en"),
        ("Maria, ¿puedes preparar la encuesta?", "en", "es"),  # no English sentence to read
        ("Maria, can you prepare the customer survey?", "de", "en"),  # German notes
    ],
)
def test_guard_owners_must_not_fill(text: str, notes_language: str, source_language: str) -> None:
    notes = _dated(action_items=[MapActionItem(task="Prepare the customer survey")])
    assert mapper.guard_owners(notes, text, SAID, notes_language, source_language) == (notes, [])


def test_guard_owners_leaves_raw_segments_alone() -> None:
    notes = _dated(summary="Tom will update the page", decisions=[PRICING], raw=True)
    assert mapper.guard_owners(notes, PRICING, SAID, "en", "en") == (notes, [])


def test_stage_gives_tasks_their_owners(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    reply = (
        '{"summary": "", "key_points": [], "decisions": ["Tom will update the pricing page."], '
        '"action_items": [], "open_questions": [], "terms": []}'
    )
    run_dir = make_run(tmp_path, config=Config())
    schemas.save(run_dir, SlidesFile(slides=[]))
    seg = _seg(0, []).model_copy(update={"transcript_text": "Tom will update the pricing page."})
    schemas.save(run_dir, SegmentsFile(segments=[seg], unaligned_slide_ids=[]))
    lm = FakeLm([reply])
    with caplog.at_level(logging.INFO, logger="cohr"):
        mapper.run_stage(ctx_for(run_dir, "map", fake_manager(Config(), tmp_path, map=lm)))
    notes = schemas.load(run_dir, MapFile).notes[0]
    assert notes.decisions == []
    assert notes.action_items == [MapActionItem(owner="Tom", task="Update the pricing page.")]
    assert "decision → task: Tom" in caplog.text


# --- tasks Map never wrote (8e, D43) --------------------------------------------------------

KENJI = (
    "Maria, can you send the contract to legal by the end of the month? Sure. "
    "Kenji will draft the hiring plan by the end of the month. "
    "Tom, can you hear me? Kenji will be out next week. Grace will love this. "
    "Okay, will do. Tom, can you ask Maria to check the numbers? "
    "Let me pull up the next slide."
)
CUE_SAID = {"Tom", "Maria", "Kenji", "Grace", "Okay"}
CONTRACT = MapActionItem(owner="Maria", task="Send the contract to legal", due="end of the month")


def test_uncovered_cues_lists_the_people_map_gave_no_task() -> None:
    # long-en seg 4: Maria's task is in Map, Kenji's isn't. The must-not-fire sentences are
    # flagged here and left to the model; a reply, two people and Maria's own task are not.
    notes = _dated(action_items=[CONTRACT])
    assert mapper.uncovered_cues(notes, KENJI, CUE_SAID, "en") == [
        ("Kenji", "Kenji will draft the hiring plan by the end of the month."),
        ("Tom", "Tom, can you hear me?"),
        ("Kenji", "Kenji will be out next week."),
        ("Grace", "Grace will love this."),
    ]


def test_uncovered_cues_compares_owners_in_the_meetings_spelling() -> None:
    # clean-hi: Map wrote Omar's owner in Devanagari ("ओमर", Maria's consonants too); his task.
    notes = _dated(action_items=[MapActionItem(owner="ओमर", task="हायरिंग प्लान तैयार करें")])
    text = "Omar will draft the hiring plan. Maria will check the numbers."
    assert mapper.uncovered_cues(notes, text, CUE_SAID | {"Omar"}, "en") == [
        ("Maria", "Maria will check the numbers.")
    ]
    # long-en seg 10: Map's owner is the cue's two words, the addressee the first
    notes = _dated(action_items=[MapActionItem(owner="Omar Crews", task="Read marketing emails")])
    text = "Action item for Omar Crews read the marketing emails by Monday."
    assert mapper.uncovered_cues(notes, text, CUE_SAID | {"Omar", "Crews"}, "en") == []


def test_uncovered_cues_must_not_fire() -> None:
    notes = _dated(action_items=[CONTRACT])
    assert mapper.uncovered_cues(notes, "Kenji, ¿puedes preparar el plan?", CUE_SAID, "es") == []
    assert mapper.uncovered_cues(_dated(summary="x", raw=True), KENJI, CUE_SAID, "en") == []


def test_cue_tasks_owns_each_task_by_its_sentence() -> None:
    cues = [("Kenji", "Kenji will draft the hiring plan."), ("Tom", "Tom, can you hear me?")]
    reply = (
        'Here: {"tasks": [{"line": 1, "task": "Draft the hiring plan", "due": "end of the month"},'
        ' {"line": 2, "task": null, "due": null}, {"line": 7, "task": "Made up", "due": null},'
        ' {"line": 1, "task": "Again", "due": null}]}'
    )
    assert mapper.cue_tasks(cues, reply) == {
        0: MapActionItem(owner="Kenji", task="Draft the hiring plan", due="end of the month")
    }
    assert mapper.cue_tasks(cues, '{"tasks": [{"line": 2, "task": "none"}]}') == {}
    # noisy-en: Tiny Aya closed the list with "}}" instead of "]}"
    broken = '{"tasks": [{"line": 1, "task": "book the venue", "due": "3 October"}}}'
    assert mapper.cue_tasks(cues, broken) == {
        0: MapActionItem(owner="Kenji", task="book the venue", due="3 October")
    }
    assert mapper.cue_tasks(cues, "no JSON") is None
    assert mapper.cue_tasks(cues, '{"tasks": []}') is None


@pytest.mark.parametrize(
    ("task", "whole"),
    [
        # long-en f86: Tiny Aya cut the object short
        ("order laptops", "order laptops for the new starters"),
        ("Order laptops.", "order laptops for the new starters"),
        # must not change: the whole task already, words the sentence doesn't say, part of a word
        ("order laptops for the new starters", "order laptops for the new starters"),
        ("buy laptops", "buy laptops"),
        ("der laptops", "der laptops"),
    ],
)
def test_whole_task(task: str, whole: str) -> None:
    sentence = "Kenji will order laptops for the new starters by the 15th of October."
    assert mapper.whole_task(task, sentence) == whole


def test_cue_tasks_completes_a_task_only_in_english() -> None:
    cues = [("Kenji", "Kenji will order laptops for the new starters by the 15th of October.")]
    reply = '{"tasks": [{"line": 1, "task": "order laptops", "due": null}]}'
    english, other = mapper.cue_tasks(cues, reply, english=True), mapper.cue_tasks(cues, reply)
    assert english and english[0].task == "order laptops for the new starters"
    assert other and other[0].task == "order laptops"


def test_cue_prompt_ends_with_the_target_language() -> None:
    cues = [("Kenji", "Kenji will draft the hiring plan.")]
    prompt = mapper.fill(load_prompt("map_tasks"), **mapper.cue_values(cues, "de"))
    assert "1. Kenji will draft the hiring plan." in prompt
    assert prompt.rstrip().splitlines()[-1].startswith('Write every "task" and "due" in German')


def _cue_run(tmp_path: Path, transcript: str, config: Config) -> Path:
    run_dir = make_run(tmp_path, config=config)
    schemas.save(run_dir, SlidesFile(slides=[]))
    seg = _seg(0, []).model_copy(update={"transcript_text": transcript})
    schemas.save(run_dir, SegmentsFile(segments=[seg], unaligned_slide_ids=[]))
    return run_dir


MAP_REPLY = (
    '{"summary": "", "key_points": [], "decisions": [], "action_items": [{"owner": "Maria", '
    '"task": "Send the contract to legal", "due": null}], "open_questions": [], "terms": []}'
)


def test_stage_adds_the_tasks_a_cue_gives(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cue_reply = (
        '{"tasks": [{"line": 1, "task": "Draft the hiring plan by the 31st", "due": "Friday"},'
        ' {"line": 2, "task": null, "due": null}, {"line": 3, "task": null, "due": null},'
        ' {"line": 4, "task": null, "due": null}]}'
    )
    run_dir = _cue_run(tmp_path, KENJI, Config())
    lm = FakeLm([MAP_REPLY, cue_reply])
    with caplog.at_level(logging.INFO, logger="cohr"):
        mapper.run_stage(ctx_for(run_dir, "map", fake_manager(Config(), tmp_path, map=lm)))
    notes = schemas.load(run_dir, MapFile).notes[0]
    # the due the segment doesn't say is dropped, the number in no source flagged (Map's guards)
    assert notes.action_items[1] == MapActionItem(
        owner="Kenji", task="Draft the hiring plan by the 31st", due=None
    )
    assert notes.dropped_dues == ["Friday"] and notes.unverified_numbers == ["31st"]
    assert lm.temperatures[1] == 0.0 and "4. Grace will love this." in lm.prompts[1]
    assert "task from a cue: Kenji" in caplog.text
    assert "cue, no task: Tom, can you hear me?" in caplog.text


@pytest.mark.parametrize(
    ("transcript", "config"),
    [
        ("Maria, can you send the contract to legal? Okay, will do.", Config()),  # covered
        (KENJI, Config(map=MapConfig(cue_tasks=False))),
    ],
)
def test_stage_makes_no_cue_call(tmp_path: Path, transcript: str, config: Config) -> None:
    run_dir = _cue_run(tmp_path, transcript, config)
    lm = FakeLm([MAP_REPLY])
    mapper.run_stage(ctx_for(run_dir, "map", fake_manager(config, tmp_path, map=lm)))
    assert len(lm.prompts) == 1
    assert [a.owner for a in schemas.load(run_dir, MapFile).notes[0].action_items] == ["Maria"]
