"""PLAN §5.5: Map prompt filling, JSON coercion, one retry, invalid-JSON fallback."""

from pathlib import Path

from cohr import mapper, schemas
from cohr.config import Config, load_prompt
from cohr.schemas import MapFile, Segment, SegmentsFile, Slide, SlidesFile
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
