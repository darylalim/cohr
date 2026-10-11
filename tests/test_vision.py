"""PLAN §5.3: slide JSON parsing, Phase 0 quirks, one retry, raw-text fallback."""

from pathlib import Path

import pytest

from cohr import schemas, vision
from cohr.config import Config, load_prompt
from cohr.schemas import Frame, FramesFile, SlidesFile
from tests.conftest import ctx_for, fake_manager, make_run

GOOD = (
    'Sure! {"title": "Budget", "text_md": "- Q3: $14,500", "tables_md": ["| a | b |"], '
    '"chart_notes": "None", "visual_notes": ""}'
)


def test_parse_finds_the_object_and_rejects_non_slides() -> None:
    assert (vision.parse(GOOD) or {}).get("title") == "Budget"
    assert vision.parse("no json here") is None
    assert vision.parse('{"title": "x",') is None
    assert vision.parse('{"foo": 1}') is None
    assert vision.parse("[1, 2]") is None


def test_tables_from_row_objects_become_markdown() -> None:
    rows = [{"Item": "Cloud", "Cost": "$9,000"}, {"Item": "Tools", "Cost": "$5,500"}]
    (table,) = vision.tables(rows)
    assert table.splitlines() == [
        "| Item | Cost |",
        "| --- | --- |",
        "| Cloud | $9,000 |",
        "| Tools | $5,500 |",
    ]


def test_tables_other_shapes() -> None:
    assert vision.tables(["| a |\n| --- |", ""]) == ["| a |\n| --- |"]
    assert vision.tables("None") == []
    assert vision.tables(None) == []
    (table,) = vision.tables([[["h1", "h2"], ["1", "2|3"]]])
    assert table.splitlines()[-1] == "| 1 | 2\\|3 |"
    two = vision.tables([[{"a": 1}], [{"b": 2}]])
    assert len(two) == 2


def _frame(fid: int, pts: float | None) -> Frame:
    return Frame(frame_id=fid, path=f"frames/{fid:03d}.png", pts=pts, phash="0", source="x")


def test_make_slide_normalizes_empty_values_and_timing() -> None:
    slide = vision.make_slide(0, _frame(3, 12.5), vision.parse(GOOD), GOOD)
    assert (slide.slide_id, slide.frame_id, slide.t, slide.t_source) == (0, 3, 12.5, "pts")
    assert slide.chart_notes == "" and slide.raw is None
    assert slide.text_md == "- Q3: $14,500"
    pdf = vision.make_slide(1, _frame(4, None), {"title": "T"}, "{}")
    assert (pdf.t, pdf.t_source, pdf.text_md, pdf.tables_md) == (None, "none", "", [])


class FakeVlm:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.seeds: list[int] = []

    def describe(
        self, image: Path, prompt: str, *, max_tokens: int, temperature: float, seed: int
    ) -> str:
        assert image.name.endswith(".png")
        self.prompts.append(prompt)
        self.seeds.append(seed)
        return self.replies.pop(0)


@pytest.fixture
def run_with_frames(tmp_path: Path) -> Path:
    run_dir = make_run(tmp_path, config=Config(seed=7))
    schemas.save(run_dir, FramesFile(frames=[_frame(0, 1.0), _frame(1, None)]))
    return run_dir


def test_retry_once_then_keep_raw_text(run_with_frames: Path, tmp_path: Path) -> None:
    fake = FakeVlm(["garbled", GOOD, "still not json", "nope"])
    models = fake_manager(Config(), tmp_path, vision=fake)
    vision.run_stage(ctx_for(run_with_frames, "vision", models))

    slides = schemas.load(run_with_frames, SlidesFile).slides
    assert slides[0].title == "Budget" and slides[0].raw is None
    assert slides[1].title == "" and slides[1].text_md == "nope" and slides[1].raw == "nope"
    retry = load_prompt("vision_retry").strip()
    assert [retry in p for p in fake.prompts] == [False, True, False, True]
    assert fake.seeds == [7, 7, 7, 7]
    assert models.loaded() == ["vision"]  # resident policy keeps it
    assert fake.replies == []


def test_sequential_policy_releases_vision(run_with_frames: Path, tmp_path: Path) -> None:
    fake = FakeVlm([GOOD, GOOD])
    models = fake_manager(Config(memory_policy="sequential"), tmp_path, vision=fake)
    vision.run_stage(ctx_for(run_with_frames, "vision", models))
    assert models.loaded() == []
