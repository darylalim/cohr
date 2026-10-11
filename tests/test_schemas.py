"""PLAN §19: snapshots guard every run file format; round-trips; strict validation."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from cohr import schemas
from cohr.schemas import (
    RUN_FILES,
    ActionItem,
    Chunk,
    ChunksFile,
    Citation,
    Claim,
    Frame,
    FramesFile,
    InputFile,
    Manifest,
    MapActionItem,
    MapFile,
    NotesDoc,
    NotesSections,
    RunFile,
    Segment,
    SegmentNotes,
    SegmentsFile,
    Slide,
    SlidesFile,
    Span,
    StageState,
    Transcript,
    TranscriptItem,
)


@pytest.mark.parametrize("model", RUN_FILES, ids=lambda m: m.__name__)
def test_schema_matches_snapshot(model: type[RunFile]) -> None:
    path = schemas.SNAPSHOT_DIR / schemas.snapshot_name(model)
    assert path.exists(), f"no snapshot for {model.__name__}; run --write-snapshots"
    stored = json.loads(path.read_text())
    current = schemas.snapshot(model)
    if stored["schema"] != current["schema"]:
        if stored["format_version"] == current["format_version"]:
            pytest.fail(
                f"{model.__name__} changed without a FORMAT_VERSION bump. Bump it, then run "
                "`uv run python -m cohr.schemas --write-snapshots` and commit both."
            )
        pytest.fail(f"{model.__name__}: FORMAT_VERSION bumped; regenerate the snapshots")
    assert stored["format_version"] == current["format_version"], "regenerate the snapshots"


def test_every_run_file_has_a_snapshot_and_unique_filename() -> None:
    names = [schemas.snapshot_name(m) for m in RUN_FILES]
    assert len(set(names)) == len(names)
    on_disk = {p.name for p in schemas.SNAPSHOT_DIR.glob("*.json")}
    assert on_disk == set(names), "stale or missing snapshot files"


NOW = datetime(2026, 10, 11, 9, 30, tzinfo=UTC)

SAMPLES: list[RunFile] = [
    Manifest(
        app_version="0.1.0",
        run_id="20261011-093000-x",
        created_at=NOW,
        inputs=[
            InputFile(
                kind="audio",
                path="/a.m4a",
                stored="00-a.m4a",
                mode="link",
                size=1,
                mtime=1.0,
                sha256="0" * 64,
            )
        ],
        config={"seed": 0, "vad": {"threshold": 0.5}},
        stages={"ingest": StageState(status="done", format_version=1, started_at=NOW)},
    ),
    ChunksFile(
        chunker="vad", chunks=[Chunk(chunk_id=0, t0=0.0, t1=12.5)], gaps=[Span(t0=12.5, t1=30.0)]
    ),
    FramesFile(
        frames=[Frame(frame_id=0, path="frames/000.png", pts=0.0, phash="ff00", source="00-v.mp4")]
    ),
    Transcript(
        language="de",
        runtime="mlx-speech",
        items=[TranscriptItem(chunk_id=0, t0=0.0, t1=1.0, text="Grüß Gott")],
    ),
    SlidesFile(
        slides=[
            Slide(
                slide_id=1,
                frame_id=0,
                t=None,
                t_source="none",
                title="Budget",
                text_md="- 14,500",
                tables_md=["| a |"],
                chart_notes="",
                visual_notes="",
            )
        ]
    ),
    SegmentsFile(
        segments=[
            Segment(seg_id=0, t0=0.0, t1=300.0, chunk_ids=[0], transcript_text="hi", slide_ids=[1])
        ],
        unaligned_slide_ids=[],
    ),
    MapFile(
        target_language="ja",
        notes=[
            SegmentNotes(
                seg_id=0,
                summary="要約",
                key_points=[],
                decisions=[],
                action_items=[MapActionItem(owner="Priya", task="screenshots", due="20 October")],
                open_questions=[],
                terms=[],
                raw="{}",
            )
        ],
    ),
    NotesDoc(
        sections=NotesSections(
            summary=[Claim(text="Budget approved", citations=[Citation(kind="seg", id=0, t=12.0)])],
            key_points=[Claim(text="uncited", citations=[])],
            decisions=[],
            action_items=[
                ActionItem(
                    text="Screenshots", citations=[Citation(kind="slide", id=1)], owner="Priya"
                )
            ],
            open_questions=[],
        ),
        passes=1,
        language="en",
    ),
]


def test_samples_cover_every_run_file() -> None:
    assert {type(s) for s in SAMPLES} == set(RUN_FILES)


@pytest.mark.parametrize("obj", SAMPLES, ids=lambda o: type(o).__name__)
def test_round_trip(tmp_path: Path, obj: RunFile) -> None:
    schemas.save(tmp_path, obj)
    assert schemas.load(tmp_path, type(obj)) == obj
    leftovers = [p.name for p in tmp_path.rglob(".*")]
    assert leftovers == [], "atomic write left a temp file"


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Chunk.model_validate({"chunk_id": 0, "t0": 0.0, "t1": 1.0, "speaker": "A"})


def test_enums_are_checked() -> None:
    with pytest.raises(ValidationError):
        Citation.model_validate({"kind": "page", "id": 1})


def test_corrupt_file_raises(tmp_path: Path) -> None:
    (tmp_path / "chunks.json").write_text('{"chunker": "vad"}')
    with pytest.raises(ValidationError):
        schemas.load(tmp_path, ChunksFile)
