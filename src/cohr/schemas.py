"""Data contracts for every run file (PLAN §19).

Stages read and write run files only through `load` and `save`. Each file model carries a
FORMAT_VERSION; changing a model means bumping it and regenerating the snapshots:

    uv run python -m cohr.schemas --write-snapshots

Times are seconds as floats, ids are ints, enums are Literals.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from cohr import atomic


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunFile(_Model):
    """A top-level run file. FORMAT_VERSION feeds the owning stage's config hash."""

    FORMAT_VERSION: ClassVar[int]
    FILENAME: ClassVar[str]


# --- manifest.json -------------------------------------------------------------------------

StageStatus = Literal["pending", "running", "done", "failed", "cancelled", "interrupted"]
InputKind = Literal["audio", "video", "slides"]


class InputFile(_Model):
    kind: InputKind
    path: str  # original absolute path ("" for uploads)
    stored: str  # name inside source/
    mode: Literal["link", "copy", "upload"]
    size: int
    mtime: float
    sha256: str


class ModelRef(_Model):
    repo: str
    revision: str


class StageState(_Model):
    status: StageStatus = "pending"
    format_version: int | None = None
    config_hash: str | None = None  # stage config + prompt + format version
    input_hash: str | None = None  # chained from the inputs and every earlier stage
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    finished_at: datetime | None = None
    peak_mem_bytes: int | None = None
    error: str | None = None


class Manifest(RunFile):
    FORMAT_VERSION: ClassVar[int] = 1
    FILENAME: ClassVar[str] = "manifest.json"

    app_version: str
    run_id: str
    created_at: datetime
    inputs: list[InputFile]
    config: dict[str, JsonValue]  # Config.model_dump(mode="json"); kept loose on purpose
    models: dict[str, ModelRef] = {}
    stages: dict[str, StageState] = {}
    pruned: bool = False


# --- chunks.json (vad) ---------------------------------------------------------------------


class Span(_Model):
    t0: float
    t1: float


class Chunk(Span):
    chunk_id: int


class ChunksFile(RunFile):
    FORMAT_VERSION: ClassVar[int] = 1
    FILENAME: ClassVar[str] = "chunks.json"

    chunker: Literal["vad", "silence"]
    chunks: list[Chunk]
    gaps: list[Span]


# --- frames/frames.json (ingest) -----------------------------------------------------------


class Frame(_Model):
    frame_id: int
    path: str  # relative to the run dir, e.g. frames/000.png
    pts: float | None  # seconds into the video; None for slide files
    phash: str  # hex
    source: str  # InputFile.stored
    page: int | None = None  # 1-based page for PDFs


class FramesFile(RunFile):
    FORMAT_VERSION: ClassVar[int] = 1
    FILENAME: ClassVar[str] = "frames/frames.json"

    frames: list[Frame]


# --- transcript.json (transcribe) ----------------------------------------------------------


class TranscriptItem(Span):
    chunk_id: int
    text: str


class Transcript(RunFile):
    FORMAT_VERSION: ClassVar[int] = 1
    FILENAME: ClassVar[str] = "transcript.json"

    language: str
    runtime: Literal["mlx-speech", "mlx-audio"]
    items: list[TranscriptItem]


# --- slides.json (vision) ------------------------------------------------------------------


class Slide(_Model):
    slide_id: int
    frame_id: int
    t: float | None
    t_source: Literal["pts", "aligned", "none"]
    align_score: float | None = None
    title: str
    text_md: str
    tables_md: list[str]
    chart_notes: str
    visual_notes: str
    raw: str | None = None


class SlidesFile(RunFile):
    FORMAT_VERSION: ClassVar[int] = 1
    FILENAME: ClassVar[str] = "slides.json"

    slides: list[Slide]


# --- segments.json (fuse) ------------------------------------------------------------------


class Segment(Span):
    seg_id: int
    chunk_ids: list[int]
    transcript_text: str
    slide_ids: list[int]


class SegmentsFile(RunFile):
    FORMAT_VERSION: ClassVar[int] = 1
    FILENAME: ClassVar[str] = "segments.json"

    segments: list[Segment]
    unaligned_slide_ids: list[int]


# --- map.json (map) ------------------------------------------------------------------------


class MapActionItem(_Model):
    owner: str | None = None
    task: str
    due: str | None = None


class NumberRepair(_Model):
    original: str  # the number as Map wrote it, e.g. "420,500"
    repaired: str  # the source number in the same style, e.g. "42,500"


class SlideConflict(_Model):
    label: str  # the slide table row's label, e.g. "Approved budget"
    spoken: float  # the value said for it, e.g. 80000
    slide: str  # the row's value as shown, e.g. "$18,000"


class SegmentNotes(_Model):
    seg_id: int
    summary: str
    key_points: list[str]
    decisions: list[str]
    action_items: list[MapActionItem]
    open_questions: list[str]
    terms: list[str]
    raw: bool = False  # the reply was never valid JSON: summary holds the raw text (§5.5)
    # Number guard (§5.5, D20): digits Map changed and the guard put back, and numbers in no
    # source with no single close source number (kept as written, never dropped or guessed).
    number_repairs: list[NumberRepair] = []
    unverified_numbers: list[str] = []
    # Speech/slide conflicts (§5.5, 6b-2): a slide row whose value was never said while another
    # was said for its label. The Map strings keep the spoken value and add the slide's.
    slide_conflicts: list[SlideConflict] = []


class MapFile(RunFile):
    # v3: number guard (number_repairs, unverified_numbers); v4: slide_conflicts. Older runs
    # re-run Map.
    FORMAT_VERSION: ClassVar[int] = 4
    FILENAME: ClassVar[str] = "map.json"

    target_language: str
    notes: list[SegmentNotes]


# --- notes.json (reduce) -------------------------------------------------------------------


class Citation(_Model):
    kind: Literal["seg", "slide"]
    id: int
    t: float | None = None  # refined jump time, seg citations only (§5.6)


class Claim(_Model):
    text: str
    citations: list[Citation]  # [] = uncited: kept and flagged, never invented (§5.6)


class ActionItem(Claim):
    owner: str | None = None
    due: str | None = None


class NotesSections(_Model):
    summary: list[Claim]
    key_points: list[Claim]
    decisions: list[Claim]
    action_items: list[ActionItem]
    open_questions: list[Claim]


class NotesGroup(_Model):
    """One pass-1 group of a two-pass reduce (§4): consecutive segments and their slides."""

    seg_ids: list[int]
    slide_ids: list[int]
    raw: str  # R7B's pass-1 reply for the group


class TwoPass(_Model):
    groups: list[NotesGroup]
    # How each pass-2 citation of a group resolved (§4): "marker" ([seg-N] markers in the
    # cited span), "claim" (the best-matching pass-1 claim's citations), "group" (every
    # segment of the group), "uncited" (it matched an uncited pass-1 claim: nothing invented).
    resolved: dict[str, int]


class NotesDoc(RunFile):
    # v3: a seg citation's `t` is placed inside the chunk at the cited sentence (D19), so
    # older runs re-run Reduce (same reply at temperature 0, new jump times).
    # v4: two_pass (pass-1 groups and how pass-2 citations resolved, §4).
    FORMAT_VERSION: ClassVar[int] = 4
    FILENAME: ClassVar[str] = "notes.json"

    sections: NotesSections
    passes: Literal[1, 2]
    language: str
    raw: str  # R7B's reply with citation markup, as generated (debugging, evals)
    two_pass: TwoPass | None = None  # passes == 2 only


class TranslatedNotes(RunFile):
    """The notes in an "Other (beta)" target language (§5.6, 6d-1): notes.json's claims, in the
    same sections and order, with the same citations; only the text (and an action item's due
    date) is translated. Written by the translate stage; absent for R7B's 23 languages."""

    FORMAT_VERSION: ClassVar[int] = 1
    FILENAME: ClassVar[str] = "notes_translated.json"

    sections: NotesSections
    language: str  # the target language
    source_language: str  # notes.json's language ("en")
    untranslated: int  # claims kept in the source language (reply unreadable twice)
    raw: list[str]  # Tiny Aya's replies, as generated (debugging, evals)


# --- I/O -----------------------------------------------------------------------------------

RUN_FILES: tuple[type[RunFile], ...] = (
    Manifest,
    ChunksFile,
    FramesFile,
    Transcript,
    SlidesFile,
    SegmentsFile,
    MapFile,
    NotesDoc,
    TranslatedNotes,
)


def load[M: RunFile](run_dir: Path, model: type[M]) -> M:
    """Read and validate a run file. Raises pydantic.ValidationError if it doesn't match."""
    return model.model_validate_json((run_dir / model.FILENAME).read_bytes())


def save(run_dir: Path, obj: RunFile) -> None:
    """Validate-on-write is implicit (pydantic models); the write itself is atomic."""
    atomic.write_text(run_dir / obj.FILENAME, obj.model_dump_json(indent=2) + "\n")


# --- schema snapshots (tests/test_schemas.py) ----------------------------------------------

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "tests" / "schemas"


def snapshot_name(model: type[RunFile]) -> str:
    return Path(model.FILENAME).name


def snapshot(model: type[RunFile]) -> dict[str, Any]:
    return {"format_version": model.FORMAT_VERSION, "schema": model.model_json_schema()}


def write_snapshots(directory: Path = SNAPSHOT_DIR) -> None:
    for model in RUN_FILES:
        text = json.dumps(snapshot(model), indent=2, sort_keys=True) + "\n"
        atomic.write_text(directory / snapshot_name(model), text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-snapshots", action="store_true", help="regenerate tests/schemas")
    if parser.parse_args().write_snapshots:
        write_snapshots()
        print(f"wrote {len(RUN_FILES)} snapshots to {SNAPSHOT_DIR}")
    else:
        parser.print_help()
