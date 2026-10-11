"""Pure helpers behind the pages (PLAN §7): statuses, rows, cited times, audio clips."""

import io
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from cohr import ui
from cohr.config import Config
from cohr.schemas import (
    Citation,
    Claim,
    InputFile,
    Manifest,
    NotesDoc,
    NotesSections,
    StageState,
)


def manifest(**stages: str) -> Manifest:
    return Manifest(
        app_version="0",
        run_id="r",
        created_at=datetime(2026, 10, 11, tzinfo=UTC),
        inputs=[
            InputFile(kind="audio", path="", stored="a", mode="upload", size=1, mtime=0, sha256="")
        ],
        config=Config().model_dump(mode="json"),
        stages={k: StageState(status=v) for k, v in stages.items()},  # ty: ignore[invalid-argument-type]
    )


def test_run_status() -> None:
    assert ui.run_status(manifest()) == "new"
    assert ui.run_status(manifest(ingest="done", render="done")) == "done"
    assert ui.run_status(manifest(ingest="done", vad="done")) == "incomplete"
    assert ui.run_status(manifest(ingest="done", vad="interrupted")) == "interrupted"
    assert ui.run_status(manifest(ingest="failed", vad="running")) == "running"
    assert not ui.resumable(manifest(ingest="done", render="done"))
    assert ui.resumable(manifest(ingest="done", vad="cancelled"))


def test_memory_rows_are_the_four_model_stages_in_order() -> None:
    m = manifest(ingest="done", reduce="done", transcribe="done", vad="done")
    m.stages["reduce"].peak_mem_bytes = 18 * 10**9
    m.stages["transcribe"].peak_mem_bytes = 5 * 10**9
    m.stages["vad"].peak_mem_bytes = 1
    assert ui.memory_rows(m) == [
        {"stage": "Transcribe", "model": "Transcribe", "peak_gb": 5.0},
        {"stage": "Reduce", "model": "Command R7B", "peak_gb": 18.0},
    ]


def test_cited_times_dedupes_sorts_and_falls_back_to_segment_start() -> None:
    notes = NotesDoc(
        sections=NotesSections(
            summary=[
                Claim(text="a", citations=[Citation(kind="seg", id=1, t=70.0)]),
                Claim(text="b", citations=[Citation(kind="seg", id=1, t=65.0)]),
            ],
            key_points=[Claim(text="c", citations=[Citation(kind="seg", id=1, t=70.0)])],
            decisions=[Claim(text="d", citations=[Citation(kind="seg", id=0, t=None)])],
            action_items=[],
            open_questions=[Claim(text="e", citations=[Citation(kind="slide", id=0)])],
        ),
        passes=1,
        language="en",
        raw="",
    )
    assert ui.cited_times(notes, {0: 0.0, 1: 60.0}) == {0: [0.0], 1: [65.0, 70.0]}
    assert ui.uncited_count(notes) == 0


def test_audio_clip_cuts_the_segment(tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    samples = (np.arange(16000 * 4) % 1000).astype(np.int16)
    sf.write(wav, samples, 16000)
    data, sr = sf.read(io.BytesIO(ui.audio_clip(str(wav), 0.0, 1.0, 2.5)), dtype="int16")
    assert sr == 16000 and len(data) == 24000
    assert np.array_equal(data, samples[16000:40000])
    tail, _ = sf.read(io.BytesIO(ui.audio_clip(str(wav), 0.0, 3.5, 9.0)), dtype="int16")
    assert len(tail) == 8000  # clipped at the end of the file


def test_media_kind() -> None:
    assert ui.media_kind("Talk.MOV") == "video"
    assert ui.media_kind("talk.m4a") == "audio"


def test_run_rows_list_unreadable_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    bad = runs / "20261011-000000-bad"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{")
    assert ui.run_rows(runs) == [{"run": bad.name, "status": "unreadable"}]
