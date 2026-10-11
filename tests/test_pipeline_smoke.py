"""Smoke test with real models (PLAN §8), opt in with the `models` marker.

Phase 3 scope: ingest → vad → transcribe → vision → fuse → map on the Phase 0 fixtures (92 s
two-voice meeting with a 20 s noise + music gap, 3 slide images without timestamps, so Fuse
text-aligns them). Phase 4 extends it to notes.md.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cohr import schemas, storage
from cohr.cli import main
from cohr.config import runs_dir
from cohr.schemas import ChunksFile, Manifest, MapFile, SegmentsFile, SlidesFile, Transcript
from tests.conftest import REAL_HF_HOME, needs_ffmpeg

pytestmark = [pytest.mark.models, needs_ffmpeg]

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if REAL_HF_HOME is None:
        monkeypatch.delenv("COHR_HF_HOME")
    else:
        monkeypatch.setenv("COHR_HF_HOME", REAL_HF_HOME)
    out = tmp_path / "fx"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fixtures.py"), str(out)], check=True)
    return out


def test_pipeline_through_map_on_fixtures(
    fixtures: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    slides = [str(fixtures / f"slide_{i}.png") for i in (1, 2, 3)]
    assert main(["run", str(fixtures / "meeting.wav"), "--slides", *slides]) == 0
    (run_dir,) = storage.list_runs(runs_dir())

    manifest = schemas.load(run_dir, Manifest)
    assert {k: v.status for k, v in manifest.stages.items()} == {
        "ingest": "done",
        "vad": "done",
        "transcribe": "done",
        "vision": "done",
        "fuse": "done",
        "map": "done",
    }

    chunks = schemas.load(run_dir, ChunksFile)
    assert chunks.chunker == "vad"
    assert all(c.t1 - c.t0 <= 30.0 for c in chunks.chunks)
    truth = json.loads((fixtures / "truth.json").read_text())
    for g0, g1 in truth["gap_spans"]:  # the noise + music insert, 1 s padding allowed
        inside = sum(max(0.0, min(c.t1, g1 - 1) - max(c.t0, g0 + 1)) for c in chunks.chunks)
        assert inside == 0.0, "non-speech was sent to Transcribe"

    text = " ".join(i.text for i in schemas.load(run_dir, Transcript).items).lower()
    for word in ("atlas", "budget", "launch"):
        assert word in text

    out = schemas.load(run_dir, SlidesFile)
    assert len(out.slides) == 3
    assert all(s.raw is None for s in out.slides)
    assert "14,500" in out.model_dump_json()

    # Fuse: each slide lands, in order, on a chunk that overlaps the lines discussing it.
    lines = truth["lines"]
    discussed = {0: (1, 2), 1: (3, 4), 2: (5, 6)}  # slide index -> script lines (fixtures.py)
    by_t0 = {c.t0: c for c in chunks.chunks}
    times = []
    for k, slide in enumerate(schemas.load(run_dir, SlidesFile).slides):
        assert slide.t_source == "aligned" and slide.t is not None, slide
        first, last = discussed[k]
        lo, hi = lines[first]["t0"], lines[last + 1]["t0"]
        chunk = by_t0[slide.t]
        assert chunk.t0 < hi and chunk.t1 > lo, (k, slide.t, lo, hi)
        times.append(slide.t)
    assert times == sorted(times)
    segments = schemas.load(run_dir, SegmentsFile)
    assert len(segments.segments) == 1 and segments.unaligned_slide_ids == []
    assert segments.segments[0].slide_ids == [0, 1, 2]

    notes = schemas.load(run_dir, MapFile).notes
    assert len(notes) == 1 and notes[0].raw is False
    assert notes[0].decisions and notes[0].action_items
    dumped = notes[0].model_dump_json()
    assert "14,500" in dumped or "14500" in dumped

    capsys.readouterr()
    assert main(["resume", run_dir.name]) == 0
    err = capsys.readouterr().err
    for stage in ("ingest", "vad", "transcribe", "vision", "fuse", "map"):
        assert f"{stage}: skipped" in err
