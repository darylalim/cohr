"""Smoke test with real models (PLAN §8), opt in with the `models` marker.

Phase 2 scope: ingest → vad → transcribe → vision on the Phase 0 fixtures (92 s two-voice
meeting with a 20 s noise + music gap, 3 slides). Phase 4 extends it to notes.md.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cohr import schemas, storage
from cohr.cli import main
from cohr.config import runs_dir
from cohr.schemas import ChunksFile, Manifest, SlidesFile, Transcript
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


def test_perception_on_fixtures(fixtures: Path, capsys: pytest.CaptureFixture[str]) -> None:
    slides = [str(fixtures / f"slide_{i}.png") for i in (1, 2, 3)]
    assert main(["run", str(fixtures / "meeting.wav"), "--slides", *slides]) == 0
    (run_dir,) = storage.list_runs(runs_dir())

    manifest = schemas.load(run_dir, Manifest)
    assert {k: v.status for k, v in manifest.stages.items()} == {
        "ingest": "done",
        "vad": "done",
        "transcribe": "done",
        "vision": "done",
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

    capsys.readouterr()
    assert main(["resume", run_dir.name]) == 0
    err = capsys.readouterr().err
    for stage in ("ingest", "vad", "transcribe", "vision"):
        assert f"{stage}: skipped" in err
