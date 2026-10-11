"""`cohr run / runs / prune / rm / models` on generated media (ingest only, no models)."""

from pathlib import Path

import pytest

from cohr import ingest, pipeline, schemas, storage
from cohr.cli import main
from cohr.config import runs_dir
from cohr.schemas import Manifest
from tests.conftest import needs_ffmpeg


@pytest.fixture
def ingest_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "default_stages", lambda: [ingest.STAGE])


@needs_ffmpeg
@pytest.mark.usefixtures("ingest_only")
def test_run_list_prune_rm(tone: Path, slide_pdf: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["run", str(tone), "--slides", str(slide_pdf), "--name", "weekly", "--target", "de"])
        == 0
    )
    (run_dir,) = storage.list_runs(runs_dir())
    assert run_dir.name.endswith("-weekly")
    m = schemas.load(run_dir, Manifest)
    assert m.stages["ingest"].status == "done"
    assert m.config["target_language"] == "de"

    capsys.readouterr()
    assert main(["runs"]) == 0
    assert "ingest:done" in capsys.readouterr().out

    assert main(["resume", run_dir.name]) == 0
    assert "ingest: skipped" in capsys.readouterr().err

    assert main(["prune", run_dir.name]) == 0
    assert not (run_dir / "audio.wav").exists()
    assert schemas.load(run_dir, Manifest).pruned

    assert main(["rm", "-y", run_dir.name]) == 0
    assert storage.list_runs(runs_dir()) == []


def test_run_needs_exactly_one_media(tone: Path) -> None:
    with pytest.raises(SystemExit, match="exactly one"):
        main(["run"])
    with pytest.raises(SystemExit, match="exactly one"):
        main(["run", str(tone), "--video", str(tone)])


def test_unknown_run() -> None:
    with pytest.raises(SystemExit, match="no run"):
        main(["prune", "nope"])


@needs_ffmpeg
def test_run_without_models_stops_before_any_stage(
    tone: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", str(tone)]) == 1  # conftest points the model cache at an empty dir
    assert "cohr models download" in capsys.readouterr().err
    (run_dir,) = storage.list_runs(runs_dir())
    assert schemas.load(run_dir, Manifest).stages == {}


def test_models_lists_missing(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines] == ["vad", "transcribe", "vision", "map", "reduce"]
    assert all("missing" in line for line in lines)
    with pytest.raises(SystemExit, match="unknown model"):
        main(["models", "list", "nope"])


def test_run_rejects_unsupported_language(tone: Path) -> None:
    with pytest.raises(SystemExit):
        main(["run", str(tone), "--lang", "fi"])
