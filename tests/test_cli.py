"""`cohr run / runs / prune / rm` end to end on generated media (ingest only, no models)."""

from pathlib import Path

import pytest

from cohr import schemas, storage
from cohr.cli import main
from cohr.config import runs_dir
from cohr.schemas import Manifest
from tests.conftest import needs_ffmpeg


@needs_ffmpeg
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
