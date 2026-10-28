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


def test_run_without_ffmpeg_says_how_to_install_it(
    tone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _tool: None)
    with pytest.raises(SystemExit, match="ffprobe not found on PATH.*brew install ffmpeg"):
        main(["run", str(tone)])
    assert storage.list_runs(runs_dir()) == []


def test_eval_without_ffmpeg_stops_before_generating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _tool: None)
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(SystemExit, match="ffmpeg not found on PATH"):
        main(["eval", "--set", "quick", "--item", "clean-en", "--root", str(root)])


def test_models_download_offline_is_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(_spec: object) -> int:
        raise ConnectionError("[Errno 8] nodename nor servname provided")

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *a, **k: offline(None))
    with pytest.raises(SystemExit, match="error: can't reach the Hugging Face Hub"):
        main(["models", "download", "vad"])


@needs_ffmpeg
def test_a_model_deleted_after_a_run_stops_the_resume_cleanly(
    tone: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The run's ingest finished earlier; its models are gone now (conftest: empty model cache).
    every_stage = pipeline.default_stages
    monkeypatch.setattr(pipeline, "default_stages", lambda: [ingest.STAGE])
    assert main(["run", str(tone)]) == 0
    (run_dir,) = storage.list_runs(runs_dir())
    monkeypatch.setattr(pipeline, "default_stages", every_stage)
    capsys.readouterr()
    assert main(["resume", run_dir.name]) == 1
    err = capsys.readouterr().err
    assert "not downloaded or incomplete" in err and "cohr models download" in err
    stages = schemas.load(run_dir, Manifest).stages
    assert stages["ingest"].status == "done" and set(stages) == {"ingest"}  # nothing else ran


def test_eval_rerun_reaches_the_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from cohr import evals

    seen: dict[str, object] = {}

    def run_set(name: str, **kwargs: object) -> None:
        seen.update(kwargs)
        raise ValueError("stop here")

    monkeypatch.setattr(evals, "run_set", run_set)
    with pytest.raises(SystemExit, match="stop here"):
        main(["eval", "--item", "ami", "--rerun", "reduce", "--rerun", "render"])
    assert seen["rerun"] == ["reduce", "render"] and seen["items"] == ["ami"]


def test_index_rebuild_and_search(capsys: pytest.CaptureFixture[str]) -> None:
    from cohr import index
    from tests.test_index import BUDGET, make_run, sections

    make_run(runs_dir(), "weekly", ["hello", "the budget is set"], notes=sections([BUDGET]))
    assert main(["index", "rebuild"]) == 0
    assert "indexed 1 run(s)" in capsys.readouterr().err
    assert (runs_dir() / index.FILENAME).is_file()

    assert main(["search", "budget"]) == 0
    out = sorted(capsys.readouterr().out.splitlines(), key=lambda line: "transcript" in line)
    assert len(out) == 2
    assert "summary" in out[0] and "1:15" in out[0] and "**budget**" in out[0]
    assert "transcript" in out[1] and "0:30" in out[1]
    assert main(["search", "budget", "--kind", "transcript"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1
    assert main(["search", "nothing-like-this"]) == 1
