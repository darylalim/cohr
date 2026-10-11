"""PLAN §3, §7, §19: skip logic, hash chaining, atomic writes, cancel, failure, recovery."""

import dataclasses
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cohr import atomic, pipeline, schemas, storage
from cohr.config import Config, FuseConfig, ReduceConfig, VadConfig
from cohr.pipeline import Cancelled, ProgressEvent, Stage, StageContext, StageFailed
from cohr.schemas import Manifest, StageState
from cohr.storage import Source


class Recorder:
    """Fake stages that write one text file each and record that they ran."""

    def __init__(self) -> None:
        self.ran: list[str] = []
        self.hooks: dict[str, object] = {}

    def stage(
        self, name: str, keys: tuple[str, ...] = (), version: int = 1, prompt: str | None = None
    ) -> Stage:
        def run(ctx: StageContext) -> None:
            self.ran.append(name)
            hook = self.hooks.get(name)
            if callable(hook):
                hook(ctx)
            atomic.write_text(ctx.run_dir / f"{name}.txt", name)

        return Stage(
            name=name,
            run=run,
            outputs=(f"{name}.txt",),
            format_version=version,
            config_keys=keys,
            prompts=(prompt,) if prompt else (),
        )


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    src = tmp_path / "in.m4a"
    src.write_bytes(b"audio")
    return storage.create_run(tmp_path / "runs", [Source(src, "audio")], Config())


def three(rec: Recorder, vad_version: int = 1) -> list[Stage]:
    return [
        rec.stage("a", ("ingest",)),
        rec.stage("b", ("vad",), vad_version),
        rec.stage("c", ("fuse",)),
    ]


def test_second_run_skips_everything(run_dir: Path) -> None:
    rec = Recorder()
    assert pipeline.run(run_dir, stages=three(rec)) == {"a": "done", "b": "done", "c": "done"}
    assert pipeline.run(run_dir, stages=three(rec)) == {
        "a": "skipped",
        "b": "skipped",
        "c": "skipped",
    }
    assert rec.ran == ["a", "b", "c"]
    manifest = schemas.load(run_dir, Manifest)
    assert all(s.status == "done" for s in manifest.stages.values())
    assert pipeline.plan(run_dir, three(Recorder())) == []


def test_config_change_reruns_that_stage_and_later(run_dir: Path) -> None:
    pipeline.run(run_dir, stages=three(Recorder()))
    rec = Recorder()
    cfg = Config(vad=VadConfig(threshold=0.6))
    assert pipeline.plan(run_dir, three(rec), cfg) == [("b", "config"), ("c", "input")]
    pipeline.run(run_dir, stages=three(rec), config=cfg)
    assert rec.ran == ["b", "c"]
    # The new config is recorded, so the next plain run skips everything.
    assert Config.model_validate(schemas.load(run_dir, Manifest).config).vad.threshold == 0.6
    assert pipeline.plan(run_dir, three(Recorder())) == []


def test_unrelated_config_change_skips(run_dir: Path) -> None:
    pipeline.run(run_dir, stages=three(Recorder()))
    rec = Recorder()
    pipeline.run(
        run_dir, stages=three(rec), config=Config(reduce=ReduceConfig(max_input_tokens=24_000))
    )
    assert rec.ran == []


def test_last_stage_config_change_reruns_only_it(run_dir: Path) -> None:
    pipeline.run(run_dir, stages=three(Recorder()))
    rec = Recorder()
    pipeline.run(run_dir, stages=three(rec), config=Config(fuse=FuseConfig(window_s=600)))
    assert rec.ran == ["c"]


def test_prompt_edit_reruns_its_stage(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def stages(rec: Recorder) -> list[Stage]:
        return [rec.stage("a"), rec.stage("m", prompt="map"), rec.stage("r", prompt="reduce")]

    pipeline.run(run_dir, stages=stages(Recorder()))
    real = pipeline.prompt_hash
    monkeypatch.setattr(
        pipeline, "prompt_hash", lambda name: "edited" if name == "map" else real(name)
    )
    rec = Recorder()
    pipeline.run(run_dir, stages=stages(rec))
    assert rec.ran == ["m", "r"]


def test_older_format_reprocesses_from_that_stage(run_dir: Path) -> None:
    """§19: a run made by an older version re-processes from the first outdated stage."""
    pipeline.run(run_dir, stages=three(Recorder(), vad_version=1))
    rec = Recorder()
    assert pipeline.plan(run_dir, three(rec, vad_version=2)) == [("b", "format"), ("c", "input")]
    pipeline.run(run_dir, stages=three(rec, vad_version=2))
    assert rec.ran == ["b", "c"]
    assert schemas.load(run_dir, Manifest).stages["b"].format_version == 2


def test_missing_output_reruns(run_dir: Path) -> None:
    pipeline.run(run_dir, stages=three(Recorder()))
    (run_dir / "b.txt").unlink()
    rec = Recorder()
    pipeline.run(run_dir, stages=three(rec))
    assert rec.ran == ["b", "c"]


def test_changed_input_reruns_everything(run_dir: Path) -> None:
    pipeline.run(run_dir, stages=three(Recorder()))
    manifest = schemas.load(run_dir, Manifest)
    manifest.inputs[0] = manifest.inputs[0].model_copy(update={"sha256": "f" * 64})
    schemas.save(run_dir, manifest)
    rec = Recorder()
    pipeline.run(run_dir, stages=three(rec))
    assert rec.ran == ["a", "b", "c"]


def test_cancel_marks_stage_and_keeps_earlier_ones(run_dir: Path) -> None:
    rec = Recorder()
    cancel = threading.Event()

    def cancel_midway(ctx: StageContext) -> None:
        cancel.set()
        ctx.report(0.5, "halfway")  # checks the flag

    rec.hooks["b"] = cancel_midway
    with pytest.raises(Cancelled):
        pipeline.run(run_dir, stages=three(rec), cancel=cancel)
    manifest = schemas.load(run_dir, Manifest)
    assert manifest.stages["a"].status == "done"
    assert manifest.stages["b"].status == "cancelled"
    assert "c" not in manifest.stages
    assert not (run_dir / "b.txt").exists(), "cancelled stage left output behind"

    rec2 = Recorder()
    assert pipeline.run(run_dir, stages=three(rec2))["a"] == "skipped"
    assert rec2.ran == ["b", "c"]


def test_failure_is_recorded_and_logged(run_dir: Path) -> None:
    rec = Recorder()

    def boom(_ctx: StageContext) -> None:
        raise ValueError("bad chunk 7")

    rec.hooks["b"] = boom
    with pytest.raises(StageFailed, match="bad chunk 7"):
        pipeline.run(run_dir, stages=three(rec))
    state = schemas.load(run_dir, Manifest).stages["b"]
    assert state.status == "failed"
    assert state.error == "bad chunk 7"
    log = (run_dir / "log.txt").read_text()
    assert "Traceback" in log and "bad chunk 7" in log
    assert "stage a: done" in log


def test_progress_events(run_dir: Path) -> None:
    events: list[ProgressEvent] = []
    rec = Recorder()
    rec.hooks["a"] = lambda ctx: ctx.report(0.5, "half")
    pipeline.run(run_dir, stages=three(rec)[:1], progress=events.append)
    assert [(e.stage, e.status) for e in events] == [
        ("a", "started"),
        ("a", "progress"),
        ("a", "done"),
    ]
    events.clear()
    pipeline.run(run_dir, stages=three(Recorder())[:1], progress=events.append)
    assert [e.status for e in events] == ["skipped"]


def test_heartbeat_and_peak_memory(run_dir: Path) -> None:
    rec = Recorder()
    rec.hooks["a"] = lambda _ctx: time.sleep(0.2)
    pipeline.run(run_dir, stages=three(rec)[:1], heartbeat_s=0.02, peak_memory=lambda: 1234)
    state = schemas.load(run_dir, Manifest).stages["a"]
    assert state.started_at and state.heartbeat_at and state.heartbeat_at > state.started_at
    assert state.peak_mem_bytes == 1234


def test_recover_interrupted(run_dir: Path) -> None:
    now = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
    manifest = schemas.load(run_dir, Manifest)
    manifest.stages = {
        "a": StageState(status="done"),
        "b": StageState(status="running", heartbeat_at=now - timedelta(seconds=120)),
        "c": StageState(status="running", heartbeat_at=now - timedelta(seconds=10)),
    }
    schemas.save(run_dir, manifest)
    assert pipeline.recover_interrupted(run_dir, now=now) == ["b"]
    stages = schemas.load(run_dir, Manifest).stages
    assert (stages["a"].status, stages["b"].status, stages["c"].status) == (
        "done",
        "interrupted",
        "running",
    )


def test_interrupted_run_resumes_from_first_incomplete(run_dir: Path) -> None:
    pipeline.run(run_dir, stages=three(Recorder()))
    manifest = schemas.load(run_dir, Manifest)
    manifest.stages["b"] = manifest.stages["b"].model_copy(update={"status": "interrupted"})
    schemas.save(run_dir, manifest)
    rec = Recorder()
    pipeline.run(run_dir, stages=three(rec))
    assert rec.ran == ["b", "c"]


def test_atomic_path_keeps_old_file_on_error(tmp_path: Path) -> None:
    dest = tmp_path / "out.json"
    dest.write_text("old")
    with pytest.raises(RuntimeError), atomic.atomic_path(dest) as tmp:
        tmp.write_text("half")
        raise RuntimeError
    assert dest.read_text() == "old"
    assert list(tmp_path.iterdir()) == [dest]


def test_atomic_dir_replaces_whole_directory(tmp_path: Path) -> None:
    dest = tmp_path / "frames"
    dest.mkdir()
    (dest / "stale.png").write_text("x")
    with atomic.atomic_dir(dest) as tmp:
        (tmp / "000.png").write_text("new")
    assert [p.name for p in dest.iterdir()] == ["000.png"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["frames"]

    with pytest.raises(RuntimeError), atomic.atomic_dir(dest) as tmp:
        (tmp / "001.png").write_text("half")
        raise RuntimeError
    assert [p.name for p in dest.iterdir()] == ["000.png"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["frames"]


def test_atomic_outputs_follow_umask(tmp_path: Path) -> None:
    atomic.write_text(tmp_path / "a.json", "{}")
    with atomic.atomic_dir(tmp_path / "frames"):
        pass
    plain = tmp_path / "plain"
    plain.write_text("")
    plain_dir = tmp_path / "plain_dir"
    plain_dir.mkdir()
    assert (tmp_path / "a.json").stat().st_mode == plain.stat().st_mode
    assert (tmp_path / "frames").stat().st_mode == plain_dir.stat().st_mode


def test_preflight_requires_models_only_for_stages_that_will_run(
    run_dir: Path, tmp_path: Path
) -> None:
    from cohr.models import ModelManager, ModelsMissing

    rec = Recorder()
    a = rec.stage("a")
    b = rec.stage("b")
    b = dataclasses.replace(b, models=("vision",))

    def resolve(_spec: object) -> Path:
        raise FileNotFoundError

    missing = ModelManager(Config(), loaders={}, resolve=resolve)
    with pytest.raises(ModelsMissing):
        pipeline.run(run_dir, stages=[a, b], models=missing)
    assert rec.ran == []  # nothing starts when a model is missing

    present = ModelManager(Config(), loaders={}, resolve=lambda _s: tmp_path)
    pipeline.run(run_dir, stages=[a, b], models=present)
    assert rec.ran == ["a", "b"]
    # Everything is done now: no stage needs a model, so a missing one doesn't matter.
    assert pipeline.run(run_dir, stages=[a, b], models=missing) == {"a": "skipped", "b": "skipped"}
    with pytest.raises(RuntimeError, match="need a ModelManager"):
        pipeline.run(run_dir, stages=[a, dataclasses.replace(b, format_version=2)])


def test_default_stages_order() -> None:
    names = [s.name for s in pipeline.default_stages()]
    assert names == [
        "ingest",
        "vad",
        "transcribe",
        "vision",
        "fuse",
        "map",
        "reduce",
        "render",
    ]
