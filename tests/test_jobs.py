"""PLAN §7 job layer: background run, progress queue, cancel, one job at a time, recovery."""

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cohr import atomic, jobs, schemas, storage
from cohr.config import Config
from cohr.jobs import JobBusy, JobRunner
from cohr.models import ModelsMissing
from cohr.pipeline import Stage, StageContext
from cohr.schemas import Manifest, StageState
from cohr.storage import Source
from tests.conftest import fake_manager


class Gate:
    """Fake stages; stage `blocking` waits on `release` so a test can act mid-run."""

    def __init__(self, blocking: str | None = None, fail: str | None = None):
        self.blocking = blocking
        self.fail = fail
        self.entered = threading.Event()
        self.release = threading.Event()
        self.ran: list[str] = []

    def stage(self, name: str) -> Stage:
        def run(ctx: StageContext) -> None:
            self.ran.append(name)
            ctx.report(0.5, "half")
            if name == self.blocking:
                self.entered.set()
                while not self.release.wait(0.01):
                    ctx.check_cancel()
            if name == self.fail:
                raise RuntimeError("boom")
            atomic.write_text(ctx.run_dir / f"{name}.txt", name)

        return Stage(name=name, run=run, outputs=(f"{name}.txt",), format_version=1)

    def stages(self) -> list[Stage]:
        return [self.stage(n) for n in ("a", "b", "c")]


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    src = tmp_path / "in.m4a"
    src.write_bytes(b"audio")
    return storage.create_run(tmp_path / "runs", [Source(src, "audio")], Config())


def runner(tmp_path: Path, gate: Gate) -> JobRunner:
    return JobRunner(fake_manager(Config(), tmp_path), stages=gate.stages)


def test_job_runs_in_background_and_reports_every_stage(tmp_path: Path, run_dir: Path) -> None:
    gate = Gate()
    job = runner(tmp_path, gate).start(run_dir)
    assert job.wait(5)
    state = job.snapshot()
    assert state.status == "done" and state.error is None and state.finished_at
    assert list(state.stages) == ["a", "b", "c"]
    assert all(v.status == "done" and v.fraction == 1.0 for v in state.stages.values())
    assert state.stages["a"].message == "half"  # kept after the stage finishes
    assert gate.ran == ["a", "b", "c"]


def test_snapshot_shows_progress_mid_run(tmp_path: Path, run_dir: Path) -> None:
    gate = Gate(blocking="b")
    job = runner(tmp_path, gate).start(run_dir)
    assert gate.entered.wait(5)
    state = job.snapshot()
    assert state.active and state.current == "b"
    assert state.stages["a"].status == "done"
    assert state.stages["b"] == jobs.StageView("progress", 0.5, "half")
    assert state.stages["c"].status == "pending"
    gate.release.set()
    assert job.wait(5)


def test_one_job_at_a_time(tmp_path: Path, run_dir: Path) -> None:
    gate = Gate(blocking="a")
    r = runner(tmp_path, gate)
    r.start(run_dir)
    assert gate.entered.wait(5)
    assert r.busy()
    with pytest.raises(JobBusy, match=run_dir.name):
        r.start(run_dir)
    gate.release.set()
    assert r.current is not None and r.current.wait(5)
    assert not r.busy()
    assert r.start(run_dir).wait(5)  # free again; everything is skipped now
    assert r.current.snapshot().stages["a"].status == "skipped"


def test_cancel_marks_the_stage_and_keeps_earlier_ones(tmp_path: Path, run_dir: Path) -> None:
    gate = Gate(blocking="b")
    job = runner(tmp_path, gate).start(run_dir)
    assert gate.entered.wait(5)
    job.cancel()
    assert job.wait(5)
    state = job.snapshot()
    assert state.status == "cancelled" and state.current == "b"
    manifest = schemas.load(run_dir, Manifest)
    assert manifest.stages["a"].status == "done"
    assert manifest.stages["b"].status == "cancelled"
    assert "c" not in manifest.stages


def test_failure_is_reported_with_the_log_hint(tmp_path: Path, run_dir: Path) -> None:
    job = runner(tmp_path, Gate(fail="b")).start(run_dir)
    assert job.wait(5)
    state = job.snapshot()
    assert state.status == "failed" and state.current == "b"
    assert state.error is not None and "boom" in state.error and "log.txt" in state.error
    assert state.stages["b"].status == "failed"


def test_missing_models_fail_the_job_before_any_stage(tmp_path: Path, run_dir: Path) -> None:
    def run(*_args: object, **_kwargs: object) -> None:
        raise ModelsMissing({"reduce": Config().models["reduce"]})

    r = JobRunner(fake_manager(Config(), tmp_path), stages=Gate().stages, run=run)
    job = r.start(run_dir)
    assert job.wait(5)
    state = job.snapshot()
    assert state.status == "failed" and "cohr models download" in (state.error or "")
    assert all(v.status == "pending" for v in state.stages.values())


def test_unexpected_errors_still_end_the_job(tmp_path: Path, run_dir: Path) -> None:
    def run(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk gone")

    job = JobRunner(fake_manager(Config(), tmp_path), stages=Gate().stages, run=run).start(run_dir)
    assert job.wait(5)
    assert job.snapshot().error == "OSError: disk gone"


def test_force_reruns_a_finished_stage_and_later(tmp_path: Path, run_dir: Path) -> None:
    gate = Gate()
    r = runner(tmp_path, gate)
    assert r.start(run_dir).wait(5)
    gate.ran.clear()
    job = r.start(run_dir, force=["b"])
    assert job.wait(5)
    assert gate.ran == ["b", "c"]
    assert job.snapshot().stages["a"].status == "skipped"


def test_job_switches_the_manager_to_the_run_config(tmp_path: Path) -> None:
    src = tmp_path / "in.m4a"
    src.write_bytes(b"audio")
    run_dir = storage.create_run(
        tmp_path / "runs", [Source(src, "audio")], Config(memory_policy="sequential")
    )
    r = runner(tmp_path, Gate())
    assert r.start(run_dir).wait(5)
    assert r.models.config.memory_policy == "sequential"


def test_recover_all_marks_stale_running_stages(run_dir: Path) -> None:
    manifest = schemas.load(run_dir, Manifest)
    old = datetime.now(UTC) - timedelta(minutes=5)
    manifest.stages["asr"] = StageState(status="running", started_at=old, heartbeat_at=old)
    schemas.save(run_dir, manifest)
    (run_dir.parent / "not-a-run").mkdir()
    assert jobs.recover_all(run_dir.parent) == {run_dir.name: ["asr"]}
    assert schemas.load(run_dir, Manifest).stages["asr"].status == "interrupted"
    assert jobs.recover_all(run_dir.parent) == {}


def test_recover_all_survives_a_broken_manifest(run_dir: Path) -> None:
    (run_dir / "manifest.json").write_text("{not json")
    assert jobs.recover_all(run_dir.parent) == {}
