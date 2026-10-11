"""Background jobs for the UI (PLAN §7): no Streamlit here, so it's unit tested without AppTest.

`JobRunner.start` runs `pipeline.run` in a thread. Progress events go through a queue; the UI
calls `Job.snapshot()` on each poll, which drains the queue into a `JobState`. One job at a time:
`start` refuses while a job runs, and every model use is serialized by the ModelManager lock.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from cohr import pipeline, schemas, storage
from cohr.config import Config
from cohr.models import ModelsMissing
from cohr.pipeline import Cancelled, ProgressEvent, Stage, StageFailed
from cohr.schemas import Manifest

if TYPE_CHECKING:
    from cohr.models import ModelManager

log = logging.getLogger("cohr")

JobStatus = Literal["running", "done", "failed", "cancelled"]
StageStatus = Literal["pending", "skipped", "started", "progress", "done", "failed", "cancelled"]


class JobBusy(RuntimeError):
    def __init__(self, run_id: str):
        super().__init__(f"a job is already running ({run_id}); wait for it or cancel it")
        self.run_id = run_id


@dataclass(frozen=True)
class StageView:
    status: StageStatus = "pending"
    fraction: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class JobState:
    """What the UI reads. Immutable: each snapshot is a new object."""

    run_id: str
    status: JobStatus
    stages: dict[str, StageView]
    started_at: float
    finished_at: float | None = None
    current: str | None = None  # the stage running now, or the one that stopped the job
    error: str | None = None

    @property
    def active(self) -> bool:
        return self.status == "running"


@dataclass(frozen=True)
class _Finished:
    status: JobStatus
    error: str | None = None


@dataclass
class Job:
    run_dir: Path
    stage_names: list[str]
    force: tuple[str, ...] = ()
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _events: "queue.Queue[ProgressEvent | _Finished]" = field(default_factory=queue.Queue)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: threading.Thread | None = None
    _state: JobState | None = None

    def __post_init__(self) -> None:
        self._state = JobState(
            run_id=self.run_dir.name,
            status="running",
            stages={name: StageView() for name in self.stage_names},
            started_at=time.time(),
        )

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def progress(self, event: ProgressEvent) -> None:
        """pipeline.run's progress callback (runs on the job thread)."""
        self._events.put(event)

    def finish(self, status: JobStatus, error: str | None = None) -> None:
        """Last event of a job (job thread); queued so it lands after every progress event."""
        self._events.put(_Finished(status, error))

    def cancel(self) -> None:
        """Stages check this between chunks, frames and segments (§7)."""
        self.cancel_event.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Join the job thread; True when it has finished."""
        if self._thread is not None:
            self._thread.join(timeout)
        return self.snapshot().status != "running"

    def snapshot(self) -> JobState:
        with self._lock:
            state = self._state
            assert state is not None
            while True:
                try:
                    item = self._events.get_nowait()
                except queue.Empty:
                    break
                state = _apply(state, item)
            self._state = state
            return state


def _apply(state: JobState, item: ProgressEvent | _Finished) -> JobState:
    if isinstance(item, _Finished):
        return replace(state, status=item.status, error=item.error, finished_at=time.time())
    stages = dict(state.stages)
    old = stages.get(item.stage, StageView())
    fraction = 1.0 if item.status in ("skipped", "done") else item.fraction
    stages[item.stage] = StageView(item.status, fraction, item.message or old.message)
    current = item.stage if item.status != "skipped" else state.current
    return replace(state, stages=stages, current=current)


RunFn = Callable[..., object]


class JobRunner:
    """One per process (the UI caches it next to the ModelManager)."""

    def __init__(
        self,
        models: "ModelManager",
        *,
        stages: Callable[[], list[Stage]] = pipeline.default_stages,
        run: RunFn = pipeline.run,
    ):
        self.models = models
        self._stages = stages
        self._run = run
        self._lock = threading.Lock()
        self._job: Job | None = None

    @property
    def current(self) -> Job | None:
        """The latest job, finished or not (the UI shows its result until the next one)."""
        return self._job

    def busy(self) -> bool:
        job = self._job
        return job is not None and job.snapshot().active

    def start(self, run_dir: Path, *, force: Iterable[str] = ()) -> Job:
        """Run a run's stale stages (plus `force`d ones) in the background. Raises JobBusy."""
        with self._lock:
            if self._job is not None and self._job.snapshot().active:
                raise JobBusy(self._job.run_id)
            stages = self._stages()
            job = Job(run_dir, [s.name for s in stages], tuple(force))
            config = Config.model_validate(schemas.load(run_dir, Manifest).config)
            thread = threading.Thread(
                target=self._work, args=(job, stages, config), name=f"job-{job.run_id}", daemon=True
            )
            job._thread = thread
            self._job = job
            thread.start()
            return job

    def _work(self, job: Job, stages: list[Stage], config: Config) -> None:
        try:
            self.models.set_config(config)
            self._run(
                job.run_dir,
                stages=stages,
                cancel=job.cancel_event,
                progress=job.progress,
                models=self.models,
                force=job.force,
            )
        except Cancelled:
            job.finish("cancelled")
        except StageFailed as e:
            job.finish("failed", f"{e}. See log.txt in the run folder.")
        except ModelsMissing as e:
            job.finish("failed", str(e))
        except Exception as e:  # anything else still ends the job instead of hanging the UI
            log.exception("job %s failed", job.run_id)
            job.finish("failed", f"{type(e).__name__}: {e}")
        else:
            job.finish("done")


def recover_all(runs_dir: Path) -> dict[str, list[str]]:
    """On app start (§7): stale `running` stages in every run become `interrupted`."""
    out = {}
    for run_dir in storage.list_runs(runs_dir):
        try:
            if changed := pipeline.recover_interrupted(run_dir):
                out[run_dir.name] = changed
        except Exception:  # one broken manifest must not stop the app from starting
            log.exception("could not check %s for interrupted stages", run_dir.name)
    return out
