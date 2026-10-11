"""Stage runner (PLAN §3, §7 job lifecycle, §19 versioning).

A stage is skipped when its manifest entry is `done`, its outputs exist, and both hashes match:
- config_hash: the stage's config section(s), its prompt file and its FORMAT_VERSION;
- input_hash: chained from the run's inputs through every earlier stage's hashes,
  so re-running one stage re-runs everything after it.
"""

import contextlib
import hashlib
import json
import logging
import threading
import time
import traceback
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from cohr import schemas
from cohr.config import Config, prompt_hash
from cohr.schemas import Manifest, StageState

log = logging.getLogger("cohr")

HEARTBEAT_S = 5.0
STALE_AFTER_S = 60.0


class Cancelled(Exception):
    pass


class StageFailed(RuntimeError):
    def __init__(self, stage: str, cause: BaseException):
        super().__init__(f"stage {stage!r} failed: {cause}")
        self.stage = stage


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    status: Literal["skipped", "started", "progress", "done", "failed", "cancelled"]
    fraction: float = 0.0
    message: str = ""


ProgressFn = Callable[[ProgressEvent], None]


@dataclass
class StageContext:
    run_dir: Path
    config: Config
    manifest: Manifest
    stage: str
    cancel: threading.Event
    progress: ProgressFn
    log: logging.Logger = log

    def check_cancel(self) -> None:
        """Stages call this between chunks, frames and segments."""
        if self.cancel.is_set():
            raise Cancelled(self.stage)

    def report(self, fraction: float, message: str = "") -> None:
        self.check_cancel()
        self.progress(ProgressEvent(self.stage, "progress", fraction, message))


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[StageContext], None]
    outputs: tuple[str, ...]  # paths relative to the run dir
    format_version: int
    config_keys: tuple[str, ...] = ()  # dotted paths into Config, e.g. "vad", "models.vad"
    prompt: str | None = None  # src/cohr/prompts/<prompt>.md


def default_stages() -> list[Stage]:
    from cohr import ingest  # stage modules import this one for StageContext

    return [ingest.STAGE]


# --- hashing -------------------------------------------------------------------------------


def _sha(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _lookup(tree: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        tree = tree[part]
    return tree


def config_hash(stage: Stage, config: Config) -> str:
    dumped = config.model_dump(mode="json")
    return _sha(
        {
            "format_version": stage.format_version,
            "config": {k: _lookup(dumped, k) for k in stage.config_keys},
            "prompt": prompt_hash(stage.prompt) if stage.prompt else None,
        }
    )


def stage_hashes(
    stages: list[Stage], config: Config, manifest: Manifest
) -> dict[str, tuple[str, str]]:
    """{stage: (config_hash, input_hash)} for every stage, in order."""
    upstream = _sha(sorted((i.kind, i.sha256) for i in manifest.inputs))
    out = {}
    for stage in stages:
        ch = config_hash(stage, config)
        out[stage.name] = (ch, upstream)
        upstream = _sha([upstream, ch])
    return out


StaleReason = Literal["new", "not done", "format", "config", "input", "missing output"]


def stale_reason(
    stage: Stage, state: StageState | None, hashes: tuple[str, str], run_dir: Path
) -> StaleReason | None:
    """Why this stage must run, or None if it can be skipped."""
    if state is None:
        return "new"
    if state.status != "done":
        return "not done"
    if state.format_version != stage.format_version:
        return "format"
    if state.config_hash != hashes[0]:
        return "config"
    if state.input_hash != hashes[1]:
        return "input"
    if not all((run_dir / o).exists() for o in stage.outputs):
        return "missing output"
    return None


def plan(
    run_dir: Path, stages: list[Stage] | None = None, config: Config | None = None
) -> list[tuple[str, StaleReason]]:
    """Stages that would run, in order. Once one is stale, every later one is too."""
    stages = stages if stages is not None else default_stages()
    manifest = schemas.load(run_dir, Manifest)
    config = config or Config.model_validate(manifest.config)
    hashes = stage_hashes(stages, config, manifest)
    todo: list[tuple[str, StaleReason]] = []
    for stage in stages:
        reason = stale_reason(stage, manifest.stages.get(stage.name), hashes[stage.name], run_dir)
        if todo and reason is None:
            reason = "input"
        if reason is not None:
            todo.append((stage.name, reason))
    return todo


# --- manifest state shared with the heartbeat thread ----------------------------------------


@dataclass
class _RunState:
    run_dir: Path
    manifest: Manifest
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, stage: str, **changes: Any) -> None:
        with self.lock:
            current = self.manifest.stages.get(stage, StageState())
            self.manifest.stages[stage] = current.model_copy(update=changes)
            schemas.save(self.run_dir, self.manifest)


def _now() -> datetime:
    return datetime.now(UTC)


@contextlib.contextmanager
def _heartbeat(state: _RunState, stage: str, every: float) -> Generator[None]:
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(every):
            state.update(stage, heartbeat_at=_now())

    thread = threading.Thread(target=beat, name=f"heartbeat-{stage}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


@contextlib.contextmanager
def run_log(run_dir: Path, level: str = "INFO") -> Generator[logging.Logger]:
    """Attach runs/<id>/log.txt to the `cohr` logger for the duration of a run."""
    handler = logging.FileHandler(run_dir / "log.txt", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(level)
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(min(log.getEffectiveLevel(), logging.getLevelNamesMapping()[level]))
    try:
        yield log
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)
        handler.close()


# --- running -------------------------------------------------------------------------------


def run(
    run_dir: Path,
    *,
    stages: list[Stage] | None = None,
    config: Config | None = None,
    cancel: threading.Event | None = None,
    progress: ProgressFn | None = None,
    peak_memory: Callable[[], int | None] | None = None,
    heartbeat_s: float = HEARTBEAT_S,
) -> dict[str, Literal["skipped", "done"]]:
    """Run every stale stage in order. Raises Cancelled or StageFailed."""
    stages = stages if stages is not None else default_stages()
    cancel = cancel or threading.Event()
    progress = progress or (lambda _event: None)
    manifest = schemas.load(run_dir, Manifest)
    if config is None:
        config = Config.model_validate(manifest.config)
    else:
        manifest.config = config.model_dump(mode="json")
    state = _RunState(run_dir, manifest)
    hashes = stage_hashes(stages, config, manifest)
    results: dict[str, Literal["skipped", "done"]] = {}
    upstream_ran = False

    with run_log(run_dir, config.log_level):
        for stage in stages:
            ch, ih = hashes[stage.name]
            reason = stale_reason(stage, manifest.stages.get(stage.name), (ch, ih), run_dir)
            if reason is None and not upstream_ran:
                results[stage.name] = "skipped"
                progress(ProgressEvent(stage.name, "skipped", 1.0))
                continue
            if cancel.is_set():
                state.update(stage.name, status="cancelled")
                progress(ProgressEvent(stage.name, "cancelled"))
                raise Cancelled(stage.name)

            upstream_ran = True
            log.info("stage %s: start (%s)", stage.name, reason or "upstream re-ran")
            started = _now()
            state.update(
                stage.name,
                status="running",
                format_version=stage.format_version,
                config_hash=ch,
                input_hash=ih,
                started_at=started,
                heartbeat_at=started,
                finished_at=None,
                error=None,
            )
            progress(ProgressEvent(stage.name, "started"))
            ctx = StageContext(run_dir, config, manifest, stage.name, cancel, progress)
            t0 = time.perf_counter()
            try:
                with _heartbeat(state, stage.name, heartbeat_s):
                    stage.run(ctx)
            except Cancelled:
                log.info("stage %s: cancelled", stage.name)
                state.update(stage.name, status="cancelled", finished_at=_now())
                progress(ProgressEvent(stage.name, "cancelled"))
                raise
            except Exception as e:
                log.error("stage %s: failed\n%s", stage.name, traceback.format_exc())
                state.update(stage.name, status="failed", finished_at=_now(), error=str(e))
                progress(ProgressEvent(stage.name, "failed", message=str(e)))
                raise StageFailed(stage.name, e) from e
            peak = peak_memory() if peak_memory else None
            state.update(stage.name, status="done", finished_at=_now(), peak_mem_bytes=peak)
            log.info("stage %s: done in %.1f s", stage.name, time.perf_counter() - t0)
            results[stage.name] = "done"
            progress(ProgressEvent(stage.name, "done", 1.0))
    return results


def recover_interrupted(
    run_dir: Path, *, now: datetime | None = None, stale_after_s: float = STALE_AFTER_S
) -> list[str]:
    """On app start: `running` stages whose heartbeat is older than 60 s become `interrupted`."""
    manifest = schemas.load(run_dir, Manifest)
    now = now or _now()
    cutoff = now - timedelta(seconds=stale_after_s)
    changed = []
    for name, st in manifest.stages.items():
        beat = st.heartbeat_at or st.started_at
        if st.status == "running" and (beat is None or beat < cutoff):
            manifest.stages[name] = st.model_copy(update={"status": "interrupted"})
            changed.append(name)
    if changed:
        schemas.save(run_dir, manifest)
        log.warning("%s: interrupted stages %s", run_dir.name, changed)
    return changed
