"""Streamlit glue shared by app.py and app_pages/ (PLAN §7).

The pure helpers (run rows, statuses, cited times, audio clips) are unit tested; the widgets are
covered by tests/test_app.py through AppTest. Job logic lives in jobs.py, not here.
"""

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import soundfile as sf
import streamlit as st

from cohr import jobs, schemas, storage
from cohr.config import Config, runs_dir
from cohr.jobs import JobRunner, JobState
from cohr.mapper import clock
from cohr.models import PIPELINE_MODELS, ModelManager
from cohr.reduce import SECTIONS
from cohr.schemas import InputKind, Manifest, NotesDoc

STAGE_LABELS = {
    "ingest": "Ingest",
    "vad": "Speech detection",
    "transcribe": "Transcribe",
    "vision": "Describe slides",
    "fuse": "Fuse",
    "map": "Map",
    "reduce": "Reduce",
    "render": "Render",
}
# The four Cohere models, one chart color each on the Runs page (§7).
STAGE_MODELS = {
    "transcribe": "Transcribe",
    "vision": "Micro Vision",
    "map": "Tiny Aya",
    "reduce": "Command R7B",
}
RUN_KEY = "run_id"  # the sidebar's current run, read by Notes (and Ask, 5b)
POLL_S = 1.0
MEM_GB = 1e9  # memory in decimal GB, like the evals and PLAN §4 (26.8 GB limit)
GPU_WARN_FRACTION = 0.9
FOOTER = (
    "Unofficial; not affiliated with Cohere. Default models are CC-BY-NC: non-commercial use only."
)

# Tests swap this for a fake ModelManager (no MLX, no cache); looked up at call time.
make_models: Callable[[], ModelManager] = lambda: ModelManager(Config())  # noqa: E731
make_runner: Callable[[ModelManager], JobRunner] = JobRunner


# --- shared resources (one per process) ----------------------------------------------------


@st.cache_resource
def get_models() -> ModelManager:
    return make_models()


@st.cache_resource
def get_runner() -> JobRunner:
    return make_runner(get_models())


@st.cache_resource
def startup(runs: str) -> dict[str, list[str]]:
    """Once per process and runs dir: stale `running` stages become `interrupted` (§7)."""
    return jobs.recover_all(Path(runs))


@st.cache_data(ttl=60, show_spinner=False)
def missing_models() -> list[str]:
    return list(get_models().missing(PIPELINE_MODELS))


# --- pure helpers --------------------------------------------------------------------------


def gb(n: int | float) -> str:
    return f"{n / MEM_GB:.1f} GB"


def run_status(manifest: Manifest) -> str:
    """One word for a run: the most urgent stage status, else done / incomplete / new."""
    statuses = {s.status for s in manifest.stages.values()}
    for status in ("running", "failed", "interrupted", "cancelled"):
        if status in statuses:
            return status
    if not statuses:
        return "new"
    render = manifest.stages.get("render")
    return "done" if render is not None and render.status == "done" else "incomplete"


def resumable(manifest: Manifest) -> bool:
    return run_status(manifest) in ("failed", "interrupted", "cancelled", "incomplete", "new")


def run_rows(runs: Path) -> list[dict[str, Any]]:
    """The Runs table, newest first. Unreadable runs are listed, not hidden."""
    rows = []
    for run_dir in storage.list_runs(runs):
        try:
            m = schemas.load(run_dir, Manifest)
        except Exception:
            rows.append({"run": run_dir.name, "status": "unreadable"})
            continue
        rows.append(
            {
                "run": run_dir.name,
                "status": run_status(m) + (" · pruned" if m.pruned else ""),
                "created": m.created_at,
                "size_mb": storage.run_size(run_dir) / storage.MB,
                "languages": "{} → {}".format(
                    m.config.get("source_language"), m.config.get("target_language")
                ),
            }
        )
    return rows


def memory_rows(manifest: Manifest) -> list[dict[str, Any]]:
    """Peak memory of the model stages, in pipeline order (§4: shown after each stage)."""
    return [
        {"stage": STAGE_LABELS[name], "model": model, "peak_gb": st_.peak_mem_bytes / MEM_GB}
        for name, model in STAGE_MODELS.items()
        if (st_ := manifest.stages.get(name)) and st_.peak_mem_bytes
    ]


def stage_rows(manifest: Manifest) -> list[dict[str, Any]]:
    rows = []
    for name, s in manifest.stages.items():
        took = (
            (s.finished_at - s.started_at).total_seconds()
            if s.finished_at and s.started_at
            else None
        )
        rows.append(
            {
                "stage": STAGE_LABELS.get(name, name),
                "status": s.status,
                "seconds": took,
                "peak_gb": s.peak_mem_bytes / MEM_GB if s.peak_mem_bytes else None,
                "error": s.error,
            }
        )
    return rows


def all_claims(notes: NotesDoc) -> list[Any]:
    return [c for name in SECTIONS for c in getattr(notes.sections, name)]


def uncited_count(notes: NotesDoc) -> int:
    return sum(1 for c in all_claims(notes) if not c.citations)


def cited_times(notes: NotesDoc, seg_t0: dict[int, float]) -> dict[int, list[float]]:
    """{seg_id: sorted refined jump times} (§5.6); a missing `t` falls back to the segment start."""
    out: dict[int, set[float]] = {}
    for claim in all_claims(notes):
        for c in claim.citations:
            if c.kind == "seg" and c.id in seg_t0:
                out.setdefault(c.id, set()).add(c.t if c.t is not None else seg_t0[c.id])
    return {k: sorted(v) for k, v in out.items()}


def media_kind(name: str) -> InputKind:
    """Upload/local path → manifest input kind. ffmpeg handles the formats themselves."""
    video = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
    return "video" if Path(name).suffix.lower() in video else "audio"


@st.cache_data(max_entries=64, show_spinner=False)
def audio_clip(path: str, mtime: float, t0: float, t1: float) -> bytes:
    """One segment of audio.wav as WAV bytes. st.audio with a path reads and hashes the whole
    file on every call (~115 MB/h); a cached clip per segment keeps reruns cheap. `mtime` is
    only part of the cache key."""
    del mtime
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        f.seek(min(int(t0 * sr), f.frames))
        data = f.read(max(0, int((t1 - t0) * sr)), dtype="int16")
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --- sidebar ---------------------------------------------------------------------------------


def current_run() -> Path | None:
    run_id = st.session_state.get(RUN_KEY)
    if not run_id:
        return None
    path = runs_dir() / run_id
    return path if storage.is_run(path) else None


def run_picker() -> None:
    ids = [p.name for p in storage.list_runs(runs_dir())]
    if st.session_state.get(RUN_KEY) not in ids:
        st.session_state[RUN_KEY] = ids[0] if ids else None
    st.selectbox(
        "Current run",
        ids,
        key=RUN_KEY,
        placeholder="No runs yet",
        help="Notes shows this run.",
    )


def _status_body() -> None:
    models = get_models()
    loaded = [m for m in models.loaded() if m in PIPELINE_MODELS]
    active, limit = models.memory.active(), models.memory.limit()
    parts = [f"Models loaded: {', '.join(loaded)}" if loaded else "No models loaded"]
    if active is not None and limit:
        parts.append(f"{gb(active)} of {gb(limit)}")
    over = active is not None and bool(limit) and active >= GPU_WARN_FRACTION * limit
    text = " · ".join(parts)
    st.caption(f":red[:material/warning: {text}, near the GPU limit]" if over else text)
    if missing := missing_models():
        st.caption(f":red[Models not downloaded: {', '.join(missing)}]")
    job = get_runner().current
    if job is not None and (state := job.snapshot()).active:
        stage = STAGE_LABELS.get(state.current or "", state.current or "starting")
        st.caption(f":material/progress_activity: {state.run_id}: {stage}")
        st.session_state["_status_polling"] = True
    elif st.session_state.pop("_status_polling", False):
        st.rerun()  # the job ended: one full rerun drops the polling


def status_line() -> None:
    """Model and memory status; polls while a job runs, so it follows the stages."""
    every = POLL_S if get_runner().busy() else None
    st.fragment(run_every=every)(_status_body)()


def sidebar() -> None:
    with st.sidebar:
        run_picker()
        status_line()
        st.caption(FOOTER)


# --- job progress ------------------------------------------------------------------------------


StatusState = Literal["running", "complete", "error"]


def _status_args(view: jobs.StageView) -> tuple[StatusState, str]:
    if view.status == "failed":
        return "error", "failed"
    if view.status == "cancelled":
        return "error", "cancelled"
    if view.status in ("started", "progress"):
        return "running", f"{view.fraction:.0%}"
    return "complete", "reused" if view.status == "skipped" else "done"


def _job_body(job: jobs.Job, show_result: bool) -> None:
    state: JobState = job.snapshot()
    seen = f"_job_active_{state.run_id}"
    if not state.active and not show_result:
        if st.session_state.pop(seen, False):
            st.rerun()  # once, so the page stops polling and refreshes what the job changed
        return
    st.markdown(f"**{state.run_id}**")
    pending = []
    for name, view in state.stages.items():
        label = STAGE_LABELS.get(name, name)
        if view.status == "pending":
            pending.append(label)
            continue
        status, note = _status_args(view)
        running = status == "running"
        with st.status(f"{label} · {note}", state=status, expanded=running):
            if running:
                st.progress(view.fraction, text=view.message or None)
            elif view.message:
                st.caption(view.message)
    if pending and state.active:
        st.caption("Next: " + ", ".join(pending))

    if state.active:
        st.session_state[seen] = True
        st.button("Cancel", icon=":material/stop_circle:", on_click=job.cancel, key="job_cancel")
        return
    if state.status == "done":
        st.success("Notes are ready.", icon=":material/check_circle:")
    elif state.status == "cancelled":
        st.warning("Cancelled. Finished stages are kept; resume from Runs.", icon=":material/info:")
    else:
        st.error(state.error or "The job failed.", icon=":material/error:")
    if st.session_state.pop(seen, False):
        st.rerun()  # once, so the page stops polling and refreshes what the job changed


def job_panel(run_id: str | None = None, *, show_result: bool = False) -> None:
    """The current job's stages (one st.status each) and Cancel; polls while it runs. A
    finished job stays visible only with `show_result` (the New run page)."""
    job = get_runner().current
    if job is None or (run_id is not None and job.run_id != run_id):
        return
    active = job.snapshot().active
    if not active and not show_result:
        return
    st.fragment(run_every=POLL_S if active else None)(_job_body)(job, show_result)


def fmt_range(t0: float, t1: float) -> str:
    return f"{clock(t0)}–{clock(t1)}"
