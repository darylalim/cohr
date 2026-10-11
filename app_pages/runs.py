"""Runs (PLAN §7, §12): table, Resume, re-run a stage, Prune and Delete, peak memory chart."""

from pathlib import Path

import streamlit as st

from cohr import pipeline, schemas, storage, ui
from cohr.config import runs_dir
from cohr.jobs import JobBusy
from cohr.schemas import Manifest

st.set_page_config(page_title="Runs · cohr", layout="wide")

ERROR_KEY = "runs_error"
STAGES = [s.name for s in pipeline.default_stages()]


def _start(run_dir: Path, force: list[str] | None = None) -> None:
    st.session_state.pop(ERROR_KEY, None)
    try:
        ui.get_runner().start(run_dir, force=force or [])
    except JobBusy as e:
        st.session_state[ERROR_KEY] = str(e)


def _open_notes(run_id: str) -> None:
    st.session_state[ui.RUN_KEY] = run_id


@st.dialog("Prune media?")
def confirm_prune(run_dir: Path) -> None:
    st.write(
        f"Removes the audio, slide images and source files of **{run_dir.name}** "
        f"({storage.run_size(run_dir) / storage.MB:.0f} MB now). Notes, citations and "
        "Re-reduce keep working; the audio player and slide images won't. Linked originals "
        "are not touched."
    )
    if st.button("Prune media", type="primary", icon=":material/content_cut:"):
        freed = storage.prune_media(run_dir, runs_dir())
        st.session_state["runs_flash"] = f"Freed {freed / storage.MB:.0f} MB."
        st.rerun()


@st.dialog("Delete run?")
def confirm_delete(run_dir: Path) -> None:
    st.write(f"Deletes **{run_dir.name}** and everything in it. This can't be undone.")
    if st.button("Delete", type="primary", icon=":material/delete:"):
        storage.delete_run(run_dir, runs_dir())
        st.session_state["runs_flash"] = f"Deleted {run_dir.name}."
        st.rerun()


st.title("Runs")
if flash := st.session_state.pop("runs_flash", None):
    st.success(flash, icon=":material/check_circle:")
if error := st.session_state.get(ERROR_KEY):
    st.error(error, icon=":material/error:")

rows = ui.run_rows(runs_dir())
if not rows:
    st.info("No runs yet. Start one on the New run page.", icon=":material/info:")
    st.stop()

event = st.dataframe(
    rows,
    hide_index=True,
    key="runs_table",
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "run": st.column_config.TextColumn("Run"),
        "status": st.column_config.TextColumn("Status"),
        "created": st.column_config.DatetimeColumn("Created", format="YYYY-MM-DD HH:mm"),
        "size_mb": st.column_config.NumberColumn("Size", format="%.0f MB"),
        "languages": st.column_config.TextColumn("Languages"),
    },
    alt="Previous runs with status, creation date, size on disk and languages",
)
selected = event.selection.rows
ui.job_panel()
if not selected:
    st.caption("Select a run to see its stages and actions.")
    st.stop()

row = rows[selected[0]]
run_dir = runs_dir() / row["run"]
if not storage.is_run(run_dir):
    st.stop()
if row["status"] == "unreadable":
    st.error(f"{run_dir}/manifest.json can't be read; see the file.", icon=":material/error:")
    st.stop()
manifest = schemas.load(run_dir, Manifest)
busy = ui.get_runner().busy()

st.subheader(run_dir.name, anchor=False)
with st.container(horizontal=True, vertical_alignment="bottom"):
    if st.button(
        "Open notes",
        icon=":material/description:",
        on_click=_open_notes,
        args=(run_dir.name,),
        disabled=not (run_dir / "notes.json").is_file(),
    ):
        st.switch_page("app_pages/notes.py")
    st.button(
        "Resume",
        icon=":material/play_arrow:",
        type="primary",
        on_click=_start,
        args=(run_dir,),
        disabled=busy or not ui.resumable(manifest),
        help="Re-run from the first incomplete stage; finished stages are reused.",
    )
    stage = st.selectbox(
        "Stage",
        STAGES,
        index=STAGES.index("reduce"),
        format_func=lambda s: ui.STAGE_LABELS.get(s, s),
        key="runs_stage",
        width=200,
    )
    st.button(
        "Re-run stage",
        icon=":material/refresh:",
        on_click=_start,
        args=(run_dir, [stage]),
        disabled=busy,
        help="Run this stage again, and every stage after it.",
    )
    if st.button(
        "Prune media",
        icon=":material/content_cut:",
        disabled=busy or manifest.pruned,
    ):
        confirm_prune(run_dir)
    if st.button("Delete", icon=":material/delete:", disabled=busy):
        confirm_delete(run_dir)

if stage_rows := ui.stage_rows(manifest):
    st.dataframe(
        stage_rows,
        hide_index=True,
        column_config={
            "stage": st.column_config.TextColumn("Stage"),
            "status": st.column_config.TextColumn("Status"),
            "seconds": st.column_config.NumberColumn("Time", format="%.1f s"),
            "peak_gb": st.column_config.NumberColumn("Peak memory", format="%.2f GB"),
            "error": st.column_config.TextColumn("Error"),
        },
        alt=f"Stage status, time and peak memory for {run_dir.name}",
    )
if memory := ui.memory_rows(manifest):
    st.bar_chart(
        memory,
        x="stage",
        y="peak_gb",
        color="model",
        horizontal=True,
        sort=False,
        x_label="Stage",
        y_label="Peak memory (GB)",
        alt=f"Peak memory per model stage for {run_dir.name}",
    )
