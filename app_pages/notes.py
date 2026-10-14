"""Notes (PLAN §7, §5.6): rendered notes with a Sources panel (audio and slides) beside or
below them (D21)."""

from pathlib import Path

import streamlit as st

from cohr import render, schemas, ui
from cohr.jobs import JobBusy
from cohr.schemas import Manifest, NotesDoc, SegmentsFile, SlidesFile

st.set_page_config(page_title="Notes · cohr", layout="wide")

SOURCES_HEIGHT = 760  # px; beside the notes, the panel scrolls on its own
ERROR_KEY = "notes_error"
LAYOUT_KEY = "notes_sources"  # "Beside" or "Below" (D21), kept for the session
LAYOUTS = ["Beside", "Below"]


def _rereduce(run_dir: Path) -> None:
    st.session_state.pop(ERROR_KEY, None)
    try:
        ui.get_runner().start(run_dir, force=["reduce"])
    except JobBusy as e:
        st.session_state[ERROR_KEY] = str(e)


run_dir = ui.current_run()
if run_dir is None:
    st.title("Notes")
    st.info("No run selected. Start one on the New run page.", icon=":material/info:")
    st.stop()

manifest = schemas.load(run_dir, Manifest)
st.title(run_dir.name)
ui.job_panel(run_dir.name)

if not (run_dir / NotesDoc.FILENAME).is_file():
    st.info(
        f"No notes yet (run status: {ui.run_status(manifest)}). Resume it from the Runs page.",
        icon=":material/hourglass_empty:",
    )
    st.stop()

notes = render.shown_notes(run_dir)
segs = schemas.load(run_dir, SegmentsFile)
slides = schemas.load(run_dir, SlidesFile).slides
media = ui.run_media(run_dir, manifest, slides)

if error := st.session_state.get(ERROR_KEY):
    st.error(error, icon=":material/error:")
reduced = notes if isinstance(notes, NotesDoc) else schemas.load(run_dir, NotesDoc)
if reduced.truncated:
    st.warning(
        "These notes end early: the model reached its output limit (reduce.max_tokens), "
        "so the last sections may be missing or cut short.",
        icon=":material/warning:",
    )
with st.container(horizontal=True, vertical_alignment="bottom"):
    st.download_button(
        "Download notes.md",
        render.render(notes, segs.segments, slides, "portable"),
        file_name=f"{run_dir.name}-notes.md",
        mime="text/markdown",
        icon=":material/download:",
        on_click="ignore",
    )
    st.button(
        "Re-reduce",
        icon=":material/refresh:",
        on_click=_rereduce,
        args=(run_dir,),
        disabled=ui.get_runner().busy(),
        help="Run Reduce and Render again, e.g. after editing the Reduce prompt.",
    )
    # In the script, not a callback: callbacks run before st.navigation registers the pages.
    if st.button("Open in Ask", icon=":material/forum:", help="Ask questions about this run."):
        st.switch_page("app_pages/ask.py")
    st.segmented_control(
        "Sources",
        LAYOUTS,
        default="Beside",
        required=True,
        key=LAYOUT_KEY,
        persist_state="session",
        help="Below suits medium windows, where the Sources column gets narrow.",
    )
if uncited := ui.uncited_count(notes):
    st.caption(f"{uncited} claim{'s' if uncited != 1 else ''} without a citation (shown in gray).")


def sources() -> None:
    cited = ui.cited_times(notes, {s.seg_id: s.t0 for s in segs.segments})
    if media.pruned:
        st.caption("Media pruned: audio and slide images are gone; notes still work.")
    for seg in segs.segments:
        ui.segment_block(seg, media, cited=cited.get(seg.seg_id, []))
    if segs.unaligned_slide_ids:
        st.divider()
        st.caption("Slides not matched to a moment in the recording")
        for slide_id in segs.unaligned_slide_ids:
            if slide_id in media.slides:
                ui.slide_block(media.slides[slide_id], media)


notes_md = render.render(notes, segs.segments, slides, "app")
heading = render.labels(notes.language)["sources"]
if st.session_state.get(LAYOUT_KEY) == "Below":
    # Notes, then the source blocks without the fixed-height box, so anchors scroll the page.
    with st.container(width=880):
        st.markdown(notes_md)
        st.divider()
        st.subheader(heading, anchor=False)
        sources()
else:
    left, right = st.columns([3, 2], gap="large")
    with left, st.container(width=880):
        st.markdown(notes_md)
    with right:
        st.subheader(heading, anchor=False)
        with st.container(height=SOURCES_HEIGHT, border=True):
            sources()
