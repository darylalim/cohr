"""Notes (PLAN §7, §5.6): rendered notes next to a Sources panel with audio and slides."""

from pathlib import Path

import streamlit as st

from cohr import render, schemas, ui
from cohr.jobs import JobBusy
from cohr.mapper import clock
from cohr.reduce import slide_label
from cohr.schemas import FramesFile, Manifest, NotesDoc, SegmentsFile, Slide, SlidesFile

st.set_page_config(page_title="Notes · cohr", layout="wide")

SOURCES_HEIGHT = 760  # px; the panel scrolls on its own
ERROR_KEY = "notes_error"


def _rereduce(run_dir: Path) -> None:
    st.session_state.pop(ERROR_KEY, None)
    try:
        ui.get_runner().start(run_dir, force=["reduce"])
    except JobBusy as e:
        st.session_state[ERROR_KEY] = str(e)


def _slide(slide: Slide, frames: dict[int, Path]) -> None:
    label = slide_label(slide.slide_id).capitalize()
    title = slide.title or label
    st.subheader(
        f"{label}: {slide.title}" if slide.title else label,
        anchor=f"slide-{slide.slide_id}",
        icon=":material/slideshow:",
    )
    image = frames.get(slide.frame_id)
    if image is not None and image.is_file():
        st.image(str(image), alt=title)
    else:
        st.caption("Slide image not available (media pruned).")


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

notes = schemas.load(run_dir, NotesDoc)
segs = schemas.load(run_dir, SegmentsFile)
slides = {s.slide_id: s for s in schemas.load(run_dir, SlidesFile).slides}
frames: dict[int, Path] = {}
if (run_dir / FramesFile.FILENAME).is_file():
    frames = {f.frame_id: run_dir / f.path for f in schemas.load(run_dir, FramesFile).frames}

if error := st.session_state.get(ERROR_KEY):
    st.error(error, icon=":material/error:")
with st.container(horizontal=True):
    st.download_button(
        "Download notes.md",
        render.render(notes, segs.segments, list(slides.values()), "portable"),
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
if uncited := ui.uncited_count(notes):
    st.caption(f"{uncited} claim{'s' if uncited != 1 else ''} without a citation (shown in gray).")

left, right = st.columns([3, 2], gap="large")
with left, st.container(width=880):
    st.markdown(render.render(notes, segs.segments, list(slides.values()), "app"))

with right:
    st.subheader(render.labels(notes.language)["sources"], anchor=False)
    wav = run_dir / "audio.wav"
    has_audio = wav.is_file() and not manifest.pruned
    cited = ui.cited_times(notes, {s.seg_id: s.t0 for s in segs.segments})
    with st.container(height=SOURCES_HEIGHT, border=True):
        if manifest.pruned:
            st.caption("Media pruned: audio and slide images are gone; notes still work.")
        for seg in segs.segments:
            span = ui.fmt_range(seg.t0, seg.t1)
            st.subheader(span, anchor=f"seg-{seg.seg_id}", icon=":material/schedule:")
            times = cited.get(seg.seg_id, [])
            if times:
                st.caption("Cited at " + ", ".join(clock(t) for t in times))
            if has_audio:
                start = times[0] if times else seg.t0
                st.audio(
                    ui.audio_clip(str(wav), wav.stat().st_mtime, seg.t0, seg.t1),
                    format="audio/wav",
                    start_time=max(0, int(start - seg.t0)),
                    alt=f"Recording {span}",
                )
            for slide_id in seg.slide_ids:
                if slide_id in slides:
                    _slide(slides[slide_id], frames)
        if segs.unaligned_slide_ids:
            st.divider()
            st.caption("Slides not matched to a moment in the recording")
            for slide_id in segs.unaligned_slide_ids:
                if slide_id in slides:
                    _slide(slides[slide_id], frames)
