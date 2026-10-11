"""Ask (PLAN §5.7, §7): chat over the current run, answers cited like the notes. Each cited
source is a pill that opens a right-side drawer with its audio (from the cited moment) and
slides. History lives in session state per run; nothing is written to the run."""

import streamlit as st

from cohr import ask, render, schemas, ui
from cohr.models import ModelLoadError
from cohr.schemas import Citation, Manifest, NotesDoc

st.set_page_config(page_title="Ask · cohr", layout="centered")

HISTORY_KEY = "ask_history"  # {run_id: [ask.Answer]}
OPEN_KEY = "ask_open"  # the Citation whose drawer opens on this run
ERROR_KEY = "ask_error"


def _open(key: str, citations: list[Citation]) -> None:
    """A pill was picked: remember its source for the drawer and clear the pill, so picking
    the same source again reopens it."""
    if (i := st.session_state.get(key)) is not None:
        st.session_state[OPEN_KEY] = citations[i]
    st.session_state[key] = None


def _clear(run_id: str) -> None:
    st.session_state[HISTORY_KEY].pop(run_id, None)
    st.session_state.pop(ERROR_KEY, None)


run_dir = ui.current_run()
if run_dir is None:
    st.title("Ask")
    st.info("No run selected. Start one on the New run page.", icon=":material/info:")
    st.stop()

st.title("Ask")
st.caption(run_dir.name)
if not (run_dir / NotesDoc.FILENAME).is_file():
    st.info(
        "Ask works once the run has notes. Finish it from the Runs page.",
        icon=":material/hourglass_empty:",
    )
    st.stop()

manifest = schemas.load(run_dir, Manifest)
sources = ui.ask_sources(str(run_dir), (run_dir / NotesDoc.FILENAME).stat().st_mtime)
media = ui.run_media(run_dir, manifest, sources.slides)
by_seg = {s.seg_id: s for s in sources.segments}
seg_t0 = {s.seg_id: s.t0 for s in sources.segments}
history: list[ask.Answer] = st.session_state.setdefault(HISTORY_KEY, {}).setdefault(
    run_dir.name, []
)


@st.dialog("Source", position="right", width="medium")
def drawer(c: Citation) -> None:
    if c.kind == "seg" and c.id in by_seg:
        t = c.t if c.t is not None else by_seg[c.id].t0
        ui.segment_block(by_seg[c.id], media, cited=[t], start=t, anchor=False)
    elif c.kind == "slide" and c.id in media.slides:
        ui.slide_block(media.slides[c.id], media, anchor=False)
    if media.pruned:
        st.caption("Media pruned: audio and slide images are gone.")


if not history:
    st.caption(
        "Answers come only from this meeting's transcript and slides, and cite their sources."
    )
for i, answer in enumerate(history):
    with st.chat_message("user"):
        st.markdown(render.app_escape(answer.question))
    with st.chat_message("assistant"):
        if answer.claims:
            st.markdown(render.answer_md(answer.claims, sources.segments, answer.language))
        else:
            st.caption("No answer came back. Try asking again in other words.")
        notes = []
        if answer.turns_dropped:
            notes.append(f"{answer.turns_dropped} earlier question(s) left out to fit")
        if answer.transcript_limit is not None:
            notes.append("transcripts shortened to fit")
        if notes:
            st.caption(" · ".join(notes).capitalize() + ".")
        cited = ui.unique_citations([c for claim in answer.claims for c in claim.citations])
        if cited:
            key = f"ask_pill_{run_dir.name}_{i}"
            st.pills(
                "Sources",
                list(range(len(cited))),
                format_func=lambda k, cited=cited: ui.citation_label(cited[k], seg_t0),
                key=key,
                on_change=_open,
                args=(key, cited),
                label_visibility="collapsed",
            )

if error := st.session_state.pop(ERROR_KEY, None):
    st.error(error, icon=":material/error:")
if history:
    st.button("Clear chat", icon=":material/delete_sweep:", on_click=_clear, args=(run_dir.name,))

busy = ui.get_runner().busy()
if busy:
    # §7 says Ask waits for the models; it waits here, not on the lock (NOTES.md, 5b).
    st.caption(":material/hourglass_top: A job is using the models. Ask is back when it finishes.")
question = st.chat_input("Ask about this meeting", disabled=busy, submit_mode="disable")

if opened := st.session_state.pop(OPEN_KEY, None):
    drawer(opened)

if question:
    with st.chat_message("user"):
        st.markdown(render.app_escape(question))
    with st.chat_message("assistant"), st.spinner("Reading the meeting…"):
        try:
            answer = ask.run(
                ui.get_models(), run_dir, question, [a.turn() for a in history], sources
            )
        except (ask.AskError, ModelLoadError) as e:
            st.session_state[ERROR_KEY] = str(e)
        else:
            history.append(answer)
    st.rerun()
