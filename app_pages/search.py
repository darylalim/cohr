"""Search (PLAN §13, §7): every run's transcript, slides and notes, and the action items of
all meetings. A result's source opens in a right-side drawer (audio from the moment, its
slides), or in Notes at that moment. The index syncs with the run files on every load."""

import streamlit as st

from cohr import index, render, schemas, storage, ui
from cohr.config import runs_dir
from cohr.schemas import Manifest, SegmentsFile, SlidesFile

st.set_page_config(page_title="Search · cohr", layout="wide")

QUERY_KEY = "search_query"
OPEN_KEY = "search_open"  # the ui.Focus whose drawer opens on this rerun
LIMIT = 50
KIND_LABELS = {"note": "Notes", "transcript": "Transcript", "slide": "Slides"}
SECTION_LABELS = render.labels("en")


def _focus(focus: ui.Focus) -> None:
    """Before the rerun (the sidebar's run picker is already drawn when the page runs)."""
    st.session_state[ui.RUN_KEY] = focus.run_id
    st.session_state[ui.FOCUS_KEY] = focus


def _listen(focus: ui.Focus) -> None:
    st.session_state[OPEN_KEY] = focus


@st.dialog("Source", position="right", width="medium")
def drawer(focus: ui.Focus) -> None:
    run_dir = runs_dir() / focus.run_id
    if not storage.is_run(run_dir) or not (run_dir / SegmentsFile.FILENAME).is_file():
        st.caption("This run has no sources to show yet.")
        return
    segs = {s.seg_id: s for s in schemas.load(run_dir, SegmentsFile).segments}
    slides = schemas.load(run_dir, SlidesFile).slides
    media = ui.run_media(run_dir, schemas.load(run_dir, Manifest), slides)
    st.caption(focus.run_id)
    if focus.seg_id in segs:
        seg = segs[focus.seg_id]
        t = focus.t if focus.t is not None else seg.t0
        ui.segment_block(seg, media, cited=[t], start=t, anchor=False)
    elif focus.slide_id in media.slides:
        ui.slide_block(media.slides[focus.slide_id], media, anchor=False)
    if media.pruned:
        st.caption("Media pruned: audio and slide images are gone.")


def snippet_md(hit: index.Hit) -> str:
    text = " ".join(render.app_escape(hit.snippet).split())
    text = text.replace(index.MARK_START, "**").replace(index.MARK_END, "**")
    return render.isolate(text, hit.lang or "")


def label(hit: index.Hit) -> str:
    if hit.kind == "note" and hit.section:
        return SECTION_LABELS[hit.section]
    return KIND_LABELS[hit.kind]


def actions(focus: ui.Focus, key: str) -> None:
    """Listen (the drawer) and Open notes, for one result or action item."""
    with st.container(horizontal=True):
        st.button(
            "Listen" if focus.seg_id is not None else "Show slide",
            icon=":material/headphones:" if focus.seg_id is not None else ":material/slideshow:",
            key=f"{key}_listen",
            on_click=_listen,
            args=(focus,),
            disabled=focus.anchor() is None,
            type="tertiary",
        )
        # In the script, not a callback: callbacks run before st.navigation registers pages.
        if st.button(
            "Open notes",
            icon=":material/description:",
            key=f"{key}_notes",
            on_click=_focus,
            args=(focus,),
            type="tertiary",
        ):
            st.switch_page("app_pages/notes.py")


st.title("Search")
runs = runs_dir()
with index.open_index(runs) as con:
    if not index.run_count(con):
        st.info("No runs yet. Start one on the New run page.", icon=":material/info:")
        st.stop()
    languages = index.languages(con)
    owners = index.owners(con)

    tab_search, tab_actions = st.tabs(["Search", "Action items"])
    with tab_search:
        query = st.text_input(
            "Search",
            key=QUERY_KEY,
            placeholder="Words, names or numbers from any meeting",
            label_visibility="collapsed",
            icon=":material/search:",
        )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            kinds = st.pills(
                "Look in",
                list(KIND_LABELS),
                format_func=KIND_LABELS.__getitem__,
                selection_mode="multi",
                key="search_kinds",
            )
            langs = st.multiselect(
                "Language",
                languages,
                key="search_langs",
                placeholder="Any",
                width=220,
            )
            dates = st.date_input("Run date", value=[], key="search_dates", width=240)
        since = dates[0] if len(dates) > 0 else None
        until = dates[1] if len(dates) > 1 else since
        results = (
            index.search(
                con, query, kinds=kinds, langs=langs, since=since, until=until, limit=LIMIT
            )
            if query.strip()
            else None
        )

    with tab_actions:
        picked = st.multiselect(
            "Owner", owners, key="search_owners", placeholder="Anyone", width=360
        )
        rows = index.action_items(con, owners=picked)

with tab_search:
    if results is None:
        st.caption(
            "Searches the transcript, slides and notes of every run. Words in Chinese, "
            "Japanese or Korean need at least 3 characters."
        )
    elif not results.hits:
        st.caption("No matches.")
    else:
        if results.any_term:
            st.caption("No result has every word; these have some of them.")
        elif len(results.hits) == LIMIT:
            st.caption(f"The {LIMIT} best matches. Add words or filters to narrow them.")
        for i, hit in enumerate(results.hits):
            focus = ui.Focus(hit.run_id, hit.seg_id, hit.slide_id, hit.t)
            with st.container(border=True):
                st.caption(
                    f"**{hit.run_id}** · {label(hit)} · {index.where(hit.t, hit.slide_id)}"
                    + (f" · {hit.lang}" if hit.lang else "")
                )
                st.markdown(snippet_md(hit))
                actions(focus, f"hit_{i}")

with tab_actions:
    if not rows:
        st.caption("No action items." if not picked else "No action items for them.")
    else:
        event = st.dataframe(
            [
                {
                    "run": r.run_id,
                    "created": r.created_at,
                    "owner": r.owner,
                    "due": r.due,
                    "task": r.text,
                    "at": index.where(r.t, r.slide_id),
                }
                for r in rows
            ],
            hide_index=True,
            key="actions_table",
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "run": st.column_config.TextColumn("Run"),
                "created": st.column_config.DatetimeColumn("Created", format="YYYY-MM-DD"),
                "owner": st.column_config.TextColumn("Owner"),
                "due": st.column_config.TextColumn("Due"),
                "task": st.column_config.TextColumn("Task", width="large"),
                "at": st.column_config.TextColumn("At"),
            },
            alt="Action items from every run, with owner, due date and where they were said",
        )
        if selected := event.selection.rows:
            r = rows[selected[0]]
            actions(ui.Focus(r.run_id, r.seg_id, r.slide_id, r.t), "action")
        else:
            st.caption("Select a task to listen to where it was said or open its notes.")

if opened := st.session_state.pop(OPEN_KEY, None):
    drawer(opened)
