"""cohr: cited meeting notes on MLX. Streamlit entry point (PLAN §7)."""

import streamlit as st

from cohr import ui
from cohr.config import runs_dir

st.set_page_config(
    page_title="cohr", page_icon=":material/edit_note:", initial_sidebar_state="auto"
)

ui.startup(str(runs_dir().resolve()))

page = st.navigation(
    [
        st.Page("app_pages/new_run.py", title="New run", icon=":material/add_circle:"),
        st.Page("app_pages/notes.py", title="Notes", icon=":material/description:"),
        st.Page("app_pages/ask.py", title="Ask", icon=":material/forum:"),
        st.Page("app_pages/runs.py", title="Runs", icon=":material/folder_open:"),
    ],
    position="sidebar",
)
ui.sidebar()
page.run()
