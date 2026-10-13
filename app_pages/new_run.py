"""New run (PLAN §7, §12): upload or local path, languages, Advanced options, live progress."""

from pathlib import Path

import streamlit as st

from cohr import storage, ui
from cohr.asr import LANGUAGES
from cohr.config import (
    BETA_LANGUAGES,
    TARGET_LANGUAGES,
    Config,
    FuseConfig,
    language_name,
    runs_dir,
)
from cohr.ingest import IngestError, probe
from cohr.jobs import JobBusy
from cohr.models import PIPELINE_MODELS, ModelsMissing
from cohr.storage import DiskSpaceError, Source

st.set_page_config(page_title="New run · cohr", layout="centered")

MEDIA_TYPES = ["wav", "mp3", "m4a", "aac", "flac", "ogg", "mp4", "mov", "mkv", "webm", "m4v"]
SLIDE_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
ERROR_KEY = "nr_error"


class FormError(ValueError):
    pass


def _local(text: str) -> Path:
    path = Path(text.strip().strip("'\"")).expanduser()
    if not path.is_file():
        raise FormError(f"No file at {path}.")
    return path


def _stage(runs: Path, uploaded) -> Path:
    return storage.stage_upload(runs, uploaded.name, iter(lambda: uploaded.read(1 << 20), b""))


def _create_run() -> Path:
    ss = st.session_state
    media_up, media_path = ss.get("nr_media"), (ss.get("nr_media_path") or "").strip()
    slide_ups = ss.get("nr_slides") or []
    slide_paths = [p for p in (ss.get("nr_slide_paths") or "").splitlines() if p.strip()]
    if (media_up is None) == (not media_path):
        raise FormError("Give one recording: upload a file or enter a local path, not both.")

    sources = []
    if media_path:
        path = _local(media_path)
        sources.append(Source(path, ui.media_kind(path.name)))
    sources += [Source(_local(p), "slides") for p in slide_paths]

    config = Config(
        source_language=ss["nr_source_lang"],
        target_language=ss["nr_target_lang"],
        memory_policy=ss["nr_policy"],
        source_mode=ss["nr_source_mode"],
        fuse=FuseConfig(window_s=ss["nr_window_min"] * 60.0),
    )
    ui.get_models().require(PIPELINE_MODELS)

    runs = runs_dir()
    uploads = [u for u in (media_up, *slide_ups) if u is not None]
    storage.check_free_space(runs, sum(u.size for u in uploads))
    staged: list[Path] = []
    try:
        if media_up is not None:
            staged.append(_stage(runs, media_up))
            kind = ui.media_kind(media_up.name)
            sources.insert(0, Source(staged[-1], kind, upload=True))
        for up in slide_ups:
            staged.append(_stage(runs, up))
            sources.append(Source(staged[-1], "slides", upload=True))
        duration = probe(sources[0].path).duration_s
        name = (ss.get("nr_name") or "").strip() or None
        return storage.create_run(runs, sources, config, name=name, duration_s=duration)
    except BaseException:
        for path in staged:
            storage.discard_upload(path)
        raise


def _submit() -> None:
    """Form callback: runs before the script, so it can set the sidebar's current run."""
    st.session_state.pop(ERROR_KEY, None)
    try:
        run_dir = _create_run()
        ui.get_runner().start(run_dir)
    except (FormError, ModelsMissing, DiskSpaceError, JobBusy) as e:
        st.session_state[ERROR_KEY] = str(e)
        return
    except IngestError as e:
        st.session_state[ERROR_KEY] = f"Could not read the recording: {e}"
        return
    st.session_state[ui.RUN_KEY] = run_dir.name


st.title("New run")
runner = ui.get_runner()
busy = runner.busy()

if error := st.session_state.get(ERROR_KEY):
    st.error(error, icon=":material/error:")
if missing := ui.missing_models():
    st.warning(
        f"Models not downloaded: {', '.join(missing)}. Run `uv run cohr models download` "
        "in a terminal, then reload this page.",
        icon=":material/download:",
    )

with st.form("new_run", border=False):
    st.file_uploader("Recording (audio or video)", type=MEDIA_TYPES, key="nr_media")
    st.text_input(
        "Or a local file path",
        key="nr_media_path",
        placeholder="/Users/you/Movies/meeting.mp4",
        help="Recommended for large videos: nothing is uploaded or copied, the run links to it.",
    )
    st.file_uploader(
        "Slides (optional)", type=SLIDE_TYPES, accept_multiple_files=True, key="nr_slides"
    )
    st.text_area("Or local slide paths, one per line", key="nr_slide_paths", height="content")
    left, right = st.columns(2)
    left.selectbox(
        "Spoken language",
        LANGUAGES,
        index=LANGUAGES.index("en"),
        format_func=language_name,
        key="nr_source_lang",
    )
    # R7B's 23 first, then "Other (beta)": Tiny Aya translates English notes (§5.6, 6d-1).
    targets = [*TARGET_LANGUAGES, *sorted(BETA_LANGUAGES, key=language_name)]
    right.selectbox(
        "Notes language",
        targets,
        index=targets.index("en"),
        format_func=lambda c: (
            f"{language_name(c)} (beta)" if c in BETA_LANGUAGES else language_name(c)
        ),
        key="nr_target_lang",
        help="Languages marked (beta): the notes are written in English, then translated by "
        "the Map model. Headings stay in English; Ask answers in English unless asked in "
        "another language.",
    )
    st.text_input("Run name (optional)", key="nr_name", placeholder="Defaults to the file name")
    with st.expander("Advanced"):
        st.number_input(
            "Window length (minutes)",
            min_value=1,
            max_value=10,
            value=5,
            key="nr_window_min",
            help="Length of each segment Map summarizes (D12: 5 minutes).",
        )
        st.segmented_control(
            "Memory policy",
            ["resident", "sequential"],
            default="resident",
            key="nr_policy",
            required=True,
            help="Resident keeps the models loaded (~20 GB, fast re-runs). Sequential unloads "
            "each one after its stage (~8 GB peak).",
        )
        st.segmented_control(
            "Local files",
            ["link", "copy"],
            default="link",
            key="nr_source_mode",
            required=True,
            format_func=lambda m: "Link" if m == "link" else "Copy into the run",
            help="Uploads are always stored in the run.",
        )
    st.form_submit_button(
        "Run", type="primary", icon=":material/play_arrow:", on_click=_submit, disabled=busy
    )

if busy:
    st.caption("A job is running. Start another when it finishes, or cancel it.")

ui.job_panel(show_result=True)

job = runner.current
if (
    job is not None
    and job.snapshot().status == "done"
    and st.button("Open notes", icon=":material/description:", type="primary")
):
    st.session_state[ui.RUN_KEY] = job.run_id
    st.switch_page("app_pages/notes.py")
