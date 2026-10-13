"""PLAN §7 UI through AppTest: a fake ModelManager and a fixture run dir, no models."""

import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import streamlit as st
from streamlit.testing.v1 import AppTest

from cohr import atomic, schemas, storage, ui
from cohr.config import Config, runs_dir
from cohr.jobs import JobRunner
from cohr.pipeline import Stage, StageContext
from cohr.schemas import (
    Citation,
    Claim,
    Frame,
    FramesFile,
    Manifest,
    MapFile,
    NotesDoc,
    NotesSections,
    Segment,
    SegmentsFile,
    Slide,
    SlidesFile,
    StageState,
    Transcript,
    TranscriptItem,
)
from cohr.storage import Source
from tests.conftest import fake_manager, pattern
from tests.test_reduce import FakeR7B

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 30


def fake_stages() -> list[Stage]:
    def run(ctx: StageContext) -> None:
        atomic.write_text(ctx.run_dir / "fake.txt", "ok")

    return [Stage(name="fake", run=run, outputs=("fake.txt",), format_version=1)]


@pytest.fixture(autouse=True)
def fake_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(ui, "make_models", lambda: fake_manager(Config(), tmp_path))
    monkeypatch.setattr(ui, "make_runner", lambda m: JobRunner(m, stages=fake_stages))
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def make_fixture_run(tmp_path: Path) -> Path:
    """Three segments, two slides (one unaligned), notes citing both kinds, audio and frames."""
    src = tmp_path / "media" / "meeting.wav"
    src.parent.mkdir(exist_ok=True)
    src.write_bytes(b"x")
    run_dir = storage.create_run(runs_dir(), [Source(src, "audio")], Config(), name="sync")
    sf.write(run_dir / "audio.wav", np.zeros(16000 * 9, dtype=np.int16), 16000)
    (run_dir / "frames").mkdir()
    frames = []
    for k in range(2):
        pattern(k, (160, 90)).save(run_dir / "frames" / f"{k:03d}.png")
        frames.append(
            Frame(frame_id=k, path=f"frames/{k:03d}.png", pts=None, phash="0", source="x")
        )
    schemas.save(run_dir, FramesFile(frames=frames))
    slides = [
        Slide(
            slide_id=k,
            frame_id=k,
            t=None,
            t_source="none",
            title=title,
            text_md="",
            tables_md=[],
            chart_notes="",
            visual_notes="",
        )
        for k, title in enumerate(["Budget", "Appendix"])
    ]
    schemas.save(run_dir, SlidesFile(slides=slides))
    segments = [
        Segment(
            seg_id=i,
            t0=3.0 * i,
            t1=3.0 * (i + 1),
            chunk_ids=[i],
            transcript_text="hi",
            slide_ids=[0] if i == 0 else [],
        )
        for i in range(3)
    ]
    schemas.save(run_dir, SegmentsFile(segments=segments, unaligned_slide_ids=[1]))
    schemas.save(run_dir, MapFile(target_language="en", notes=[]))
    items = [TranscriptItem(chunk_id=i, t0=3.0 * i, t1=3.0 * (i + 1), text="hi") for i in range(3)]
    schemas.save(run_dir, Transcript(language="en", runtime="mlx-speech", items=items))
    schemas.save(
        run_dir,
        NotesDoc(
            sections=NotesSections(
                summary=[
                    Claim(
                        text="Budget is $14,500.",
                        citations=[Citation(kind="seg", id=0, t=1.5), Citation(kind="slide", id=0)],
                    ),
                    Claim(text="Nobody said this.", citations=[]),
                ],
                key_points=[],
                decisions=[Claim(text="Ship it", citations=[Citation(kind="seg", id=2, t=7.0)])],
                action_items=[],
                open_questions=[],
            ),
            passes=1,
            language="en",
            raw="",
        ),
    )
    manifest = schemas.load(run_dir, Manifest)
    t = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)
    for name, peak in (("ingest", None), ("transcribe", 5.5), ("map", 9.0), ("reduce", 18.2)):
        manifest.stages[name] = StageState(
            status="done",
            started_at=t,
            finished_at=t + timedelta(seconds=30),
            peak_mem_bytes=int(peak * storage.GB) if peak else None,
        )
    manifest.stages["render"] = StageState(status="done")
    schemas.save(run_dir, manifest)
    return run_dir


def app(page: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    if page:
        at.switch_page(page).run()
    assert not at.exception, at.exception
    return at


PAGES = ["app_pages/new_run.py", "app_pages/notes.py", "app_pages/ask.py", "app_pages/runs.py"]


@pytest.mark.parametrize("page", [None, *PAGES])
def test_every_page_renders_without_runs(page: str | None) -> None:
    app(page)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_a_run(tmp_path: Path, page: str) -> None:
    make_fixture_run(tmp_path)
    at = app(page)
    assert any(c.value == ui.FOOTER for c in at.sidebar.caption)


def test_notes_has_one_sources_block_per_segment(tmp_path: Path) -> None:
    run_dir = make_fixture_run(tmp_path)
    at = app("app_pages/notes.py")
    assert at.title[0].value == run_dir.name
    anchors = [h.anchor for h in at.subheader]
    assert [a for a in anchors if a and a.startswith("seg-")] == ["seg-0", "seg-1", "seg-2"]
    assert [a for a in anchors if a and a.startswith("slide-")] == ["slide-0", "slide-1"]
    assert anchors.index("slide-1") > anchors.index("seg-2")  # unaligned slides come last
    md = "\n".join(m.value for m in at.markdown)
    assert "[0:01](#seg-0)" in md and "[slide 1](#slide-0)" in md
    assert "\\$14,500" in md and ":gray-badge[uncited]" in md
    assert any("1 claim without a citation" in c.value for c in at.caption)
    assert len(at.get("audio")) == 3
    assert len(at.get("image")) == 2


def test_notes_without_media_after_prune(tmp_path: Path) -> None:
    run_dir = make_fixture_run(tmp_path)
    storage.prune_media(run_dir)
    at = app("app_pages/notes.py")
    assert len(at.get("audio")) == 0
    assert any("Media pruned" in c.value for c in at.caption)


def test_notes_rereduce_starts_a_forced_job(tmp_path: Path) -> None:
    run_dir = make_fixture_run(tmp_path)
    at = app("app_pages/notes.py")
    rereduce = next(b for b in at.button if b.label == "Re-reduce")
    rereduce.click().run()
    assert not at.exception
    job = ui.get_runner().current
    assert job is not None and job.run_id == run_dir.name and job.force == ("reduce",)


def test_runs_lists_the_run(tmp_path: Path) -> None:
    run_dir = make_fixture_run(tmp_path)
    at = app("app_pages/runs.py")
    table = at.dataframe[0].value
    assert list(table["run"]) == [run_dir.name]
    assert list(table["status"]) == ["done"]


def test_new_run_refuses_a_missing_local_path(tmp_path: Path) -> None:
    at = app("app_pages/new_run.py")
    at.text_input(key="nr_media_path").set_value(str(tmp_path / "nope.mp4"))
    at.button[0].click().run()
    assert not at.exception
    assert "No file at" in at.error[0].value
    assert storage.list_runs(runs_dir()) == []


def test_new_run_needs_a_recording() -> None:
    at = app("app_pages/new_run.py")
    at.button[0].click().run()
    assert "Give one recording" in at.error[0].value


@pytest.mark.skipif(not __import__("shutil").which("ffprobe"), reason="ffprobe not installed")
def test_new_run_from_a_local_path_starts_a_job(tmp_path: Path) -> None:
    wav = tmp_path / "media" / "talk.wav"
    wav.parent.mkdir(exist_ok=True)
    sf.write(wav, np.zeros(16000, dtype=np.int16), 16000)
    at = app("app_pages/new_run.py")
    at.text_input(key="nr_media_path").set_value(str(wav))
    at.selectbox(key="nr_target_lang").select("de")
    at.button[0].click().run()
    assert not at.exception and not at.error
    [run_dir] = storage.list_runs(runs_dir())
    assert at.session_state[ui.RUN_KEY] == run_dir.name
    manifest = schemas.load(run_dir, Manifest)
    assert manifest.config["target_language"] == "de"
    assert manifest.inputs[0].mode == "link"
    job = ui.get_runner().current
    assert job is not None and job.wait(5) and job.snapshot().status == "done"


def test_new_run_offers_beta_languages_after_r7bs() -> None:
    at = app("app_pages/new_run.py")
    box = at.selectbox(key="nr_target_lang")
    assert box.value == "en"
    assert "Swahili (beta)" in box.options[23:] and "German" in box.options[:23]
    assert not any("(beta)" in o for o in box.options[:23])


def test_recovery_runs_on_app_start(tmp_path: Path) -> None:
    run_dir = make_fixture_run(tmp_path)
    manifest = schemas.load(run_dir, Manifest)
    old = datetime.now(UTC) - timedelta(minutes=5)
    manifest.stages["map"] = StageState(status="running", started_at=old, heartbeat_at=old)
    schemas.save(run_dir, manifest)
    app()
    assert schemas.load(run_dir, Manifest).stages["map"].status == "interrupted"


def test_notes_below_renders_every_source_block(tmp_path: Path) -> None:
    make_fixture_run(tmp_path)
    at = app("app_pages/notes.py")
    at.segmented_control(key="notes_sources").set_value("Below").run()
    assert not at.exception
    anchors = [h.anchor for h in at.subheader if h.anchor]
    assert anchors == ["seg-0", "slide-0", "seg-1", "seg-2", "slide-1"]
    assert len(at.get("audio")) == 3 and len(at.get("image")) == 2
    md = "\n".join(m.value for m in at.markdown)
    assert "[0:01](#seg-0)" in md  # the anchors are the same in both layouts


def test_open_in_ask_switches_page_with_the_run(tmp_path: Path) -> None:
    run_dir = make_fixture_run(tmp_path)
    at = app("app_pages/notes.py")
    next(b for b in at.button if b.label == "Open in Ask").click().run()
    assert not at.exception
    assert at.title[0].value == "Ask"
    assert at.caption[0].value == run_dir.name
    assert at.session_state[ui.RUN_KEY] == run_dir.name


def _with_r7b(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reply: str) -> FakeR7B:
    lm = FakeR7B(reply)
    monkeypatch.setattr(ui, "make_models", lambda: fake_manager(Config(), tmp_path, reduce=lm))
    st.cache_resource.clear()
    return lm


def test_ask_answers_with_pills_for_cited_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_fixture_run(tmp_path)
    reply = "<co>The budget is $14,500</co: 0:[0,3]>. Nobody knows the date."
    lm = _with_r7b(monkeypatch, tmp_path, reply)
    at = app("app_pages/ask.py")
    at.chat_input[0].set_value("What is the budget?").run()
    assert not at.exception
    assert lm.calls  # answered through the ModelManager
    md = "\n".join(m.value for m in at.markdown)
    assert "What is the budget?" in md
    # No transcript chunk matches the span, so the jump time is the segment start (§5.6).
    assert "The budget is \\$14,500. \\[0:00\\] \\[slide 1\\]" in md
    assert ":gray[Nobody knows the date.] :gray-badge[uncited]" in md
    [pills] = at.button_group
    assert pills.options == ["0:00", "slide 1"]


def test_ask_pill_opens_the_source_drawer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_fixture_run(tmp_path)
    _with_r7b(monkeypatch, tmp_path, "<co>Ship it</co: 0:[2]>.")
    at = app("app_pages/ask.py")
    at.chat_input[0].set_value("What was decided?").run()
    at.button_group[0].set_value(0).run()
    assert not at.exception
    assert any(h.value == "0:06–0:09" for h in at.subheader)  # seg 2's block, in the drawer
    assert len(at.get("audio")) == 1
    assert at.button_group[0].value is None  # cleared, so the same pill reopens it


def test_ask_follow_up_sends_history_without_markup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_fixture_run(tmp_path)
    lm = _with_r7b(monkeypatch, tmp_path, "<co>Ship it</co: 0:[2]>.")
    at = app("app_pages/ask.py")
    at.chat_input[0].set_value("What was decided?").run()
    at.chat_input[0].set_value("Who owns that?").run()
    assert not at.exception
    sent = lm.instructions[-1]
    assert "Q: What was decided?\nA: Ship it." in sent and "Question: Who owns that?" in sent
    assert len(at.button_group) == 2  # one pill row per answer


def test_ask_input_is_disabled_while_a_job_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def wait(ctx: StageContext) -> None:
        release.wait(10)
        atomic.write_text(ctx.run_dir / "fake.txt", "ok")

    stage = Stage(name="fake", run=wait, outputs=("fake.txt",), format_version=1)
    monkeypatch.setattr(ui, "make_runner", lambda m: JobRunner(m, stages=lambda: [stage]))
    run_dir = make_fixture_run(tmp_path)
    at = app("app_pages/ask.py")
    assert not at.chat_input[0].disabled
    job = ui.get_runner().start(run_dir)
    try:
        at.run()
        assert at.chat_input[0].disabled
        assert any("A job is using the models" in c.value for c in at.caption)
    finally:
        release.set()
        job.wait(5)
