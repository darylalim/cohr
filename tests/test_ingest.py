"""PLAN §5.1: ffmpeg normalize, scene frames with pts, phash dedupe, PDF/image slides."""

import shutil
from pathlib import Path

import imagehash
import pytest
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from cohr import ingest, pipeline, schemas, storage
from cohr.config import Config, IngestConfig
from cohr.ingest import IngestError
from cohr.schemas import FramesFile
from cohr.storage import Source
from tests.conftest import needs_ffmpeg, run_ffmpeg

INGEST = [ingest.STAGE]

SHOWINFO = """\
[Parsed_showinfo_2 @ 0x6000] config in time_base: 1/10, frame_rate: 10/1
[Parsed_showinfo_2 @ 0x6000] n:   0 pts:      0 pts_time:0       duration:1 fmt:rgb24
[Parsed_showinfo_2 @ 0x6000] n:   1 pts:     20 pts_time:2       duration:1 fmt:rgb24
[Parsed_showinfo_2 @ 0x6000] n:   2 pts:  41237 pts_time:4.1237  duration:1 fmt:rgb24
frame=    3 fps=0.0 q=-0.0 Lsize=N/A time=00:00:04.20
"""


def test_parse_showinfo() -> None:
    assert ingest.parse_showinfo(SHOWINFO) == [0.0, 2.0, 4.1237]
    assert ingest.parse_showinfo("frame= 3 fps=0.0") == []


def test_scene_filter_escapes_commas() -> None:
    f = ingest.scene_filter(0.3, 1600)
    assert f.startswith("select='eq(n\\,0)+gt(scene\\,0.3)'")
    assert "min(1600\\,iw)" in f and "min(1600\\,ih)" in f and f.endswith(",showinfo")


def _h(bits: str) -> imagehash.ImageHash:
    return imagehash.hex_to_hash(f"{int(bits, 2):016x}")


def test_dedupe_compares_with_last_kept_frame() -> None:
    a = _h("0" * 64)
    a_close = _h("1" * 6 + "0" * 58)  # distance 6 from a: duplicate
    b = _h("1" * 7 + "0" * 57)  # distance 7 from a: kept
    b_close = _h("1" * 8 + "0" * 56)  # distance 1 from b: duplicate
    assert ingest.dedupe([a, a_close, b, b_close, a], max_distance=6) == [0, 2, 4]
    assert ingest.dedupe([], 6) == []


def test_fit_caps_long_edge() -> None:
    assert ingest.fit(Image.new("RGB", (3200, 1800)), 1600).size == (1600, 900)
    assert ingest.fit(Image.new("RGB", (900, 2400)), 1600).size == (600, 1600)
    small = Image.new("RGB", (100, 50))
    assert ingest.fit(small, 1600) is small


def test_unsupported_slide_file(tmp_path: Path) -> None:
    p = tmp_path / "deck.key"
    p.write_bytes(b"")
    with pytest.raises(IngestError, match="unsupported"):
        ingest.slide_images(p, 1600)


def test_missing_ffmpeg_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _tool: None)
    with pytest.raises(IngestError, match="brew install ffmpeg"):
        ingest.require()


def test_pdf_pages_scale_to_cap(slide_pdf: Path) -> None:
    pages = ingest.pdf_pages(slide_pdf, 1600)
    assert len(pages) == 2
    assert all(max(p.size) == 1600 for p in pages)


@needs_ffmpeg
def test_probe(tone: Path, video: Path) -> None:
    a = ingest.probe(tone)
    assert a.has_audio and not a.has_video and a.duration_s == pytest.approx(3.0, abs=0.1)
    v = ingest.probe(video)
    assert v.has_audio and v.has_video and v.duration_s == pytest.approx(6.0, abs=0.2)


@needs_ffmpeg
def test_normalize_audio(tmp_path: Path, tone: Path) -> None:
    out = tmp_path / "audio.wav"
    ingest.normalize_audio(tone, out)
    info = sf.info(out)
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert info.duration == pytest.approx(3.0, abs=0.1)
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


@needs_ffmpeg
def test_normalize_audio_failure_leaves_nothing(tmp_path: Path) -> None:
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"not audio")
    with pytest.raises(IngestError, match="ffmpeg failed"):
        ingest.normalize_audio(bad, tmp_path / "audio.wav")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.m4a"]


@needs_ffmpeg
def test_ingest_stage_video_and_slides(
    tmp_path: Path, video: Path, slide_pdf: Path, slide_png: Path
) -> None:
    sources = [Source(video, "video"), Source(slide_pdf, "slides"), Source(slide_png, "slides")]
    run_dir = storage.create_run(tmp_path / "runs", sources, Config())
    assert pipeline.run(run_dir, stages=INGEST) == {"ingest": "done"}

    assert sf.info(run_dir / "audio.wav").samplerate == 16000
    frames = schemas.load(run_dir, FramesFile).frames
    video_frames = [f for f in frames if f.pts is not None]
    slide_frames = [f for f in frames if f.pts is None]
    assert [round(f.pts or 0) for f in video_frames] == [0, 2, 4]
    assert [(f.source, f.page) for f in slide_frames] == [
        ("01-deck.pdf", 1),
        ("01-deck.pdf", 2),
        ("02-extra.png", None),
    ]
    assert [f.frame_id for f in frames] == list(range(6))
    for f in frames:
        with Image.open(run_dir / f.path) as img:
            assert max(img.size) <= 1600
    with Image.open(run_dir / video_frames[0].path) as img:
        assert img.size == (1600, 800)
    with Image.open(run_dir / slide_frames[2].path) as img:
        assert img.size == (600, 1600)
    assert sorted(p.name for p in (run_dir / "frames").iterdir()) == [
        "000.png",
        "001.png",
        "002.png",
        "003.png",
        "004.png",
        "005.png",
        "frames.json",
    ]
    assert pipeline.run(run_dir, stages=INGEST) == {"ingest": "skipped"}


@needs_ffmpeg
def test_ingest_dedupes_repeated_scene(tmp_path: Path, video: Path) -> None:
    """A huge dedupe distance treats every scene as a duplicate of the first frame."""
    run_dir = storage.create_run(tmp_path / "runs", [Source(video, "video")], Config())
    config = Config(ingest=IngestConfig(dedupe_max_distance=256))
    pipeline.run(run_dir, stages=INGEST, config=config)
    frames = schemas.load(run_dir, FramesFile).frames
    assert [f.pts for f in frames] == [0.0]


def text_slide(title: str, rows: list[str]) -> Image.Image:
    """A white slide with black text, like most decks (no colour change between slides)."""
    img = Image.new("RGB", (1600, 900), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 60), title, fill="black", font=ImageFont.load_default(size=72))
    for i, row in enumerate(rows):
        d.text((100, 220 + 70 * i), f"- {row}", fill="black", font=ImageFont.load_default(size=44))
    return img


@needs_ffmpeg
def test_ingest_keeps_every_text_slide_of_a_deck_recording(tmp_path: Path) -> None:
    """Phase 4b-1 eval: scene scores between white text slides are ~0.04-0.07 and their 8x8
    phashes 4-8 bits apart, so the old defaults kept only the first slide."""
    media = tmp_path / "media"
    media.mkdir()
    slides = [
        text_slide("Quarterly Budget", ["Spend plan by team", "Approved: $14,500"]),
        text_slide("Retention", ["Trial users coming back", "Week-2 return: 38%"]),
        text_slide("Support Queue", ["Ticket backlog", "Open tickets: 409"]),
    ]
    inputs: list[str] = []
    for k, img in enumerate(slides):
        img.save(media / f"s{k}.png")
        inputs += ["-loop", "1", "-framerate", "5", "-t", "2", "-i", str(media / f"s{k}.png")]
    out = media / "deck.mp4"
    run_ffmpeg(
        *inputs,
        "-f",
        "lavfi",
        "-i",
        "anullsrc=duration=6",
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
        "-map",
        "[v]",
        "-map",
        "3:a",
        "-shortest",
        str(out),
    )
    run_dir = storage.create_run(tmp_path / "runs", [Source(out, "video")], Config())
    pipeline.run(run_dir, stages=INGEST)
    frames = schemas.load(run_dir, FramesFile).frames
    assert [round(f.pts or 0) for f in frames] == [0, 2, 4]
    assert all(len(f.phash) == 64 for f in frames)  # 16x16 bits as hex


@needs_ffmpeg
def test_ingest_audio_only(tmp_path: Path, tone: Path) -> None:
    run_dir = storage.create_run(tmp_path / "runs", [Source(tone, "audio")], Config())
    pipeline.run(run_dir, stages=INGEST)
    assert schemas.load(run_dir, FramesFile).frames == []
    assert (run_dir / "audio.wav").exists()


@needs_ffmpeg
def test_ingest_missing_source(tmp_path: Path, tone: Path) -> None:
    run_dir = storage.create_run(tmp_path / "runs", [Source(tone, "audio")], Config())
    tone.unlink()
    with pytest.raises(pipeline.StageFailed, match="source missing"):
        pipeline.run(run_dir, stages=INGEST)


def test_missing_tools_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda tool: None if tool == "ffprobe" else "/x")
    assert ingest.missing_tools() == ["ffprobe"]
    with pytest.raises(ingest.ToolMissing, match="ffprobe not found"):
        ingest.require("ffprobe")
