"""Shared fixtures. Media is generated on the fly into tmp_path; nothing binary is committed."""

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed",
)


@pytest.fixture(autouse=True)
def _isolated_runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never touch the real runs/ (PLAN §12)."""
    monkeypatch.setenv("COHR_RUNS_DIR", str(tmp_path / "runs"))


def pattern(kind: int, size: tuple[int, int] = (2000, 1000)) -> Image.Image:
    """Visually distinct test images (solid colours all share one phash, so draw shapes)."""
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    w, h = size
    if kind == 0:  # checkerboard
        step = w // 10
        for x in range(0, w, step):
            for y in range(0, h, step):
                if (x // step + y // step) % 2:
                    d.rectangle([x, y, x + step, y + step], fill="black")
    elif kind == 1:  # vertical stripes on the left half
        for x in range(0, w // 2, w // 20):
            d.rectangle([x, 0, x + w // 40, h], fill="navy")
    else:  # big circle, bottom right
        d.ellipse([w // 2, h // 3, w - 10, h - 10], fill="darkred")
    return img


def run_ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True
    )


@pytest.fixture
def tone(tmp_path: Path) -> Path:
    """3 s stereo 44.1 kHz AAC, so normalizing has real work to do."""
    out = tmp_path / "media" / "tone.m4a"
    out.parent.mkdir(exist_ok=True)
    run_ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=44100", "-ac", "2", str(out)
    )
    return out


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """6 s: three 2 s scenes (patterns 0, 1, 2) at 2000x1000, with a tone."""
    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    pngs = []
    for k in range(3):
        p = media / f"scene{k}.png"
        pattern(k).save(p)
        pngs.append(p)
    out = media / "screen.mp4"
    inputs: list[str] = []
    for p in pngs:
        inputs += ["-loop", "1", "-framerate", "10", "-t", "2", "-i", str(p)]
    run_ffmpeg(
        *inputs,
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=330:duration=6",
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
        "-map",
        "[v]",
        "-map",
        "3:a",
        "-shortest",
        str(out),
    )
    return out


@pytest.fixture
def slide_pdf(tmp_path: Path) -> Path:
    out = tmp_path / "media" / "deck.pdf"
    out.parent.mkdir(exist_ok=True)
    first, *rest = (pattern(k, (800, 600)) for k in (2, 0))
    first.save(out, save_all=True, append_images=rest)
    return out


@pytest.fixture
def slide_png(tmp_path: Path) -> Path:
    """Portrait, taller than the cap, to check the long-edge downscale."""
    out = tmp_path / "media" / "extra.png"
    out.parent.mkdir(exist_ok=True)
    pattern(1, (900, 2400)).save(out)
    return out
