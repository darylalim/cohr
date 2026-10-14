"""Ingest (PLAN §5.1): ffmpeg + PIL + pypdfium2, no models.

Outputs: audio.wav (16 kHz mono PCM) and frames/ (NNN.png + frames.json). Video frames are
taken at scene changes with their pts, then deduplicated by perceptual hash; slide files (PDF
pages, images) follow with no timestamp.
"""

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imagehash
import pypdfium2
from PIL import Image

from cohr import atomic, schemas
from cohr.pipeline import Stage, StageContext
from cohr.schemas import Frame, FramesFile, InputFile

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_SHOWINFO = re.compile(r"\bn:\s*\d+\b.*?\bpts_time:\s*(-?[\d.]+)")


class IngestError(RuntimeError):
    pass


class ToolMissing(IngestError):
    """ffmpeg or ffprobe isn't installed: not a problem with the recording."""


TOOLS = ("ffmpeg", "ffprobe")  # both come with `brew install ffmpeg`


def require(tool: str = "ffmpeg") -> str:
    path = shutil.which(tool)
    if path is None:
        raise ToolMissing(f"{tool} not found on PATH. Install it with `brew install ffmpeg`.")
    return path


def missing_tools() -> list[str]:
    return [t for t in TOOLS if shutil.which(t) is None]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise IngestError(f"{Path(cmd[0]).name} failed ({proc.returncode}): {tail}")
    return proc


@dataclass(frozen=True)
class Probe:
    duration_s: float | None
    has_audio: bool
    has_video: bool


def probe(path: Path) -> Probe:
    proc = _run(
        [
            require("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type:stream_disposition=attached_pic",
            "-of",
            "json",
            str(path),
        ]
    )
    info = json.loads(proc.stdout)
    streams = info.get("streams", [])
    duration = info.get("format", {}).get("duration")
    return Probe(
        duration_s=float(duration) if duration not in (None, "N/A") else None,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        # Cover art in audio files is a "video" stream with attached_pic=1; not a video.
        has_video=any(
            s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
            for s in streams
        ),
    )


def normalize_audio(src: Path, dest: Path) -> None:
    """ffmpeg -i src -ac 1 -ar 16000 -c:a pcm_s16le dest (§5.1)."""
    with atomic.atomic_path(dest) as tmp:
        _run(
            [
                require(),
                "-nostdin",
                "-hide_banner",
                "-y",
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(tmp),
            ]
        )


def scene_filter(threshold: float, max_edge: int) -> str:
    """Keep the first frame and every scene change; cap the long edge (portrait-safe)."""
    select = f"select='eq(n\\,0)+gt(scene\\,{threshold})'"
    w = f"if(gte(iw\\,ih)\\,min({max_edge}\\,iw)\\,-2)"
    h = f"if(gte(iw\\,ih)\\,-2\\,min({max_edge}\\,ih))"
    return f"{select},scale='{w}':'{h}',showinfo"


def parse_showinfo(stderr: str) -> list[float]:
    """pts_time of each frame that passed the filter chain, in output order."""
    return [
        float(m.group(1))
        for line in stderr.splitlines()
        if "showinfo" in line and (m := _SHOWINFO.search(line))
    ]


def scene_frames(
    video: Path, out_dir: Path, threshold: float, max_edge: int
) -> list[tuple[Path, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = _run(
        [
            require(),
            "-nostdin",
            "-hide_banner",
            "-i",
            str(video),
            "-an",
            "-vf",
            scene_filter(threshold, max_edge),
            "-fps_mode",
            "vfr",
            str(out_dir / "raw_%05d.png"),
        ]
    )
    files = sorted(out_dir.glob("raw_*.png"))
    pts = parse_showinfo(proc.stderr)
    if len(pts) != len(files):
        raise IngestError(f"ffmpeg wrote {len(files)} frames but reported {len(pts)} timestamps")
    return list(zip(files, pts, strict=True))


def dedupe(hashes: list[imagehash.ImageHash], max_distance: int) -> list[int]:
    """Indices to keep: drop a frame within `max_distance` of the last kept frame (§5.1)."""
    keep: list[int] = []
    for i, h in enumerate(hashes):
        if not keep or h - hashes[keep[-1]] > max_distance:
            keep.append(i)
    return keep


def fit(img: Image.Image, max_edge: int) -> Image.Image:
    if max(img.size) <= max_edge:
        return img
    scale = max_edge / max(img.size)
    return img.resize(
        (round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS
    )


def pdf_pages(pdf: Path, max_edge: int) -> list[Image.Image]:
    doc = pypdfium2.PdfDocument(pdf)
    try:
        pages = []
        for page in doc:
            w, h = page.get_size()  # points
            pages.append(page.render(scale=max_edge / max(w, h)).to_pil().convert("RGB"))
        return pages
    finally:
        doc.close()


def slide_images(path: Path, max_edge: int) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_pages(path, max_edge)
    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as img:
            return [fit(img.convert("RGB"), max_edge)]
    raise IngestError(f"unsupported slide file {path.name}: use PDF or an image")


def _media_input(inputs: list[InputFile]) -> InputFile:
    media = [i for i in inputs if i.kind in ("audio", "video")]
    if len(media) != 1:
        raise IngestError(f"a run needs exactly one audio or video input, got {len(media)}")
    return media[0]


def run_stage(ctx: StageContext) -> None:
    cfg = ctx.config.ingest
    run_dir = ctx.run_dir
    source = run_dir / "source"
    media = _media_input(ctx.manifest.inputs)
    media_path = source / media.stored
    if not media_path.exists():
        raise IngestError(f"source missing: {media.path} (moved or deleted?)")
    info = probe(media_path)
    if not info.has_audio:
        raise IngestError(f"{media.stored} has no audio stream")

    ctx.report(0.05, "normalizing audio")
    normalize_audio(media_path, run_dir / "audio.wav")

    frames: list[Frame] = []
    with atomic.atomic_dir(run_dir / "frames") as tmp:
        if media.kind == "video" and info.has_video:
            ctx.report(0.3, "extracting scene frames")
            raw = scene_frames(media_path, tmp / ".raw", cfg.scene_threshold, cfg.max_edge_px)
            hashes = []
            for path, _ in raw:
                ctx.check_cancel()
                with Image.open(path) as img:
                    hashes.append(imagehash.phash(img, hash_size=cfg.phash_size))
            for i in dedupe(hashes, cfg.dedupe_max_distance):
                path, pts = raw[i]
                fid = len(frames)
                path.rename(tmp / f"{fid:03d}.png")
                frames.append(
                    Frame(
                        frame_id=fid,
                        path=f"frames/{fid:03d}.png",
                        pts=pts,
                        phash=str(hashes[i]),
                        source=media.stored,
                    )
                )
            shutil.rmtree(tmp / ".raw")
            ctx.log.info("ingest: %d scene frames, %d after dedupe", len(raw), len(frames))

        slides = [i for i in ctx.manifest.inputs if i.kind == "slides"]
        for n, item in enumerate(slides):
            ctx.report(0.6 + 0.4 * n / len(slides), f"rendering {item.stored}")
            path = source / item.stored
            pdf = path.suffix.lower() == ".pdf"
            for page, img in enumerate(slide_images(path, cfg.max_edge_px), start=1):
                ctx.check_cancel()
                fid = len(frames)
                img.save(tmp / f"{fid:03d}.png")
                frames.append(
                    Frame(
                        frame_id=fid,
                        path=f"frames/{fid:03d}.png",
                        pts=None,
                        phash=str(imagehash.phash(img, hash_size=cfg.phash_size)),
                        source=item.stored,
                        page=page if pdf else None,
                    )
                )
    schemas.save(run_dir, FramesFile(frames=frames))


STAGE = Stage(
    name="ingest",
    run=run_stage,
    outputs=("audio.wav", FramesFile.FILENAME),
    format_version=FramesFile.FORMAT_VERSION,
    config_keys=("ingest",),
)
