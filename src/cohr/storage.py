"""Run directories on local disk (PLAN §12): create, sources, disk checks, size, prune, delete."""

import hashlib
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from cohr import schemas
from cohr.config import Config
from cohr.schemas import InputFile, InputKind, Manifest, ModelRef

GB = 1024**3
MB = 1024**2
AUDIO_BYTES_PER_HOUR = 200 * MB  # estimate for audio.wav + frames (§12)
RUN_MARGIN_BYTES = 1 * GB


class DiskSpaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Source:
    path: Path
    kind: InputKind


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "run"


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def estimate_run_bytes(sources: list[Source], mode: str, duration_s: float | None) -> int:
    copied = sum(s.path.stat().st_size for s in sources) if mode == "copy" else 0
    hours = (duration_s or 0.0) / 3600
    return copied + int(hours * AUDIO_BYTES_PER_HOUR)


def check_free_space(directory: Path, needed: int, margin: int = RUN_MARGIN_BYTES) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < needed + margin:
        raise DiskSpaceError(
            f"Not enough disk space in {directory}: need {(needed + margin) / GB:.1f} GB "
            f"(including a {margin / GB:.0f} GB margin), {free / GB:.1f} GB free."
        )


def _stored_name(index: int, path: Path) -> str:
    return f"{index:02d}-{path.name}"


def create_run(
    runs_dir: Path,
    sources: list[Source],
    config: Config,
    *,
    name: str | None = None,
    duration_s: float | None = None,
    now: datetime | None = None,
) -> Path:
    """Make runs/<YYYYmmdd-HHMMSS>-<slug>/ with source/ and a fresh manifest."""
    if not sources:
        raise ValueError("a run needs at least one input")
    for s in sources:
        if not s.path.is_file():
            raise FileNotFoundError(s.path)
    check_free_space(runs_dir, estimate_run_bytes(sources, config.source_mode, duration_s))

    now = now or datetime.now().astimezone()  # local time: run ids are what people scan
    base = f"{now:%Y%m%d-%H%M%S}-{slugify(name or sources[0].path.stem)}"
    run_dir = runs_dir / base
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = runs_dir / f"{base}-{n}"
    (run_dir / "source").mkdir(parents=True)

    inputs = []
    for i, s in enumerate(sources):
        original = s.path.resolve()
        stored = _stored_name(i, original)
        target = run_dir / "source" / stored
        if config.source_mode == "copy":
            shutil.copy2(original, target)
        else:
            target.symlink_to(original)
        st = original.stat()
        inputs.append(
            InputFile(
                kind=s.kind,
                path=str(original),
                stored=stored,
                mode=config.source_mode,
                size=st.st_size,
                mtime=st.st_mtime,
                sha256=sha256_file(original),
            )
        )

    manifest = Manifest(
        app_version=version("cohr"),
        run_id=run_dir.name,
        created_at=now,
        inputs=inputs,
        config=config.model_dump(mode="json"),
        models={k: ModelRef(repo=m.repo, revision=m.revision) for k, m in config.models.items()},
    )
    schemas.save(run_dir, manifest)
    return run_dir


def is_run(path: Path) -> bool:
    return (path / Manifest.FILENAME).is_file()


def list_runs(runs_dir: Path) -> list[Path]:
    """Newest first."""
    if not runs_dir.is_dir():
        return []
    return sorted((p for p in runs_dir.iterdir() if is_run(p)), reverse=True)


def run_size(run_dir: Path) -> int:
    """Bytes on disk, not following symlinks (linked sources don't count)."""
    total = 0
    for p in run_dir.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total


def _check_inside(run_dir: Path, runs_dir: Path | None) -> None:
    if not is_run(run_dir):
        raise ValueError(f"{run_dir} is not a run directory (no manifest.json)")
    if runs_dir is not None and runs_dir.resolve() not in run_dir.resolve().parents:
        raise ValueError(f"{run_dir} is not inside {runs_dir}")


def delete_run(run_dir: Path, runs_dir: Path | None = None) -> None:
    _check_inside(run_dir, runs_dir)
    shutil.rmtree(run_dir)


def prune_media(run_dir: Path, runs_dir: Path | None = None) -> int:
    """Remove source/, audio.wav and frame images; keep every JSON file and the notes.

    Returns the bytes freed. Notes, Ask and re-Reduce keep working (§12).
    """
    _check_inside(run_dir, runs_dir)
    before = run_size(run_dir)
    shutil.rmtree(run_dir / "source", ignore_errors=True)
    (run_dir / "audio.wav").unlink(missing_ok=True)
    frames = run_dir / "frames"
    if frames.is_dir():
        for p in frames.iterdir():
            if p.suffix.lower() != ".json":
                p.unlink()
    manifest = schemas.load(run_dir, Manifest)
    schemas.save(run_dir, manifest.model_copy(update={"pruned": True}))
    return before - run_size(run_dir)
