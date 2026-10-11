"""PLAN §12: run dirs, link vs copy, disk-space refusal, size, prune keeps notes usable, delete."""

import hashlib
import shutil
from collections import namedtuple
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cohr import schemas, storage
from cohr.config import Config
from cohr.schemas import Manifest
from cohr.storage import DiskSpaceError, Source

NOW = datetime(2026, 10, 11, 9, 30, 5, tzinfo=UTC)


@pytest.fixture
def media(tmp_path: Path) -> Path:
    p = tmp_path / "Team Sync (Oct).m4a"
    p.write_bytes(b"x" * 1000)
    return p


def test_create_run_links_sources(tmp_path: Path, media: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = storage.create_run(runs, [Source(media, "audio")], Config(), now=NOW)
    assert run_dir.name == "20261011-093005-team-sync-oct"
    link = run_dir / "source" / "00-Team Sync (Oct).m4a"
    assert link.is_symlink() and link.resolve() == media.resolve()
    m = schemas.load(run_dir, Manifest)
    (item,) = m.inputs
    assert (item.kind, item.mode, item.size) == ("audio", "link", 1000)
    assert item.sha256 == hashlib.sha256(b"x" * 1000).hexdigest()
    assert m.config["seed"] == 0
    assert m.models["transcribe"].revision == "a0acb7f93cd32d82c4fbf801b6d8fb39d20c509f"
    assert m.stages == {} and not m.pruned


def test_create_run_copy_mode_and_name_collision(tmp_path: Path, media: Path) -> None:
    runs = tmp_path / "runs"
    cfg = Config(source_mode="copy")
    a = storage.create_run(runs, [Source(media, "audio")], cfg, name="Q4 plan", now=NOW)
    b = storage.create_run(runs, [Source(media, "audio")], cfg, name="Q4 plan", now=NOW)
    assert (a.name, b.name) == ("20261011-093005-q4-plan", "20261011-093005-q4-plan-2")
    copied = b / "source" / f"00-{media.name}"
    assert copied.is_file() and not copied.is_symlink()
    assert storage.list_runs(runs) == [b, a]


def test_create_run_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        storage.create_run(tmp_path / "runs", [Source(tmp_path / "nope.wav", "audio")], Config())


def test_disk_space_refusal(tmp_path: Path, media: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: Usage(10, 10, 500 * storage.MB))
    with pytest.raises(DiskSpaceError, match="need 1.2 GB"):
        storage.create_run(tmp_path / "runs", [Source(media, "audio")], Config(), duration_s=3600)
    assert not any((tmp_path / "runs").iterdir())


def test_estimate_counts_copies_only_in_copy_mode(media: Path) -> None:
    src = [Source(media, "audio")]
    assert storage.estimate_run_bytes(src, "link", 1800) == storage.AUDIO_BYTES_PER_HOUR // 2
    assert storage.estimate_run_bytes(src, "copy", None) == 1000


def _populate(run_dir: Path) -> None:
    (run_dir / "audio.wav").write_bytes(b"a" * 5000)
    (run_dir / "frames").mkdir()
    (run_dir / "frames" / "000.png").write_bytes(b"p" * 3000)
    schemas.save(run_dir, schemas.FramesFile(frames=[]))
    for name in ("transcript.json", "segments.json", "map.json", "slides.json", "notes.json"):
        (run_dir / name).write_text("{}")
    (run_dir / "notes.md").write_text("# notes")
    (run_dir / "notes_raw.txt").write_text("raw")


def test_prune_keeps_notes_usable(tmp_path: Path, media: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = storage.create_run(runs, [Source(media, "audio")], Config(source_mode="copy"))
    _populate(run_dir)
    before = storage.run_size(run_dir)
    freed = storage.prune_media(run_dir, runs)
    assert freed == before - storage.run_size(run_dir) and freed >= 9000
    assert not (run_dir / "source").exists()
    assert not (run_dir / "audio.wav").exists()
    assert sorted(p.name for p in (run_dir / "frames").iterdir()) == ["frames.json"]
    for name in (
        "transcript.json",
        "segments.json",
        "map.json",
        "slides.json",
        "notes.json",
        "notes.md",
    ):
        assert (run_dir / name).exists(), name
    assert schemas.load(run_dir, Manifest).pruned
    assert media.exists(), "prune must never touch the original"


def test_prune_linked_run_leaves_original(tmp_path: Path, media: Path) -> None:
    run_dir = storage.create_run(tmp_path / "runs", [Source(media, "audio")], Config())
    storage.prune_media(run_dir)
    assert media.read_bytes() == b"x" * 1000


def test_run_size_ignores_linked_sources(tmp_path: Path, media: Path) -> None:
    linked = storage.create_run(tmp_path / "runs", [Source(media, "audio")], Config(), name="l")
    copied = storage.create_run(
        tmp_path / "runs", [Source(media, "audio")], Config(source_mode="copy"), name="c"
    )
    assert storage.run_size(copied) - storage.run_size(linked) == 1000


def test_delete_run_safety(tmp_path: Path, media: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = storage.create_run(runs, [Source(media, "audio")], Config())
    with pytest.raises(ValueError, match="not a run"):
        storage.delete_run(tmp_path)
    with pytest.raises(ValueError, match="not inside"):
        storage.delete_run(run_dir, tmp_path / "elsewhere")
    storage.delete_run(run_dir, runs)
    assert not run_dir.exists() and media.exists()


@pytest.mark.parametrize(
    ("text", "slug"),
    [("Team Sync (Oct)", "team-sync-oct"), ("会議", "run"), ("a" * 60, "a" * 40), ("--x--", "x")],
)
def test_slugify(text: str, slug: str) -> None:
    assert storage.slugify(text) == slug
