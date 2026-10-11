"""Atomic file output (PLAN §3): write to a temp file in the same directory, then os.replace.

A crash or cancel never leaves a half-written file that the skip logic would trust.
"""

import contextlib
import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


# mkstemp/mkdtemp create 0600/0700; give outputs the modes a plain open()/mkdir() would.
_FILE_MODE = 0o666 & ~_umask()
_DIR_MODE = 0o777 & ~_umask()


@contextlib.contextmanager
def atomic_path(dest: Path) -> Generator[Path]:
    """Yield a temp path next to `dest`; on success it replaces `dest`, on error it's removed.

    The temp name keeps `dest`'s suffix so tools that pick a format by extension (ffmpeg) work.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=dest.suffix)
    os.close(fd)
    tmp_path = Path(tmp)
    tmp_path.chmod(_FILE_MODE)
    try:
        yield tmp_path
        os.replace(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)


@contextlib.contextmanager
def atomic_dir(dest: Path) -> Generator[Path]:
    """Yield an empty temp directory; on success it replaces `dest` as a whole."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}."))
    tmp.chmod(_DIR_MODE)
    try:
        yield tmp
        if dest.exists():
            old = dest.with_name(f".{dest.name}.old")
            shutil.rmtree(old, ignore_errors=True)
            os.replace(dest, old)
            os.replace(tmp, dest)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(tmp, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_bytes(dest: Path, data: bytes) -> None:
    with atomic_path(dest) as tmp:
        tmp.write_bytes(data)


def write_text(dest: Path, text: str) -> None:
    write_bytes(dest, text.encode("utf-8"))
