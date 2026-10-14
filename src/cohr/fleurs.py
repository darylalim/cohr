"""FLEURS test clips for the `full` eval set (PLAN §18, 6e-1): Transcribe alone, no VAD.

google/fleurs keeps each language's test audio in one tar (159–658 MB), so the sample is drawn
from the first `pool` tar members and only the start of each tar is streamed: one clip per
sentence (the TSV repeats a sentence per speaker), `clips` of them chosen by `seed`. The
chosen WAVs and sample.json stay in evals/data/fleurs/<lang>/ (gitignored, never committed);
a sample whose spec matches is reused without the network. Scoring is in evals.py.
"""

import random
import tarfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Protocol

from pydantic import BaseModel, ConfigDict

from cohr import atomic

REPO = "google/fleurs"
DATA_DIR = Path("evals") / "data" / "fleurs"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FleursSpec(_Model):
    """The `fleurs:` block of an eval set (evals/sets/full.yaml)."""

    revision: str  # pinned dataset revision
    seed: int = 1
    clips: int = 20  # per language
    pool: int = 100  # tar members read per language (the sample is drawn from these)
    languages: dict[str, str]  # Transcribe language → FLEURS config (zh → cmn_hans_cn)


class Clip(_Model):
    file: str  # e.g. 1003119935936341070.wav
    sentence_id: int
    text: str  # raw transcription (cased, punctuated)


class Sample(_Model):
    language: str
    config: str
    revision: str
    seed: int
    clips: int
    pool: int
    items: list[Clip]


class FileSystem(Protocol):
    """The part of huggingface_hub.HfFileSystem used here (a fake in tests)."""

    def read_text(self, path: str, encoding: str | None = None) -> str: ...

    def open(self, path: str, mode: str = "rb") -> Any: ...


def parse_tsv(text: str) -> dict[str, Clip]:
    """test.tsv rows by file name. Columns: sentence id, file, raw text, normalized text,
    characters, samples, gender."""
    rows = {}
    for line in text.splitlines():
        cols = line.split("\t")
        if len(cols) >= 3 and cols[0].isdigit():
            rows[cols[1]] = Clip(file=cols[1], sentence_id=int(cols[0]), text=cols[2].strip())
    return rows


def read_members(fileobj: IO[bytes], n: int) -> dict[str, bytes]:
    """The first `n` files of a streamed .tar.gz, by base name, in tar order."""
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        for member in tar:
            if len(out) >= n:
                break
            if member.isfile() and (f := tar.extractfile(member)) is not None:
                out[Path(member.name).name] = f.read()
    return out


def pick(members: Sequence[str], rows: Mapping[str, Clip], n: int, seed: int) -> list[Clip]:
    """`n` clips by seed, one per sentence (its first member in tar order), in sentence order."""
    first: dict[int, Clip] = {}
    for name in members:
        if (row := rows.get(name)) is not None:
            first.setdefault(row.sentence_id, row)
    if len(first) < n:
        raise ValueError(f"only {len(first)} sentences in the first {len(members)} clips, need {n}")
    chosen = random.Random(seed).sample(sorted(first), n)
    return [first[s] for s in sorted(chosen)]


def sample_dir(language: str, root: Path = Path()) -> Path:
    return root / DATA_DIR / language


def load_sample(
    spec: FleursSpec,
    language: str,
    root: Path = Path(),
    *,
    fs: FileSystem | None = None,
    log: Callable[[str], None] = print,
) -> tuple[Sample, Path]:
    """The language's sample and its folder, streamed from the Hub unless already kept."""
    folder = sample_dir(language, root)
    path = folder / "sample.json"
    want = {
        "language": language,
        "config": spec.languages[language],
        "revision": spec.revision,
        "seed": spec.seed,
        "clips": spec.clips,
        "pool": spec.pool,
    }
    if path.is_file():
        kept = Sample.model_validate_json(path.read_bytes())
        if kept.model_dump(exclude={"items"}) == want and all(
            (folder / c.file).is_file() for c in kept.items
        ):
            return kept, folder
    if fs is None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
    base = f"datasets/{REPO}@{spec.revision}/data/{want['config']}"
    log(f"fleurs-{language}: streaming the first {spec.pool} test clips ({want['config']})")
    rows = parse_tsv(fs.read_text(f"{base}/test.tsv", encoding="utf-8"))
    with fs.open(f"{base}/audio/test.tar.gz", "rb") as f:
        audio = read_members(f, spec.pool)
    chosen = pick(list(audio), rows, spec.clips, spec.seed)
    for clip in chosen:
        atomic.write_bytes(folder / clip.file, audio[clip.file])
    sample = Sample.model_validate({**want, "items": chosen})
    atomic.write_text(path, sample.model_dump_json(indent=2) + "\n")
    return sample, folder
