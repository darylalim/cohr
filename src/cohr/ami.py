"""AMI meetings for the `full` eval set (PLAN §18, 6e-1): real meetings, audio only, English.

Audio is the official mirror's headset mix (one 16 kHz WAV per meeting, the whole meeting);
reference summaries come from the official annotations zip (`abstractive/<id>.abssumm.xml`).
Both are CC BY 4.0, fetched into evals/data/ami/ (gitignored, never committed) and kept.
Scoring is in evals.py: the judge's coverage per reference sentence.
"""

import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from cohr import atomic

AUDIO_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{id}/audio/{id}.Mix-Headset.wav"
)
ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
DATA_DIR = Path("evals") / "data" / "ami"
ANNOTATIONS = DATA_DIR / "ami_public_manual_1.6.2.zip"
# The parts of abssumm.xml, each a list of <sentence> (evals.AMI_HOME: where they are looked for).
PARTS = ("abstract", "decisions", "actions", "problems")


class AmiSpec(BaseModel):
    """The `ami:` block of an eval set (evals/sets/full.yaml)."""

    model_config = ConfigDict(extra="forbid")

    meetings: list[str]


def parse_summary(data: bytes) -> dict[str, list[str]]:
    """Reference sentences per part (abstract, decisions, actions, problems), in order.
    The XML declares ISO-8859-1; ElementTree decodes it from the bytes."""
    root = ET.fromstring(data)
    out: dict[str, list[str]] = {part: [] for part in PARTS}
    for part in PARTS:
        for el in root.iter(part):
            out[part] += [
                t for s in el.iter("sentence") if (t := " ".join("".join(s.itertext()).split()))
            ]
    return out


def reference(meeting: str, zip_path: Path) -> dict[str, list[str]]:
    with zipfile.ZipFile(zip_path) as z:
        return parse_summary(z.read(f"abstractive/{meeting}.abssumm.xml"))


def download(url: str, dest: Path, log: Callable[[str], None] = print) -> Path:
    """Fetch `url` to `dest` atomically unless it's there (a partial download never is)."""
    if dest.is_file():
        return dest
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, atomic.atomic_path(dest) as tmp:
        with tmp.open("wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        expected = resp.headers.get("Content-Length")
        if expected is not None and tmp.stat().st_size != int(expected):
            raise OSError(f"{url}: got {tmp.stat().st_size} of {expected} bytes")
    return dest


def ensure(
    meeting: str, root: Path = Path(), log: Callable[[str], None] = print
) -> tuple[Path, dict[str, list[str]]]:
    """The meeting's audio and its reference sentences, downloading what's missing."""
    zip_path = download(ANNOTATIONS_URL, root / ANNOTATIONS, log)
    refs = reference(meeting, zip_path)  # first: a meeting without a summary fails before audio
    wav = download(
        AUDIO_URL.format(id=meeting), root / DATA_DIR / f"{meeting}.Mix-Headset.wav", log
    )
    return wav, refs
