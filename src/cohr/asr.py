"""Transcribe (PLAN §5.2): Cohere Transcribe on mlx-speech (D1), one chunk at a time.

Each chunk is read from audio.wav by its chunks.json offsets (no chunk WAVs). The source
language is the user's choice: neither runtime auto-detects it (D3).

D48 (8i): a reply with no sentence punctuation before its end can be missing its opening
line (xling-ja chunks 5, 7, 15: "先月は7700ドルでした" never written). The chunk's first span
up to a pause is then decoded alone, and put in front of the reply when the reply doesn't
hold it.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from cohr import schemas, vad
from cohr.pipeline import Stage, StageContext
from cohr.schemas import ChunksFile, Transcript, TranscriptItem

SR = 16000
RUNTIME = "mlx-speech"
LANGUAGES = ("ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh")
NO_SPACE = ("ja", "zh")  # written without spaces between words (as mlx-speech joins them)

# D48. Every xling-ja chunk that lost its opening had no 。 inside; AMI's unpunctuated
# chunks (87 in 5 meetings) lose no more words than punctuated ones, and the probe, gated on
# them, fired on 7, 5 of them real lost speech by AMI's reference words.
SENTENCE_MARKS = ".?!。？！؟"
PROBE_MIN_CHARS = 40  # shorter replies are often one sentence with nothing to lose
PROBE_MIN_SPEECH_S = 1.0  # the first span holds at least this much speech…
PROBE_MIN_PAUSE_S = 0.4  # …and ends at the first pause this long (TTS lines: 0.6 s apart)
PROBE_TAIL_S = 1.0  # a span ending this close to the chunk's end is the whole chunk
PROBE_MIN_COVER = 0.6  # xling-ja: lost openings 0.08–0.33 of the span found, kept ones ≥ 0.81
_WORD = re.compile(r"[^\W_]+")
_CHAR = re.compile(r"[^\W_]")  # NO_SPACE languages compare characters


class AsrError(ValueError):
    pass


class CohereAsr:
    def __init__(self, model: Any):
        self.model = model

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        result = self.model.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32), sample_rate=SR, language=language
        )
        return result.text


def load(path: Path) -> CohereAsr:
    from mlx_speech.generation import CohereAsrModel

    return CohereAsr(CohereAsrModel.from_path(path))


def read_chunk(wav: Path, t0: float, t1: float) -> np.ndarray:
    audio, sr = sf.read(wav, start=round(t0 * SR), stop=round(t1 * SR), dtype="float32")
    if sr != SR:
        raise AsrError(f"audio.wav must be 16 kHz, got {sr} Hz")
    return audio


def noise_threshold_db(wav: Path) -> float:
    """`vad.noise_threshold_db` of the whole file, read one second at a time."""
    levels = [
        float(vad.frame_db(block, 1.0)[0])
        for block in sf.blocks(wav, blocksize=SR, dtype="float32")
        if len(block) == SR
    ]
    if not levels:
        return -60.0
    return float(np.percentile(levels, vad.NOISE_FLOOR_PERCENTILE)) + vad.NOISE_MARGIN_DB


def unpunctuated(text: str) -> bool:
    """A reply long enough to hold sentences with no sentence mark before its last character."""
    return len(text) > PROBE_MIN_CHARS and not any(c in SENTENCE_MARKS for c in text[:-1])


def first_span_end(audio: np.ndarray, threshold_db: float) -> float | None:
    """Where the chunk's first span ends: the first pause ≥ PROBE_MIN_PAUSE_S after at least
    PROBE_MIN_SPEECH_S from its first loud frame (energy, as the silence fallback). None when
    there's no such pause, or it's within PROBE_TAIL_S of the end."""
    hop = vad.ENERGY_HOP_S
    loud = vad.frame_db(audio, hop) >= threshold_db
    if not loud.any():
        return None
    start = int(np.argmax(loud))
    need, quiet = round(PROBE_MIN_PAUSE_S / hop), 0
    for i in range(start + round(PROBE_MIN_SPEECH_S / hop), len(loud)):
        quiet = quiet + 1 if not loud[i] else 0
        if quiet >= need:
            end = (i - quiet + 1) * hop
            return end if end < len(audio) / SR - PROBE_TAIL_S else None
    return None


def _units(text: str, language: str) -> list[re.Match[str]]:
    return list((_CHAR if language in NO_SPACE else _WORD).finditer(text))


def with_opening(probe: str, reply: str, language: str) -> str | None:
    """`reply` with the probe's text in front when the reply's head holds less than
    PROBE_MIN_COVER of it (in order), else None. A probe ending in the reply's own first words
    ("b uh tha that button" before "That button and …") gives only what comes before them."""
    pu, ru = _units(probe, language), _units(reply, language)
    if not pu:
        return None
    p, r = [m.group(0).casefold() for m in pu], [m.group(0).casefold() for m in ru]
    head = r[: round(len(p) * 1.5) + 2]
    found = sum(
        b.size for b in SequenceMatcher(None, p, head, autojunk=False).get_matching_blocks()
    )
    if found / len(p) >= PROBE_MIN_COVER:
        return None
    keep = len(p)
    for k in range(min(len(p), len(r)), 0, -1):
        if p[-k:] == r[:k]:
            keep = len(p) - k
            break
    if keep == 0:
        return None
    opening = (probe[: pu[keep].start()] if keep < len(p) else probe).strip()
    return opening + ("" if language in NO_SPACE else " ") + reply


def run_stage(ctx: StageContext) -> None:
    language = ctx.config.source_language
    if language not in LANGUAGES:
        raise AsrError(
            f"Transcribe doesn't support source language {language!r}; "
            f"choose one of {', '.join(LANGUAGES)}"
        )
    chunks = schemas.load(ctx.run_dir, ChunksFile).chunks
    wav = ctx.run_dir / "audio.wav"
    items: list[TranscriptItem] = []
    threshold: float | None = None
    if chunks:
        with ctx.model("transcribe") as asr:
            for i, c in enumerate(chunks):
                ctx.report(i / len(chunks), f"chunk {i + 1}/{len(chunks)}")
                audio = read_chunk(wav, c.t0, c.t1)
                text = asr.transcribe(audio, language).strip()
                if unpunctuated(text):
                    threshold = noise_threshold_db(wav) if threshold is None else threshold
                    end = first_span_end(audio, threshold)
                    if end is not None:
                        probe = asr.transcribe(audio[: round(end * SR)], language).strip()
                        if (fixed := with_opening(probe, text, language)) is not None:
                            ctx.log.info(
                                "transcribe: chunk %d opening restored: %s",
                                c.chunk_id,
                                fixed[: len(fixed) - len(text)].strip(),
                            )
                            text = fixed
                items.append(TranscriptItem(chunk_id=c.chunk_id, t0=c.t0, t1=c.t1, text=text))
    ctx.log.info(
        "transcribe: %d chunks, %d characters", len(items), sum(len(i.text) for i in items)
    )
    schemas.save(ctx.run_dir, Transcript(language=language, runtime=RUNTIME, items=items))


STAGE = Stage(
    name="transcribe",
    run=run_stage,
    outputs=(Transcript.FILENAME,),
    format_version=Transcript.FORMAT_VERSION,
    config_keys=("source_language", "models.transcribe"),
    models=("transcribe",),
)
