"""Transcribe (PLAN §5.2): Cohere Transcribe on mlx-speech (D1), one chunk at a time.

Each chunk is read from audio.wav by its chunks.json offsets (no chunk WAVs). The source
language is the user's choice: neither runtime auto-detects it (D3).
"""

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from cohr import schemas
from cohr.pipeline import Stage, StageContext
from cohr.schemas import ChunksFile, Transcript, TranscriptItem

SR = 16000
RUNTIME = "mlx-speech"
LANGUAGES = ("ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh")


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
    if chunks:
        with ctx.model("transcribe") as asr:
            for i, c in enumerate(chunks):
                ctx.report(i / len(chunks), f"chunk {i + 1}/{len(chunks)}")
                text = asr.transcribe(read_chunk(wav, c.t0, c.t1), language).strip()
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
