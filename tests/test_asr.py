"""PLAN §5.2: transcribe reads each chunk from audio.wav by its offsets; language checks."""

from pathlib import Path

import numpy as np
import pytest

from cohr import asr, schemas
from cohr.asr import AsrError
from cohr.config import Config
from cohr.schemas import Chunk, ChunksFile, Span, Transcript
from tests.conftest import ctx_for, fake_manager, make_run

SR = 16000


class FakeAsr:
    """Reports the first sample (a ramp encodes the time) and the length of each chunk."""

    def __init__(self) -> None:
        self.languages: list[str] = []

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        self.languages.append(language)
        return f" start={audio[0] * 100:.2f} n={len(audio)} "


def _ramp(seconds: int) -> np.ndarray:
    # Sample value = time / 100 s, exact enough in 16-bit PCM to read offsets back.
    return (np.arange(seconds * SR) / SR / 100).astype(np.float32)


def test_transcribe_slices_audio_by_chunk_offsets(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, _ramp(30), Config(source_language="de"))
    chunks = [Chunk(chunk_id=0, t0=2.0, t1=5.0), Chunk(chunk_id=1, t0=10.5, t1=12.0)]
    schemas.save(run_dir, ChunksFile(chunker="vad", chunks=chunks, gaps=[Span(t0=0, t1=2)]))
    fake = FakeAsr()
    asr.run_stage(ctx_for(run_dir, "transcribe", fake_manager(Config(), tmp_path, transcribe=fake)))

    out = schemas.load(run_dir, Transcript)
    assert out.language == "de" and out.runtime == "mlx-speech"
    assert [(i.chunk_id, i.t0, i.t1) for i in out.items] == [(0, 2.0, 5.0), (1, 10.5, 12.0)]
    assert out.items[0].text == "start=2.00 n=48000"  # stripped
    assert out.items[1].text == "start=10.50 n=24000"
    assert fake.languages == ["de", "de"]


def test_no_chunks_writes_empty_transcript_without_loading(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, np.zeros(SR, dtype=np.float32))
    schemas.save(run_dir, ChunksFile(chunker="vad", chunks=[], gaps=[Span(t0=0, t1=1)]))
    models = fake_manager(Config(), tmp_path)  # no transcribe loader: loading would fail
    asr.run_stage(ctx_for(run_dir, "transcribe", models))
    assert schemas.load(run_dir, Transcript).items == []


def test_unsupported_language_is_a_clear_error(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, np.zeros(SR, dtype=np.float32), Config(source_language="fi"))
    with pytest.raises(AsrError, match="doesn't support source language 'fi'"):
        asr.run_stage(ctx_for(run_dir, "transcribe", fake_manager(Config(), tmp_path)))


def test_fourteen_languages() -> None:
    assert len(asr.LANGUAGES) == 14 and "en" in asr.LANGUAGES and "ja" in asr.LANGUAGES
