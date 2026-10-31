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


# --- D48: the opening probe ---------------------------------------------------------------

# xling-ja chunk 7 (8i): the reply lost "先月は7700ドルでした。" and has no 。 inside.
JA_LOST = (
    "補足するとクラウド費用のはかり方は春から変わっていませんクラウド費用には季節性があるので"
    "一つの期間だけで判断しない方がいいと思います。"
)
JA_OPENING = "先月は7700ドルでした。"


def _meeting() -> np.ndarray:
    """20 s: a quiet floor, and from 4 s a 2 s line, a 0.6 s pause, then 6 s of speech."""
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(20 * SR) * 1e-4).astype(np.float32)
    tone = (0.1 * np.sin(2 * np.pi * 220 * np.arange(20 * SR) / SR)).astype(np.float32)
    for a, b in ((4.0, 6.0), (6.6, 12.6)):
        audio[round(a * SR) : round(b * SR)] += tone[round(a * SR) : round(b * SR)]
    return audio


class ProbeAsr:
    """The whole chunk's reply, or the probe's for anything shorter; records each length."""

    def __init__(self, reply: str, probe: str, chunk_s: float) -> None:
        self.reply, self.probe, self.n = reply, probe, round(chunk_s * SR)
        self.lengths: list[int] = []

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        self.lengths.append(len(audio))
        return self.reply if len(audio) == self.n else self.probe


def _transcribe(tmp_path: Path, language: str, fake: ProbeAsr) -> str:
    run_dir = make_run(tmp_path, _meeting(), Config(source_language=language))
    chunks = [Chunk(chunk_id=3, t0=4.0, t1=12.6)]
    schemas.save(run_dir, ChunksFile(chunker="vad", chunks=chunks, gaps=[]))
    models = fake_manager(Config(), tmp_path, transcribe=fake)
    asr.run_stage(ctx_for(run_dir, "transcribe", models))
    return schemas.load(run_dir, Transcript).items[0].text


def test_unpunctuated_reply_gets_its_lost_opening(tmp_path: Path) -> None:
    fake = ProbeAsr(JA_LOST, JA_OPENING, 8.6)
    assert _transcribe(tmp_path, "ja", fake) == JA_OPENING + JA_LOST
    assert len(fake.lengths) == 2 and abs(fake.lengths[1] - 2.0 * SR) < 0.1 * SR  # up to the pause


def test_punctuated_reply_is_not_probed(tmp_path: Path) -> None:
    punctuated = "補足すると、クラウド費用のはかり方は春から変わっていません。" + JA_LOST
    fake = ProbeAsr(punctuated, JA_OPENING, 8.6)
    assert _transcribe(tmp_path, "ja", fake) == punctuated
    assert len(fake.lengths) == 1


def test_english_reply_holding_its_opening_is_kept(tmp_path: Path) -> None:
    # AMI-like spontaneous speech: no punctuation, nothing lost.
    reply = "um so I was thinking we could go with the rubber buttons on the side and keep it"
    fake = ProbeAsr(reply, "Um, so I was thinking.", 8.6)
    assert _transcribe(tmp_path, "en", fake) == reply
    assert len(fake.lengths) == 2


def test_unpunctuated() -> None:
    assert asr.unpunctuated(JA_LOST)  # one 。, at the end
    assert not asr.unpunctuated(JA_OPENING + JA_LOST)
    assert not asr.unpunctuated("Yeah, okay then, I think we're done")  # 40 characters or less
    assert not asr.unpunctuated("Right. " + "and so on " * 5)


def test_with_opening() -> None:
    assert asr.with_opening(JA_OPENING, JA_LOST, "ja") == JA_OPENING + JA_LOST
    assert (
        asr.with_opening("昨日リテンションの数字", "リテンションの数字をくわしく見ました", "ja")
        is None
    )
    # The probe's last words open the reply (AMI IS1009c chunk 51): only what comes before.
    reply = "That button and the transistor will amplify the signal"
    assert asr.with_opening("b uh tha that button", reply, "en") == "b uh tha " + reply
    assert asr.with_opening("That brings us to the chip.", "Well there isn't", "en") == (
        "That brings us to the chip. Well there isn't"
    )
    assert asr.with_opening("...", reply, "en") is None


def test_first_span_end() -> None:
    audio = _meeting()
    threshold = -40.0
    end = asr.first_span_end(audio[4 * SR : round(12.6 * SR)], threshold)
    assert end is not None and abs(end - 2.0) < 0.1
    # The span would be the whole chunk (no pause, or one within 1 s of its end).
    assert asr.first_span_end(audio[round(6.6 * SR) : round(12.6 * SR)], threshold) is None
    assert asr.first_span_end(audio[4 * SR : round(6.9 * SR)], threshold) is None
    assert asr.first_span_end(np.zeros(SR, dtype=np.float32), threshold) is None
