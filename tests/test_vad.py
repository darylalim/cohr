"""PLAN §5.1a, §8: chunk packing from synthetic speech runs, silence fallback, the vad stage."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cohr import schemas, vad
from cohr.config import Config, VadConfig
from cohr.models import ModelManager
from cohr.schemas import ChunksFile
from cohr.vad import Detection
from tests.conftest import NullMemory, ctx_for, fake_manager, make_run

SR = 16000


def test_merge_runs_closes_short_pauses_only() -> None:
    runs = [(0.0, 1.0), (1.3, 2.0), (3.0, 4.0)]
    assert vad.merge_runs(runs, 0.5) == [(0.0, 2.0), (3.0, 4.0)]
    assert vad.merge_runs([(3.0, 4.0), (0.0, 1.0)], 0.5) == [(0.0, 1.0), (3.0, 4.0)]


def test_pack_respects_max_length_and_inner_gap() -> None:
    runs = [(0.0, 10.0), (10.5, 20.0), (20.5, 29.0), (29.5, 35.0)]
    assert vad.pack(runs, 30.0, 2.0) == [(0.0, 29.0), (29.5, 35.0)]
    # A long pause (made-up text risk) splits even when the chunk would fit.
    assert vad.pack([(0.0, 3.0), (10.0, 12.0)], 30.0, 2.0) == [(0.0, 3.0), (10.0, 12.0)]


def test_split_long_cuts_at_least_speechy_frame_in_window() -> None:
    hop = 0.1
    scores = np.ones(700, dtype=np.float32)  # 70 s of confident speech
    scores[270] = 0.05  # 27.0 s: inside the last 5 s before 30 s
    scores[100] = 0.0  # 10 s: lower, but outside the search window
    pieces = vad.split_long((0.0, 70.0), 30.0, scores, hop, 5.0)
    assert pieces[0] == (0.0, pytest.approx(27.0))
    assert all(b - a <= 30.0 + 1e-9 for a, b in pieces)
    assert pieces[-1][1] == 70.0
    assert all(p[1] == q[0] for p, q in zip(pieces, pieces[1:], strict=False))


def test_split_long_without_scores_cuts_at_max() -> None:
    assert vad.split_long((5.0, 70.0), 30.0, None, 0.1, 5.0) == [
        (5.0, 35.0),
        (35.0, 65.0),
        (65.0, 70.0),
    ]
    assert vad.split_long((0.0, 20.0), 30.0, None, 0.1, 5.0) == [(0.0, 20.0)]


def test_gaps_are_the_complement_of_chunks() -> None:
    assert vad.gaps([(2.0, 5.0), (8.0, 9.0)], 12.0) == [(0.0, 2.0), (5.0, 8.0), (9.0, 12.0)]
    assert vad.gaps([(0.0, 12.0)], 12.0) == []
    assert vad.gaps([], 4.0) == [(0.0, 4.0)]


def test_chunk_end_to_end_records_gaps_and_caps_length() -> None:
    det = Detection(
        runs=[(1.0, 4.0), (4.2, 8.0), (30.0, 75.0), (90.0, 91.0)],
        scores=np.full(1000, 0.9, dtype=np.float32),
        hop_s=0.1,
    )
    out = vad.chunk(det, 100.0, VadConfig(), "vad")
    spans = [(c.t0, c.t1) for c in out.chunks]
    assert spans[0] == (1.0, 8.0)  # merged across the 0.2 s pause
    assert all(c.t1 - c.t0 <= 30.0 for c in out.chunks)
    assert [c.chunk_id for c in out.chunks] == list(range(len(out.chunks)))
    assert (out.gaps[0].t0, out.gaps[0].t1) == (0.0, 1.0)
    assert (out.gaps[1].t0, out.gaps[1].t1) == (8.0, 30.0)
    assert out.gaps[-1].t1 == 100.0
    covered = sum(c.t1 - c.t0 for c in out.chunks) + sum(g.t1 - g.t0 for g in out.gaps)
    assert covered == pytest.approx(100.0)


def _speechlike(seconds: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    ) + rng.normal(0, 0.001, len(t)).astype(np.float32)


def _quiet(seconds: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0, 0.002, int(seconds * SR)).astype(np.float32)


def test_silence_fallback_uses_relative_threshold() -> None:
    rng = np.random.default_rng(0)
    audio = np.concatenate(
        [_quiet(3, rng), _speechlike(4, rng), _quiet(5, rng), _speechlike(3, rng)]
    )
    # Raise everything by 20 dB: a fixed threshold would break, a relative one doesn't.
    for gain in (1.0, 10.0):
        det = vad.detect_silence(audio * gain)
        assert len(det.runs) == 2
        (a0, b0), (a1, b1) = det.runs
        assert a0 == pytest.approx(3.0, abs=0.1) and b0 == pytest.approx(7.0, abs=0.1)
        assert a1 == pytest.approx(12.0, abs=0.1) and b1 == pytest.approx(15.0, abs=0.1)


def test_silence_fallback_fills_short_dips() -> None:
    rng = np.random.default_rng(1)
    audio = np.concatenate(
        [_quiet(2, rng), _speechlike(2, rng), _quiet(0.1, rng), _speechlike(2, rng), _quiet(2, rng)]
    )
    assert len(vad.detect_silence(audio).runs) == 1


class FakeVad:
    def __init__(self, runs: list[tuple[float, float]]):
        self.runs = runs
        self.calls = 0

    def detect(self, audio: np.ndarray, cfg: VadConfig) -> Detection:
        self.calls += 1
        return Detection(self.runs, np.full(len(audio) // 512, 0.9, dtype=np.float32), 512 / SR)


def test_vad_stage_writes_chunks_and_releases_model(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, np.zeros(SR * 20, dtype=np.float32))
    fake = FakeVad([(1.0, 4.0), (12.0, 15.0)])
    models = fake_manager(Config(), tmp_path, vad=fake)
    vad.run_stage(ctx_for(run_dir, "vad", models))
    out = schemas.load(run_dir, ChunksFile)
    assert out.chunker == "vad"
    assert [(c.t0, c.t1) for c in out.chunks] == [(1.0, 4.0), (12.0, 15.0)]
    assert [(g.t0, g.t1) for g in out.gaps] == [(0.0, 1.0), (4.0, 12.0), (15.0, 20.0)]
    assert fake.calls == 1
    assert models.loaded() == []  # VAD is released after its step, even when resident


def test_vad_stage_falls_back_to_silence_when_model_fails(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    audio = np.concatenate([_quiet(3, rng), _speechlike(4, rng), _quiet(3, rng)])
    run_dir = make_run(tmp_path, audio)

    def broken(_path: Path) -> object:
        raise RuntimeError("bad weights")

    models = ModelManager(
        Config(), loaders={"vad": broken}, resolve=lambda _s: tmp_path, memory=NullMemory()
    )
    vad.run_stage(ctx_for(run_dir, "vad", models))
    out = schemas.load(run_dir, ChunksFile)
    assert out.chunker == "silence"
    assert len(out.chunks) == 1
    assert out.chunks[0].t0 == pytest.approx(3.0, abs=0.1)


def test_silence_chunker_by_config_loads_no_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    audio = np.concatenate([_quiet(2, rng), _speechlike(3, rng), _quiet(2, rng)])
    run_dir = make_run(tmp_path, audio, Config(vad=VadConfig(chunker="silence")))
    vad.run_stage(ctx_for(run_dir, "vad", None))  # would raise if it asked for a model
    assert schemas.load(run_dir, ChunksFile).chunker == "silence"


class FakeSilero:
    """mlx-audio's generate(): one probability per 512 samples, timestamps in seconds."""

    def __init__(self) -> None:
        self.lengths: list[int] = []

    def generate(self, audio: np.ndarray, **_kw: object) -> SimpleNamespace:
        self.lengths.append(len(audio))
        n = -(-len(audio) // 512)
        end = len(audio) / SR
        return SimpleNamespace(
            probabilities=np.full(n, 0.9, dtype=np.float32),
            timestamps=[{"start": 1.0, "end": end}],
        )


def test_silero_runs_per_block_and_offsets_times() -> None:
    model = FakeSilero()
    audio = np.zeros(int((2 * vad.SILERO_BLOCK_S + 30) * SR), dtype=np.float32)
    det = vad.SileroVad(model).detect(audio, VadConfig())
    block = vad.SILERO_BLOCK_S * SR
    assert model.lengths == [block, block, 30 * SR]
    assert len(det.scores) == -(-len(audio) // 512)  # one score per window, in order
    b = float(vad.SILERO_BLOCK_S)
    assert det.runs == [(1.0, b), (b + 1.0, 2 * b), (2 * b + 1.0, 2 * b + 30.0)]
    short = FakeSilero()
    vad.SileroVad(short).detect(np.zeros(SR * 20, dtype=np.float32), VadConfig())
    assert short.lengths == [SR * 20]  # quick-sized input: one call, as before
