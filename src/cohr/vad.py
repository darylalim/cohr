"""Speech detection and chunking (PLAN §5.1a).

Silero VAD (mlx_audio.vad) finds speech runs; non-speech is never sent to Transcribe, which
makes up text from music, noise and silence (Phase 0). Fallback when the model can't load: a
relative-threshold silence detector on the same audio. Both feed the same pure packing:

    runs ─► merge (< merge_gap_s apart) ─► split runs > max_chunk_s at the least-speechy
    frame in their last split_search_s ─► pack consecutive runs up to max_chunk_s, never across
    a pause > max_inner_gap_s ─► chunks; everything outside a chunk is a gap.

Output: chunks.json only. asr.py reads each chunk from audio.wav by its offsets.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from cohr import schemas
from cohr.config import VadConfig
from cohr.pipeline import Stage, StageContext
from cohr.schemas import Chunk, ChunksFile, Span

SR = 16000
SILERO_HOP_S = 512 / SR  # one speech probability per 512 samples at 16 kHz
ENERGY_HOP_S = 0.032  # same frame length for the silence fallback
NOISE_FLOOR_PERCENTILE = 10  # quietest ~10% of 1 s windows (§5.1a)
NOISE_MARGIN_DB = 6.0
MIN_SILENCE_S = 0.3  # silence fallback: shorter dips are pauses inside speech

Run = tuple[float, float]


@dataclass(frozen=True)
class Detection:
    """Speech runs (seconds) and a per-frame score where lower = better place to cut."""

    runs: list[Run]
    scores: np.ndarray
    hop_s: float


# --- pure chunking -------------------------------------------------------------------------


def merge_runs(runs: list[Run], gap_s: float) -> list[Run]:
    merged: list[Run] = []
    for a, b in sorted(runs):
        if merged and a - merged[-1][1] < gap_s:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def split_long(
    run: Run, max_s: float, scores: np.ndarray | None, hop_s: float, search_s: float
) -> list[Run]:
    """Cut a run longer than max_s at the lowest-score frame within its last search_s, repeatedly.

    Every piece is ≤ max_s. Without scores (or frames in the window) the cut is at max_s.
    """
    a, b = run
    pieces: list[Run] = []
    while b - a > max_s:
        hi = a + max_s
        lo = max(hi - search_s, a)
        cut = hi
        if scores is not None and len(scores):
            first = math.ceil(lo / hop_s - 1e-9)
            last = min(math.floor(hi / hop_s + 1e-9), len(scores) - 1)
            if first <= last:
                window = scores[first : last + 1]
                cut = (first + int(np.argmin(window))) * hop_s
                if cut <= a:  # degenerate window: keep making progress
                    cut = hi
        pieces.append((a, cut))
        a = cut
    pieces.append((a, b))
    return pieces


def pack(runs: list[Run], max_s: float, max_inner_gap_s: float) -> list[Run]:
    """Group consecutive runs into chunks ≤ max_s, cutting only between runs."""
    chunks: list[Run] = []
    for a, b in runs:
        if chunks and b - chunks[-1][0] <= max_s and a - chunks[-1][1] <= max_inner_gap_s:
            chunks[-1] = (chunks[-1][0], b)
        else:
            chunks.append((a, b))
    return chunks


def gaps(chunks: list[Run], duration: float, min_s: float = 0.01) -> list[Run]:
    """The complement of the chunks within [0, duration]: never transcribed."""
    out: list[Run] = []
    t = 0.0
    for a, b in chunks:
        if a - t >= min_s:
            out.append((t, a))
        t = max(t, b)
    if duration - t >= min_s:
        out.append((t, duration))
    return out


def chunk(det: Detection, duration: float, cfg: VadConfig, chunker: str) -> ChunksFile:
    runs = [(max(0.0, a), min(duration, b)) for a, b in det.runs if b > a]
    runs = merge_runs(runs, cfg.merge_gap_s)
    pieces = [
        p
        for r in runs
        for p in split_long(r, cfg.max_chunk_s, det.scores, det.hop_s, cfg.split_search_s)
    ]
    packed = pack(pieces, cfg.max_chunk_s, cfg.max_inner_gap_s)
    return ChunksFile(
        chunker="vad" if chunker == "vad" else "silence",
        chunks=[
            Chunk(chunk_id=i, t0=round(a, 3), t1=round(b, 3)) for i, (a, b) in enumerate(packed)
        ],
        gaps=[Span(t0=round(a, 3), t1=round(b, 3)) for a, b in gaps(packed, duration)],
    )


# --- silence fallback (pure numpy) ---------------------------------------------------------


def frame_db(audio: np.ndarray, hop_s: float = ENERGY_HOP_S) -> np.ndarray:
    n = max(1, round(hop_s * SR))
    usable = len(audio) // n * n
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[:usable].reshape(-1, n).astype(np.float64)
    return (20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)).astype(np.float32)


def noise_threshold_db(audio: np.ndarray) -> float:
    """Noise floor (10th percentile of 1 s window levels) + 6 dB: relative, not a fixed -30 dB."""
    levels = frame_db(audio, 1.0)
    if not len(levels):
        return -60.0
    return float(np.percentile(levels, NOISE_FLOOR_PERCENTILE)) + NOISE_MARGIN_DB


def detect_silence(audio: np.ndarray) -> Detection:
    db = frame_db(audio)
    loud = db >= noise_threshold_db(audio)
    min_frames = math.ceil(MIN_SILENCE_S / ENERGY_HOP_S)
    # Fill quiet dips shorter than MIN_SILENCE_S inside speech.
    i = 0
    while i < len(loud):
        if not loud[i]:
            j = i
            while j < len(loud) and not loud[j]:
                j += 1
            if i > 0 and j < len(loud) and j - i < min_frames:
                loud[i:j] = True
            i = j
        else:
            i += 1
    runs: list[Run] = []
    start = None
    for k, on in enumerate(loud):
        if on and start is None:
            start = k
        elif not on and start is not None:
            runs.append((start * ENERGY_HOP_S, k * ENERGY_HOP_S))
            start = None
    if start is not None:
        runs.append((start * ENERGY_HOP_S, len(loud) * ENERGY_HOP_S))
    return Detection(runs=runs, scores=db, hop_s=ENERGY_HOP_S)


# --- Silero (MLX) --------------------------------------------------------------------------


class SileroVad:
    def __init__(self, model: Any):
        self.model = model

    def detect(self, audio: np.ndarray, cfg: VadConfig) -> Detection:
        out = self.model.generate(
            audio,
            sample_rate=SR,
            threshold=cfg.threshold,
            min_speech_duration_ms=cfg.min_speech_ms,
            min_silence_duration_ms=cfg.min_silence_ms,
            speech_pad_ms=cfg.pad_ms,
            return_seconds=True,
        )
        probs = np.asarray(out.probabilities, dtype=np.float32).reshape(-1)
        runs = [(float(t["start"]), float(t["end"])) for t in out.timestamps]
        return Detection(runs=runs, scores=probs, hop_s=SILERO_HOP_S)


def load(path: Path) -> SileroVad:
    from mlx_audio.vad import load as load_vad

    return SileroVad(load_vad(str(path)))


# --- stage ---------------------------------------------------------------------------------


def run_stage(ctx: StageContext) -> None:
    from cohr.models import ModelLoadError

    cfg = ctx.config.vad
    audio, sr = sf.read(ctx.run_dir / "audio.wav", dtype="float32")
    if sr != SR or audio.ndim != 1:
        raise ValueError(f"audio.wav must be 16 kHz mono, got {sr} Hz, shape {audio.shape}")
    duration = len(audio) / SR

    chunker = cfg.chunker
    det: Detection | None = None
    if chunker == "vad":
        ctx.report(0.1, "detecting speech")
        try:
            with ctx.model("vad") as vad:
                det = vad.detect(audio, cfg)
        except ModelLoadError as e:
            ctx.log.warning("vad: %s; falling back to silence detection", e)
            chunker = "silence"
    if det is None:
        ctx.report(0.1, "detecting silence")
        det = detect_silence(audio)

    out = chunk(det, duration, cfg, chunker)
    speech = sum(c.t1 - c.t0 for c in out.chunks)
    ctx.log.info(
        "vad: %s, %d chunks, %.1f s of %.1f s sent to Transcribe",
        chunker,
        len(out.chunks),
        speech,
        duration,
    )
    schemas.save(ctx.run_dir, out)


STAGE = Stage(
    name="vad",
    run=run_stage,
    outputs=(ChunksFile.FILENAME,),
    format_version=ChunksFile.FORMAT_VERSION,
    config_keys=("vad", "models.vad"),
    models=("vad",),
)
