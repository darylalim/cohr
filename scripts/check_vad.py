"""Phase 0 VAD check (PLAN §5.1a, §9; decision D5). Throwaway.

  uv run python scripts/check_vad.py REPO REVISION FIXTURES_DIR

Silero (mlx_audio.vad) vs the relative-threshold silencedetect fallback on the fixture meeting:
speech-frame F1 against truth.json, gap coverage, packed chunks <= max_chunk_s. Also checks that
music/noise/silence clips yield no speech.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.01  # 10 ms frames for F1
MERGE_GAP_S = 0.5
MAX_CHUNK_S = 30.0


def mask(spans: list[tuple[float, float]], dur: float) -> np.ndarray:
    m = np.zeros(int(dur / FRAME) + 1, dtype=bool)
    for a, b in spans:
        m[int(a / FRAME) : int(b / FRAME)] = True
    return m


def f1(pred: np.ndarray, ref: np.ndarray) -> float:
    tp = (pred & ref).sum()
    p = tp / max(pred.sum(), 1)
    r = tp / max(ref.sum(), 1)
    return round(float(2 * p * r / max(p + r, 1e-9)), 4)


def pack(runs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] < MERGE_GAP_S:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    chunks: list[tuple[float, float]] = []
    for a, b in merged:
        if chunks and b - chunks[-1][0] <= MAX_CHUNK_S:
            chunks[-1] = (chunks[-1][0], b)
        else:
            chunks.append((a, b))
    return chunks


def silence_runs(path: Path, dur: float) -> tuple[list[tuple[float, float]], float]:
    audio, _ = sf.read(path, dtype="float32")
    win = audio[: len(audio) // SR * SR].reshape(-1, SR)
    rms_db = 20 * np.log10(np.sqrt((win**2).mean(axis=1)) + 1e-9)
    floor = float(np.percentile(rms_db, 10))
    thr = floor + 6
    err = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={thr:.1f}dB:d=0.3",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    ).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    runs, t = [], 0.0
    for s, e in zip(starts, ends + [dur] * (len(starts) - len(ends)), strict=False):
        if s > t:
            runs.append((t, s))
        t = e
    if t < dur:
        runs.append((t, dur))
    return runs, thr


def main() -> None:
    repo, revision, fx = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    import mlx.core as mx
    from mlx_audio.vad import load

    out: dict = {"repo": repo}
    t = time.time()
    model = load(repo, revision=revision)
    out["load_s"] = round(time.time() - t, 2)

    truth = json.loads((fx / "truth.json").read_text())
    dur = truth["duration_s"]
    audio, _ = sf.read(fx / "meeting.wav", dtype="float32")
    ref = mask([tuple(s) for s in truth["speech_spans"]], dur)
    gaps = truth["gap_spans"]

    mx.reset_peak_memory()
    t = time.time()
    ts = model.get_speech_timestamps(audio, sample_rate=SR, return_seconds=True)
    out["vad_s"] = round(time.time() - t, 2)
    out["peak_mb"] = round(mx.get_peak_memory() / 1e6, 1)
    runs = [(float(x["start"]), float(x["end"])) for x in ts]

    for name, rr in (("silero", runs), ("silence", None)):
        if rr is None:
            rr, thr = silence_runs(fx / "meeting.wav", dur)
            out["silence_threshold_db"] = round(thr, 1)
        chunks = pack(rr)
        in_gap = sum(max(0.0, min(b, g1) - max(a, g0)) for a, b in rr for g0, g1 in gaps)
        out[name] = {
            "runs": len(rr),
            "speech_f1": f1(mask(rr, dur), ref),
            "speech_s_inside_gap": round(in_gap, 2),
            "chunks": [(round(a, 2), round(b, 2)) for a, b in chunks],
            "max_chunk_s": round(max(b - a for a, b in chunks), 2),
        }

    clips = {}
    for name in ("music", "noise", "silence"):
        a, _ = sf.read(fx / f"{name}.wav", dtype="float32")
        r = model.get_speech_timestamps(a, sample_rate=SR, return_seconds=True)
        clips[name] = round(sum(float(x["end"]) - float(x["start"]) for x in r), 2)
    out["speech_s_detected_in_clips"] = clips
    print(json.dumps(out))


if __name__ == "__main__":
    main()
