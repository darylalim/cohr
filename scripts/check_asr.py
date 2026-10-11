"""Phase 0 Transcribe checks (PLAN §5.2, §9; decisions D1, D2, D4). Throwaway.

Run one runtime per process so peak memory is clean:
  uv run --group eval python scripts/check_asr.py RUNTIME REPO REVISION FIXTURES_DIR
RUNTIME: mlx-audio | mlx-speech. Prints one JSON object.

Measures on the fixture meeting (speech runs from truth.json packed into <=30 s chunks):
load time, WER vs the script, seconds per audio minute, peak memory. Then the max input
length probe (D2) and the made-up text check on music/noise/silence (no VAD).
"""

import json
import os
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

SR = 16000


def norm(text: str) -> str:
    t = jiwer.ToLowerCase()(text)
    t = jiwer.RemovePunctuation()(t)
    return " ".join(t.split())


def pack(spans: list[list[float]], max_s: float) -> list[tuple[float, float]]:
    chunks: list[tuple[float, float]] = []
    for t0, t1 in spans:
        if chunks and t1 - chunks[-1][0] <= max_s:
            chunks[-1] = (chunks[-1][0], t1)
        else:
            chunks.append((t0, t1))
    return chunks


def load(runtime: str, repo: str, revision: str):
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, revision=revision))
    sub = root / "mlx-int8"
    if runtime == "mlx-speech":
        from mlx_speech.generation import CohereAsrModel

        model = CohereAsrModel.from_path(sub if sub.exists() else root)

        def run(audio: np.ndarray, lang: str) -> str:
            return model.transcribe(audio, sample_rate=SR, language=lang).text

        return run, str(sub if sub.exists() else root)

    from mlx_audio.stt.utils import load as load_stt

    errors = []
    for path in (root, sub):
        if not path.exists():
            continue
        try:
            model = load_stt(str(path))
        except Exception as e:  # noqa: BLE001 - record and try the next layout
            errors.append(f"{path.name}: {type(e).__name__}: {e}")
            continue

        def run(audio: np.ndarray, lang: str, m=model) -> str:
            return m.transcribe(language=lang, audio_arrays=[audio], sample_rates=[SR])[0]

        return run, str(path)
    raise RuntimeError("; ".join(errors))


def main() -> None:
    runtime, repo, revision, fx = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    import mlx.core as mx

    out: dict = {"runtime": runtime, "repo": repo}
    mx.reset_peak_memory()
    t = time.time()
    try:
        run, path = load(runtime, repo, revision)
    except Exception as e:  # noqa: BLE001
        out["load_error"] = f"{type(e).__name__}: {e}"[:600]
        print(json.dumps(out))
        return
    out["loaded_from"] = path
    out["load_s"] = round(time.time() - t, 2)
    out["load_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)

    truth = json.loads((fx / "truth.json").read_text())
    audio, _ = sf.read(fx / "meeting.wav", dtype="float32")
    chunks = pack(truth["speech_spans"], 30.0)
    run(audio[: SR * 2], "en")  # warm-up
    mx.reset_peak_memory()
    t = time.time()
    texts = [run(audio[int(a * SR) : int(b * SR)], "en") for a, b in chunks]
    dt = time.time() - t
    speech_s = sum(b - a for a, b in chunks)
    hyp = " ".join(texts)
    out["chunks"] = len(chunks)
    out["wer"] = round(jiwer.wer(norm(truth["script_text"]), norm(hyp)), 4)
    out["s_per_audio_min"] = round(dt / (speech_s / 60), 2)
    out["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    out["hyp_head"] = hyp[:300]

    # D2: max input length. Concatenate speech (no gaps) and transcribe one clip of each length.
    speech = np.concatenate([audio[int(a * SR) : int(b * SR)] for a, b in truth["speech_spans"]])
    probe = {}
    for secs in (20, 30, 35, 40, 60):
        clip = speech[: secs * SR]
        n_ref = len(norm(" ".join(truth["script_text"].split())).split())
        words = len(norm(run(clip, "en")).split())
        # Words per second of input; truncation shows as a drop at longer lengths.
        probe[secs] = round(words / secs, 2)
    out["words_per_s_by_len"] = probe
    out["ref_words_per_s"] = round(n_ref / (len(speech) / SR), 2)

    # Made-up text check (no VAD): anything produced here would become a cited "fact".
    madeup = {}
    for name in ("music", "noise", "silence"):
        a, _ = sf.read(fx / f"{name}.wav", dtype="float32")
        madeup[name] = run(a, "en").strip()
    out["madeup"] = madeup
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
