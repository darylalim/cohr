"""Build one eval item from its script (PLAN §18): audio, slides, deck or video, truth.json.

Speech comes from macOS `say` (deterministic: the same text and voice give the same samples),
non-speech from numpy (seeded pink noise, a chord progression, silence), slides from PIL.
Video items are a screen recording of the slides (ffmpeg, one slide per topic, switched when
the topic starts); PDF items are the same slides as a separate deck.

Everything is written into a temp dir and moved into place at the end (atomic, §3).
"""

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from cohr import atomic
from cohr.evals import ItemSpec, Truth, TruthFact, TruthLine, TruthSlide, gen_hash
from cohr.schemas import Span
from evals.gen.script import GapKind, Script, SlideSpec, Variant, generate

SR = 16000
PAUSE_S = 0.6
SLIDE_SIZE = (1600, 900)
# A different header colour per slide, so a screen recording has a clear scene change.
HEADER_COLORS = ("#1f4e79", "#7a2e2e", "#2e6b3a", "#6b4c9a", "#9a6b1f", "#2e6b6b", "#4a4a4a")
FPS = 5
# PIL's default font has no CJK glyphs and no "é" (6d-2: Japanese slides came out as boxes,
# "Métrica" as "M□trica"). These ship with macOS, which the generator needs anyway for `say`.
# English keeps the default font, so its slides are unchanged.
FONTS = {
    "es": Path("/System/Library/Fonts/Helvetica.ttc"),
    "ja": Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
}

Synth = Callable[[str, str], np.ndarray]  # (voice, text) → mono float32 at SR


SAY_TIMEOUT_S = 60  # `say` hung on one line for 40 min while generating long-en (6c)
SAY_TRIES = 3


def say(voice: str, text: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        aiff, wav = Path(tmp) / "line.aiff", Path(tmp) / "line.wav"
        for attempt in range(SAY_TRIES):
            try:
                cmd = ["say", "-v", voice, "-o", str(aiff), text]
                subprocess.run(cmd, check=True, timeout=SAY_TIMEOUT_S)
                break
            except subprocess.TimeoutExpired:
                if attempt == SAY_TRIES - 1:
                    raise
        _ffmpeg("-i", str(aiff), "-ac", "1", "-ar", str(SR), str(wav))
        audio, _ = sf.read(wav, dtype="float32")
    return audio


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True
    )


# --- non-speech ----------------------------------------------------------------------------


def pink_noise(seconds: float, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    n = int(seconds * SR)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spectrum[1:] /= np.sqrt(freqs[1:])
    spectrum[0] = 0
    noise = np.fft.irfft(spectrum, n)
    return (noise / (np.abs(noise).max() or 1.0) * amplitude).astype(np.float32)


def music(seconds: float) -> np.ndarray:
    """A chord progression, three sines per 2 s bar."""
    t = np.arange(int(seconds * SR)) / SR
    bars = ((261.6, 329.6, 392.0), (220.0, 261.6, 329.6), (174.6, 220.0, 261.6))
    out = np.zeros_like(t, dtype=np.float32)
    for i in range(int(np.ceil(seconds / 2))):
        mask = (t >= i * 2) & (t < (i + 1) * 2)
        out[mask] = sum(np.sin(2 * np.pi * f * t[mask]) for f in bars[i % len(bars)]) / 3 * 0.3
    return out


def gap_audio(kind: GapKind, seconds: float, rng: np.random.Generator) -> np.ndarray:
    if kind == "noise":
        return pink_noise(seconds, 0.3, rng)
    if kind == "music":
        return music(seconds)
    return np.zeros(int(seconds * SR), dtype=np.float32)


# --- slides --------------------------------------------------------------------------------


def font(language: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if language in FONTS:
        return ImageFont.truetype(FONTS[language], size)
    return ImageFont.load_default(size=size)


def draw_slide(slide: SlideSpec, index: int, language: str = "en") -> Image.Image:
    img = Image.new("RGB", SLIDE_SIZE, "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, SLIDE_SIZE[0], 150], fill=HEADER_COLORS[index % len(HEADER_COLORS)])
    d.text((80, 40), slide.title, fill="white", font=font(language, 72))
    body = font(language, 44)
    y = 200
    for b in slide.bullets:
        d.text((100, y), f"- {b}", fill="black", font=body)
        y += 70
    y += 30
    col_w, row_h = 600, 64
    for r, row in enumerate(slide.table):
        for c, cell in enumerate(row):
            x0, y0 = 100 + c * col_w, y + r * row_h
            d.rectangle([x0, y0, x0 + col_w, y0 + row_h], outline="black", width=2)
            d.text((x0 + 16, y0 + 10), cell, fill="black", font=body)
    return img


def screen_recording(
    slides: list[Path], starts: list[float], audio: Path, duration: float, out: Path
) -> None:
    """An mp4 showing slide k from starts[k] until the next start, with the meeting audio."""
    listing = out.with_suffix(".txt")
    ends = [*starts[1:], duration]
    entries = [
        f"file '{p.resolve()}'\nduration {e - s:.3f}"
        for p, s, e in zip(slides, starts, ends, strict=True)
    ]
    listing.write_text("\n".join([*entries, f"file '{slides[-1].resolve()}'"]) + "\n")
    _ffmpeg(
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-i",
        str(audio),
        "-vf",
        f"fps={FPS},format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "stillimage",
        "-c:a",
        "aac",
        "-shortest",
        str(out),
    )
    listing.unlink()


# --- item ----------------------------------------------------------------------------------


def variant(spec: ItemSpec) -> Variant:
    return Variant(
        topics=spec.topics,
        decisions=spec.decisions,
        actions=spec.actions,
        gaps=tuple(spec.gaps),
        gap_s=spec.gap_s,
        conflict=spec.conflict,
        regions=tuple(spec.regions),
        discussion=spec.discussion,
        open_questions=spec.open_questions,
        language=spec.language,
    )


def assemble(
    script: Script, spec: ItemSpec, synth: Synth
) -> tuple[np.ndarray, list[TruthLine], list[Span]]:
    """The meeting audio, each line's time span, and the non-speech gap spans."""
    rng = np.random.default_rng(spec.seed)
    pause = np.zeros(int(PAUSE_S * SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    lines: list[TruthLine] = []
    gaps: list[Span] = []
    t = 0.0
    for i, line in enumerate(script.lines):
        audio = synth(line.voice, line.text)
        lines.append(
            TruthLine(
                voice=line.voice, text=line.text, t0=round(t, 2), t1=round(t + len(audio) / SR, 2)
            )
        )
        parts += [audio, pause]
        t += (len(audio) + len(pause)) / SR
        if i in script.gaps_after:
            gap = gap_audio(script.gaps_after[i], spec.gap_s, rng)
            gaps.append(Span(t0=round(t, 2), t1=round(t + len(gap) / SR, 2)))
            parts += [gap, pause]
            t += (len(gap) + len(pause)) / SR
    meeting = np.concatenate(parts)
    if spec.background_noise:
        meeting = (
            meeting + pink_noise(len(meeting) / SR, spec.background_noise, rng)[: len(meeting)]
        )
    return np.clip(meeting, -1.0, 1.0).astype(np.float32), lines, gaps


def topic_spans(script: Script, lines: list[TruthLine]) -> list[tuple[float, float]]:
    spans = []
    for k in range(len(script.slides)):
        idx = [i for i, line in enumerate(script.lines) if line.topic == k]
        spans.append((lines[idx[0]].t0, lines[idx[-1]].t1))
    return spans


def build(spec: ItemSpec, out: Path, synth: Synth = say, root: Path = Path()) -> Truth:
    """Write the item into `out` (replaced as a whole) and return its truth."""
    if spec.media == "audio" and spec.slides != "pdf":
        raise ValueError(f"{spec.id}: an audio item needs slides: pdf")
    script = generate(spec.seed, variant(spec))
    meeting, lines, gaps = assemble(script, spec, synth)
    duration = round(len(meeting) / SR, 2)
    spans = topic_spans(script, lines)

    facts = [
        TruthFact(
            id=f.id,
            kind=f.kind,
            text=f.text,
            owner=f.owner,
            due=f.due,
            value=f.value,
            misspoken=f.misspoken,
            t0=lines[f.line].t0,
            t1=lines[f.line].t1,
            slide=f.topic if f.on_slide else None,
        )
        for f in script.facts
    ]
    slides = [
        TruthSlide(
            slide_id=k,
            title=s.title,
            cells=[cell for row in s.table for cell in row],
            t0=spans[k][0],
            t1=spans[k][1],
        )
        for k, s in enumerate(script.slides)
    ]
    media = "meeting.mp4" if spec.media == "video" else "meeting.wav"
    truth = Truth(
        item=spec,
        gen_hash=gen_hash(root),
        language=spec.language,
        duration_s=duration,
        media=media,
        deck="deck.pdf" if spec.slides == "pdf" else None,
        script_text=" ".join(line.text for line in lines),
        lines=lines,
        facts=facts,
        speech_spans=[Span(t0=line.t0, t1=line.t1) for line in lines],
        gap_spans=gaps,
        slides=slides,
    )

    with atomic.atomic_dir(out) as tmp:
        sf.write(tmp / "meeting.wav", meeting, SR, subtype="PCM_16")
        images = [draw_slide(s, k, spec.language) for k, s in enumerate(script.slides)]
        pngs = []
        for k, img in enumerate(images):
            pngs.append(tmp / f"slide_{k + 1}.png")
            img.save(pngs[-1])
        if spec.slides == "pdf":
            images[0].save(tmp / "deck.pdf", save_all=True, append_images=images[1:])
        if spec.media == "video":
            starts = [0.0, *(s[0] for s in spans[1:])]
            screen_recording(pngs, starts, tmp / "meeting.wav", duration, tmp / media)
        (tmp / "truth.json").write_text(truth.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return truth
