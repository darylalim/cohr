"""Phase 0 fixtures (PLAN §8): generated on the fly, never committed.

Writes into OUT_DIR:
  meeting.wav       ~2 min, two `say` voices, 20 s of non-speech (noise + tones) spliced in
  slide_1..3.png    PIL slides with titles, bullets and a table of known numbers
  music.wav, noise.wav, silence.wav   30 s each, for the made-up text check (§9)
  truth.json        script text per line, speech spans, gap spans, slide facts

Usage: uv run python scripts/fixtures.py OUT_DIR
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

SR = 16000

# (voice, text). Facts here match the slides: budget 14,500, launch 3 November, 38 percent.
SCRIPT = [
    (
        "Samantha",
        "Good morning everyone. This is the weekly planning meeting for the Atlas project.",
    ),
    (
        "Daniel",
        "Thanks. Let's start with the budget. The approved budget for the next quarter is "
        "fourteen thousand five hundred dollars.",
    ),
    (
        "Samantha",
        "That matches the slide. Most of it goes to user testing and the new onboarding flow.",
    ),
    (
        "Daniel",
        "We decided to launch the beta on the third of November. Marketing needs the final "
        "screenshots by the twentieth of October.",
    ),
    (
        "Samantha",
        "Priya will own the screenshots. Tom will write the release notes by the end of next week.",
    ),
    (
        "Daniel",
        "Next topic is retention. Thirty eight percent of trial users came back in the "
        "second week, up from thirty one percent last month.",
    ),
    (
        "Samantha",
        "The onboarding checklist seems to be the main reason. We should keep it and add "
        "a short video.",
    ),
    (
        "Daniel",
        "One open question is whether we support offline mode in the beta. Nobody owns that yet.",
    ),
    (
        "Samantha",
        "Let's revisit offline mode next meeting. Last item, the support queue has "
        "forty two open tickets.",
    ),
    (
        "Daniel",
        "I'll ask the support team to close the duplicates before Friday. That's "
        "everything from me.",
    ),
    (
        "Samantha",
        "Great. To recap, the budget is fourteen thousand five hundred, the beta launches "
        "on the third of November, and Priya and Tom have their action items.",
    ),
]

SLIDES = [
    (
        "Q4 Budget",
        ["Approved: $14,500", "User testing and onboarding"],
        [["Item", "Amount"], ["User testing", "$8,000"], ["Onboarding", "$6,500"]],
    ),
    (
        "Beta Launch Plan",
        ["Launch: 3 November", "Screenshots due 20 October (Priya)", "Release notes (Tom)"],
        None,
    ),
    (
        "Retention",
        ["Week-2 return rate up", "Keep onboarding checklist"],
        [["Month", "Week-2 return"], ["September", "31%"], ["October", "38%"]],
    ),
]

GAP_AFTER_LINE = 5  # splice the non-speech block after this line index
PAUSE_S = 0.6


def say_to_array(voice: str, text: str, tmp: Path) -> np.ndarray:
    aiff = tmp / "line.aiff"
    wav = tmp / "line.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(aiff),
            "-ac",
            "1",
            "-ar",
            str(SR),
            str(wav),
        ],
        check=True,
    )
    audio, _ = sf.read(wav, dtype="float32")
    return audio


def lavfi(source: str, seconds: float) -> np.ndarray:
    out = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            source,
            "-t",
            str(seconds),
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def music(seconds: float) -> np.ndarray:
    # A simple chord progression as "music": three sines per bar.
    t = np.arange(int(seconds * SR)) / SR
    bars = [
        (261.6, 329.6, 392.0),
        (220.0, 261.6, 329.6),
        (174.6, 220.0, 261.6),
        (196.0, 246.9, 293.7),
    ]
    out = np.zeros_like(t, dtype=np.float32)
    bar_len = 2.0
    for i, freqs in enumerate(bars * int(np.ceil(seconds / (bar_len * len(bars))))):
        mask = (t >= i * bar_len) & (t < (i + 1) * bar_len)
        out[mask] = sum(np.sin(2 * np.pi * f * t[mask]) for f in freqs) / 3 * 0.3
    return out


def noise(seconds: float) -> np.ndarray:
    return lavfi(f"anoisesrc=color=pink:amplitude=0.05:duration={seconds}", seconds)


def draw_slide(title: str, bullets: list[str], table: list[list[str]] | None, path: Path) -> None:
    img = Image.new("RGB", (1600, 900), "white")
    d = ImageDraw.Draw(img)
    d.text((80, 60), title, fill="black", font=ImageFont.load_default(size=72))
    body = ImageFont.load_default(size=44)
    y = 200
    for b in bullets:
        d.text((100, y), f"- {b}", fill="black", font=body)
        y += 70
    if table:
        y += 30
        col_w, row_h = 420, 64
        for r, row in enumerate(table):
            for c, cell in enumerate(row):
                x0, y0 = 100 + c * col_w, y + r * row_h
                d.rectangle([x0, y0, x0 + col_w, y0 + row_h], outline="black", width=2)
                d.text((x0 + 16, y0 + 10), cell, fill="black", font=body)
    img.save(path)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pause = np.zeros(int(PAUSE_S * SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    lines, speech_spans, gap_spans = [], [], []
    t = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (voice, text) in enumerate(SCRIPT):
            a = say_to_array(voice, text, Path(tmp))
            speech_spans.append([round(t, 2), round(t + len(a) / SR, 2)])
            lines.append({"voice": voice, "text": text, "t0": round(t, 2)})
            parts += [a, pause]
            t += (len(a) + len(pause)) / SR
            if i == GAP_AFTER_LINE:
                gap = np.concatenate([noise(10), music(10)])
                gap_spans.append([round(t, 2), round(t + len(gap) / SR, 2)])
                parts += [gap, pause]
                t += (len(gap) + len(pause)) / SR
    meeting = np.concatenate(parts)
    sf.write(out / "meeting.wav", meeting, SR, subtype="PCM_16")

    sf.write(out / "music.wav", music(30), SR, subtype="PCM_16")
    sf.write(out / "noise.wav", noise(30), SR, subtype="PCM_16")
    sf.write(out / "silence.wav", np.zeros(30 * SR, dtype=np.float32), SR, subtype="PCM_16")

    for i, (title, bullets, table) in enumerate(SLIDES, start=1):
        draw_slide(title, bullets, table, out / f"slide_{i}.png")

    truth = {
        "duration_s": round(len(meeting) / SR, 2),
        "lines": lines,
        "script_text": " ".join(text for _, text in SCRIPT),
        "speech_spans": speech_spans,
        "gap_spans": gap_spans,
        "slides": [
            {"slide_id": i, "title": s[0], "bullets": s[1], "table": s[2]}
            for i, s in enumerate(SLIDES, start=1)
        ],
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"meeting.wav {truth['duration_s']} s, gaps {gap_spans} -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
