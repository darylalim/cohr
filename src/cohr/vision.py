"""Describe slides (PLAN §5.3): North Micro Vision on mlx-vlm, one frame per call.

The prompt asks for JSON only. If it doesn't parse, retry once with the retry prompt; after
that, keep the raw text as text_md. Phase 0 quirks normalized here: tables_md as row objects
instead of Markdown strings, and "None" for empty fields.
"""

import json
import re
from pathlib import Path
from typing import Any

from cohr import schemas
from cohr.config import load_prompt
from cohr.pipeline import Stage, StageContext
from cohr.schemas import Frame, FramesFile, Slide, SlidesFile

KEYS = ("title", "text_md", "tables_md", "chart_notes", "visual_notes")
_EMPTY = {"", "none", "null", "n/a", "na", "-"}
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


# --- parsing (pure) ------------------------------------------------------------------------


def parse(text: str) -> dict[str, Any] | None:
    """The outermost {...} in the output, if it's a JSON object with at least one slide key."""
    m = _JSON_OBJECT.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not any(k in obj for k in KEYS):
        return None
    return obj


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(t for v in value if (t := _text(v)))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    return "" if text.lower() in _EMPTY else text


def _cell(value: Any) -> str:
    return _text(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _table(value: Any) -> str:
    """One table from a Markdown string, a list of row dicts, or a list of row lists."""
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value:
        return _text(value)
    if all(isinstance(r, dict) for r in value):
        header: list[str] = []
        for r in value:
            header += [str(k) for k in r if str(k) not in header]
        body = [[_cell(r.get(k, "")) for k in header] for r in value]
        return markdown_table([header, *body])
    if all(isinstance(r, list) for r in value):
        return markdown_table([[_cell(c) for c in r] for r in value])
    return _text(value)


def tables(value: Any) -> list[str]:
    if value is None or isinstance(value, str):
        return [t] if (t := _text(value)) else []
    if isinstance(value, dict):
        return [t] if (t := _table(value)) else []
    if isinstance(value, list):
        # Phase 0: a single table came back as a list of row objects.
        if value and all(isinstance(r, dict) for r in value):
            return [_table(value)]
        return [t for v in value if (t := _table(v))]
    return [t] if (t := _text(value)) else []


def make_slide(slide_id: int, frame: Frame, obj: dict[str, Any] | None, raw: str) -> Slide:
    timing: dict[str, Any] = {
        "slide_id": slide_id,
        "frame_id": frame.frame_id,
        "t": frame.pts,
        "t_source": "pts" if frame.pts is not None else "none",
    }
    if obj is None:
        return Slide(
            **timing,
            title="",
            text_md=raw.strip(),
            tables_md=[],
            chart_notes="",
            visual_notes="",
            raw=raw,
        )
    return Slide(
        **timing,
        title=_text(obj.get("title")),
        text_md=_text(obj.get("text_md")),
        tables_md=tables(obj.get("tables_md")),
        chart_notes=_text(obj.get("chart_notes")),
        visual_notes=_text(obj.get("visual_notes")),
    )


# --- model (MLX) ---------------------------------------------------------------------------


class MicroVision:
    def __init__(self, model: Any, processor: Any):
        self.model = model
        self.processor = processor

    def describe(
        self, image: Path, prompt: str, *, max_tokens: int, temperature: float, seed: int
    ) -> str:
        import mlx.core as mx
        from mlx_vlm import apply_chat_template, generate

        mx.random.seed(seed)
        formatted = str(
            apply_chat_template(self.processor, self.model.config, prompt, num_images=1)
        )
        result = generate(
            self.model,
            self.processor,
            formatted,
            image=[str(image)],
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        return result.text if hasattr(result, "text") else str(result)


def load(path: Path) -> MicroVision:
    from mlx_vlm import load as load_vlm

    model, processor = load_vlm(str(path))
    return MicroVision(model, processor)


# --- stage ---------------------------------------------------------------------------------


def run_stage(ctx: StageContext) -> None:
    cfg = ctx.config.vision
    frames = schemas.load(ctx.run_dir, FramesFile).frames
    prompt = load_prompt("vision")
    retry = f"{prompt}\n\n{load_prompt('vision_retry')}"
    slides: list[Slide] = []
    retried = failed = 0
    if frames:
        with ctx.model("vision") as vlm:
            for i, frame in enumerate(frames):
                ctx.report(i / len(frames), f"slide {i + 1}/{len(frames)}")
                image = ctx.run_dir / frame.path
                obj = None
                raw = ""
                for attempt, text in enumerate((prompt, retry)):
                    retried += attempt
                    raw = vlm.describe(
                        image,
                        text,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        seed=ctx.config.seed,
                    )
                    if (obj := parse(raw)) is not None:
                        break
                failed += obj is None
                slides.append(make_slide(i, frame, obj, raw))
    ctx.log.info("vision: %d slides, %d retried, %d kept as raw text", len(slides), retried, failed)
    schemas.save(ctx.run_dir, SlidesFile(slides=slides))


STAGE = Stage(
    name="vision",
    run=run_stage,
    outputs=(SlidesFile.FILENAME,),
    format_version=SlidesFile.FORMAT_VERSION,
    config_keys=("seed", "vision", "models.vision"),
    prompts=("vision", "vision_retry"),
    models=("vision",),
)
