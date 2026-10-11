"""Map (PLAN §5.5): Tiny Aya Global on mlx-lm, one call per segment, JSON in the target language.

The prompt ends with the target-language instruction (Phase 0: with it only at the top, de and ja
notes came back in English). Invalid JSON is retried once; after that the raw text becomes the
summary with `raw: true`, so one bad segment never fails the stage.
"""

import json
import re
from pathlib import Path
from typing import Any

from cohr import schemas
from cohr.config import language_name, load_prompt
from cohr.fuse import slide_text
from cohr.pipeline import Stage, StageContext
from cohr.schemas import MapActionItem, MapFile, Segment, SegmentNotes, SegmentsFile, SlidesFile

KEYS = ("summary", "key_points", "decisions", "action_items", "open_questions", "terms")
_EMPTY = {"", "none", "null", "n/a", "na", "-", "?", "unknown"}
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
_PLACEHOLDER = re.compile(r"\{(target_language|time_range|slides|transcript)\}")


# --- prompt and parsing (pure) -------------------------------------------------------------


def clock(t: float) -> str:
    s = int(t)
    return (
        f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"
    )


def fill(template: str, **values: str) -> str:
    """Substitute the {placeholders} in one pass, so text inside a value is never substituted
    (the prompts also contain literal JSON braces, which str.format would trip on)."""
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def prompt_values(seg: Segment, slides: list[str], target_language: str) -> dict[str, str]:
    return {
        "target_language": language_name(target_language),
        "time_range": f"{clock(seg.t0)}–{clock(seg.t1)}",
        "slides": "\n\n".join(slides) or "(none)",
        "transcript": seg.transcript_text,
    }


def parse(text: str) -> dict[str, Any] | None:
    """The outermost {...} in the reply, if it's a JSON object with at least one Map key."""
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


def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    return "" if text.casefold() in _EMPTY else text


def _strings(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    return [t for v in items if (t := _str(v))]


def _actions(value: Any) -> list[MapActionItem]:
    out = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, dict):
            task = _str(item.get("task"))
            owner, due = _str(item.get("owner")) or None, _str(item.get("due")) or None
        else:
            task, owner, due = _str(item), None, None
        if task:
            out.append(MapActionItem(owner=owner, task=task, due=due))
    return out


def make_notes(seg_id: int, obj: dict[str, Any] | None, raw: str) -> SegmentNotes:
    if obj is None:
        return SegmentNotes(
            seg_id=seg_id,
            summary=raw.strip(),
            key_points=[],
            decisions=[],
            action_items=[],
            open_questions=[],
            terms=[],
            raw=True,
        )
    summary = obj.get("summary")
    return SegmentNotes(
        seg_id=seg_id,
        summary=" ".join(_strings(summary)) if isinstance(summary, list) else _str(summary),
        key_points=_strings(obj.get("key_points")),
        decisions=_strings(obj.get("decisions")),
        action_items=_actions(obj.get("action_items")),
        open_questions=_strings(obj.get("open_questions")),
        terms=_strings(obj.get("terms")),
    )


# --- model (MLX) ---------------------------------------------------------------------------


class TinyAya:
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        return generate(
            self.model,
            self.tokenizer,
            text,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


def load(path: Path) -> TinyAya:
    from mlx_lm import load as load_lm

    loaded = load_lm(str(path))  # (model, tokenizer); typed as maybe + config
    return TinyAya(loaded[0], loaded[1])


# --- stage ---------------------------------------------------------------------------------


def run_stage(ctx: StageContext) -> None:
    cfg = ctx.config.map
    budget = ctx.config.fuse.max_segment_tokens
    segments = schemas.load(ctx.run_dir, SegmentsFile).segments
    slides = {s.slide_id: s for s in schemas.load(ctx.run_dir, SlidesFile).slides}
    prompt = load_prompt("map")
    retry = f"{prompt}\n\n{load_prompt('map_retry')}"
    notes: list[SegmentNotes] = []
    retried = failed = 0
    if segments:
        with ctx.model("map") as lm:
            for k, seg in enumerate(segments):
                ctx.report(k / len(segments), f"segment {k + 1}/{len(segments)}")
                seg_slides = [slide_text(slides[i]) for i in seg.slide_ids]
                values = prompt_values(seg, seg_slides, ctx.config.target_language)
                size = lm.count_tokens(values["transcript"] + "".join(seg_slides))
                if size > budget:
                    ctx.log.warning("map: segment %d is %d tokens (budget %d)", k, size, budget)
                obj = None
                raw = ""
                for attempt, template in enumerate((prompt, retry)):
                    retried += attempt
                    raw = lm.generate(
                        fill(template, **values),
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        seed=ctx.config.seed,
                    )
                    if (obj := parse(raw)) is not None:
                        break
                if obj is None:
                    failed += 1
                    ctx.log.warning("map: segment %d kept as raw text: %r", k, raw[:200])
                notes.append(make_notes(seg.seg_id, obj, raw))
    ctx.log.info("map: %d segments, %d retried, %d kept as raw text", len(notes), retried, failed)
    schemas.save(ctx.run_dir, MapFile(target_language=ctx.config.target_language, notes=notes))


STAGE = Stage(
    name="map",
    run=run_stage,
    outputs=(MapFile.FILENAME,),
    format_version=MapFile.FORMAT_VERSION,
    config_keys=("seed", "target_language", "map", "models.map"),
    prompts=("map", "map_retry"),
    models=("map",),
)
