"""Phase 6b-2 step 0: re-generate each eval run's Map replies and count tokens. Throwaway.

  uv run python scripts/map_tokens.py [SEED]   (default 1: the seed the runs were made with)

For every quick run under evals/data/runs (long-en skipped), Map's prompt is rebuilt per
segment and generated again with the same seed and the default config, so the reply is the
one the run got (key points differ only where the number guard repaired them). Prints the
prompt and reply token counts and the reply's end: a reply at `map.max_tokens` without
`<|END_RESPONSE|>` was cut off. Loads Tiny Aya (~3.6 GB).
"""

import json
import sys
from pathlib import Path

from cohr import schemas
from cohr.config import Config, load_prompt
from cohr.fuse import slide_text
from cohr.mapper import fill, load, make_notes, parse, prompt_values
from cohr.models import local_path
from cohr.schemas import MapFile, SegmentsFile, SlidesFile


def main(seed: int = 1, root: Path = Path()) -> None:
    cfg = Config()
    lm = load(local_path(cfg.models["map"]))
    prompt = load_prompt("map")
    for run in sorted((root / "evals/data/runs").glob("*/*")):
        if run.parent.name == "long-en":
            continue
        slides = {s.slide_id: s for s in schemas.load(run, SlidesFile).slides}
        stored = {n.seg_id: n for n in schemas.load(run, MapFile).notes}
        for seg in schemas.load(run, SegmentsFile).segments:
            seg_slides = [slide_text(slides[i]) for i in seg.slide_ids]
            text = fill(prompt, **prompt_values(seg, seg_slides, cfg.target_language))
            raw = lm.generate(
                text, max_tokens=cfg.map.max_tokens, temperature=cfg.map.temperature, seed=seed
            )
            same = (
                make_notes(seg.seg_id, parse(raw), raw).key_points == stored[seg.seg_id].key_points
            )
            print(
                run.parent.name, seg.seg_id, "in", lm.count_tokens(text),
                "out", lm.count_tokens(raw), f"/ {cfg.map.max_tokens}",
                "same key points" if same else "key points differ (guard?)",
                json.dumps(raw[-40:]),
            )  # fmt: skip


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
