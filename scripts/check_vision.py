"""Phase 0 Micro Vision check (PLAN §5.3, §9). Throwaway.

  uv run python scripts/check_vision.py REPO REVISION FIXTURES_DIR

For each fixture slide: JSON-only prompt (src/cohr/prompts/vision.md), temperature 0, one retry
with "Return only valid JSON". Reports JSON validity, title match, table numbers found, timing.
"""

import json
import re
import sys
import time
from pathlib import Path

PROMPT = (Path(__file__).parents[1] / "src/cohr/prompts/vision.md").read_text()
RETRY = "Return only valid JSON."
KEYS = {"title", "text_md", "tables_md", "chart_notes", "visual_notes"}


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and obj.keys() >= KEYS else None


def main() -> None:
    repo, revision, fx = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    import mlx.core as mx
    from mlx_vlm import apply_chat_template, generate, load

    out: dict = {"repo": repo}
    mx.reset_peak_memory()
    t = time.time()
    model, processor = load(repo, revision=revision)
    out["load_s"] = round(time.time() - t, 2)
    out["load_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)

    truth = json.loads((fx / "truth.json").read_text())
    results = []
    mx.reset_peak_memory()
    for s in truth["slides"]:
        img = str(fx / f"slide_{s['slide_id']}.png")
        r: dict = {"slide_id": s["slide_id"]}
        t = time.time()
        obj = None
        for attempt, text in enumerate((PROMPT, PROMPT + "\n\n" + RETRY)):
            prompt = str(apply_chat_template(processor, model.config, text, num_images=1))
            res = generate(
                model,
                processor,  # ty: ignore[invalid-argument-type]
                prompt,
                image=[img],
                max_tokens=1024,
                temperature=0.0,
                verbose=False,
            )
            raw = res.text if hasattr(res, "text") else str(res)
            obj = parse(raw)
            r["attempts"] = attempt + 1
            if obj:
                break
        r["s"] = round(time.time() - t, 2)
        r["valid_json"] = obj is not None
        if obj:
            r["title"] = obj["title"]
            r["title_ok"] = obj["title"].strip().lower() == s["title"].lower()
            blob = json.dumps(obj, ensure_ascii=False)
            cells = [c for row in (s["table"] or [])[1:] for c in row]
            r["table_cells_found"] = f"{sum(c in blob for c in cells)}/{len(cells)}"
            r["json"] = obj
        else:
            r["raw_head"] = raw[:400]
        results.append(r)
    out["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    out["slides"] = results
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
