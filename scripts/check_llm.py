"""Phase 0 checks for the text models (PLAN §5.5, §5.6, §9; decision D7). Throwaway.

  uv run python scripts/check_llm.py aya  REPO REVISION FIXTURES_DIR
  uv run python scripts/check_llm.py r7b  REPO REVISION FIXTURES_DIR

aya: Map prompt on the fixture as one segment, target en/de/ja x 4 seeds at temperature 0.3.
     Reports valid-JSON rate (go/no-go: >= 90%), output language, timing.
r7b: lists the tokenizer's chat templates, renders the `rag` template with documents and
     enable_citations, generates (up to 3 tries) and parses <co>…</co: …> citations.
"""

import json
import re
import sys
import time
from pathlib import Path

PROMPTS = Path(__file__).parents[1] / "src/cohr/prompts"
MAP_KEYS = {"summary", "key_points", "decisions", "action_items", "open_questions", "terms"}
CITE_RE = re.compile(r"<co>(.*?)</co:\s*([^>]+)>", re.S)
LANG_NAMES = {"en": "English", "de": "German", "ja": "Japanese"}


def fill(template: str, **values: str) -> str:
    for k, v in values.items():
        template = template.replace("{" + k + "}", v)
    return template


def slides_text(truth: dict) -> str:
    lines = []
    for s in truth["slides"]:
        lines.append(f"[slide {s['slide_id']}] {s['title']}")
        lines += [f"- {b}" for b in s["bullets"]]
        for row in s["table"] or []:
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and obj.keys() >= MAP_KEYS else None


def gen(
    model, tokenizer, prompt: str, max_tokens: int, temp: float, seed: int
) -> tuple[str, float]:
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    t = time.time()
    text = generate(
        model, tokenizer, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp)
    )
    return text, time.time() - t


def check_aya(model, tokenizer, truth: dict) -> dict:
    from lingua import LanguageDetectorBuilder

    detector = LanguageDetectorBuilder.from_all_languages().build()
    template = (PROMPTS / "map.md").read_text()
    runs = []
    for lang, name in LANG_NAMES.items():
        for seed in range(4):
            user = fill(
                template,
                target_language=name,
                time_range="00:00–01:32",
                slides=slides_text(truth),
                transcript=truth["script_text"],
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False
            )
            text, dt = gen(model, tokenizer, prompt, 1024, 0.3, seed)
            obj = parse_json(text)
            detected = None
            if obj:
                d = detector.detect_language_of(str(obj.get("summary", "")))
                detected = d.iso_code_639_1.name.lower() if d else None
            runs.append(
                {
                    "lang": lang,
                    "seed": seed,
                    "valid": obj is not None,
                    "s": round(dt, 1),
                    "detected": detected,
                    "head": text[:160] if not obj else None,
                    "obj": obj if seed == 0 else None,
                }
            )
    valid = sum(1 for r in runs if r["valid"])
    return {"valid_rate": round(valid / len(runs), 3), "runs": runs}


def check_r7b(model, tokenizer, truth: dict) -> dict:
    out: dict = {}
    tc = tokenizer.chat_template
    out["chat_templates"] = sorted(tc.keys()) if isinstance(tc, dict) else ["<single>"]
    tpl_cfg = (
        json.loads(Path(tokenizer.name_or_path, "tokenizer_config.json").read_text())
        if Path(tokenizer.name_or_path, "tokenizer_config.json").exists()
        else {}
    )
    if isinstance(tpl_cfg.get("chat_template"), list):
        out["chat_templates"] = [t["name"] for t in tpl_cfg["chat_template"]]

    docs = [{"title": "00:00–01:32 · slides 1,2,3", "snippet": truth["script_text"]}]
    for s in truth["slides"]:
        body = "\n".join(
            [s["title"], *s["bullets"]] + ["| " + " | ".join(r) + " |" for r in (s["table"] or [])]
        )
        docs.append({"title": f"slide {s['slide_id']}", "snippet": body})
    user = fill((PROMPTS / "reduce.md").read_text(), target_language="English")
    conv = [{"role": "user", "content": user}]
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    prompt = hf_tok.apply_chat_template(
        conv,
        documents=docs,
        chat_template="rag",
        enable_citations=True,
        tokenize=False,
        add_generation_prompt=True,
    )
    out["prompt_tokens"] = len(hf_tok.encode(prompt))
    out["prompt_tail"] = prompt[-700:]
    tries = []
    for seed in range(3):
        text, dt = gen(model, tokenizer, prompt, 1200, 0.0 if seed == 0 else 0.3, seed)
        cites = CITE_RE.findall(text)
        idx = sorted(
            {int(i) for _, ref in cites for i in re.findall(r"\d+", ref.split(":", 1)[-1])}
        )
        tries.append(
            {
                "seed": seed,
                "s": round(dt, 1),
                "citations": len(cites),
                "doc_indices": idx,
                "valid_indices": all(0 <= i < len(docs) for i in idx),
                "markers": [
                    m
                    for m in (
                        "## SUMMARY",
                        "## KEY_POINTS",
                        "## DECISIONS",
                        "## ACTION_ITEMS",
                        "## OPEN_QUESTIONS",
                    )
                    if m in text
                ],
                "text": text,
            }
        )
        if cites:
            break
    out["tries"] = tries
    return out


def main() -> None:
    mode, repo, revision, fx = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    import mlx.core as mx
    from mlx_lm import load

    out: dict = {"mode": mode, "repo": repo}
    mx.reset_peak_memory()
    t = time.time()
    loaded = load(repo, revision=revision)
    model, tokenizer = loaded[0], loaded[1]
    out["load_s"] = round(time.time() - t, 2)
    out["load_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    truth = json.loads((fx / "truth.json").read_text())
    mx.reset_peak_memory()
    out.update(
        check_aya(model, tokenizer, truth) if mode == "aya" else check_r7b(model, tokenizer, truth)
    )
    out["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
