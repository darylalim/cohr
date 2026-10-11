"""Phase 0 exit check (PLAN §9): all five pipeline models chained on the §8 fixtures. Throwaway.

  uv run python scripts/spike.py [--mem32k]

VAD -> chunks -> Transcribe -> transcript; Micro Vision -> slide JSON; whole fixture as ONE
segment -> Tiny Aya Map JSON -> R7B `rag` (one document per segment and slide) -> notes ->
citation parse. All models stay loaded (`resident` policy, §4). Writes only to a temp dir.
Passes only if the notes contain at least one citation that resolves to the segment or a slide.

--mem32k: afterwards, with everything still loaded, runs R7B on a ~32K-token prompt and reports
peak memory against the GPU limit (§9 "all pipeline models loaded together").
"""

import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import fixtures  # noqa: E402
from check_llm import CITE_RE, fill, parse_json  # noqa: E402
from check_vad import pack  # noqa: E402
from check_vision import PROMPT as VISION_PROMPT  # noqa: E402
from check_vision import parse as parse_slide  # noqa: E402

SR = 16000
PROMPTS = Path(__file__).parents[1] / "src/cohr/prompts"
MODELS = {
    "vad": ("mlx-community/silero-vad-v6", "2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06"),
    "asr": (
        "mlx-community/cohere-transcribe-03-2026-mlx-8bit",
        "a0acb7f93cd32d82c4fbf801b6d8fb39d20c509f",
    ),
    "vision": (
        "mlx-community/North-Micro-Vision-Instruct-8bit",
        "26e6b3aef507899fbf86057593d99fc12418e26f",
    ),
    "map": ("mlx-community/tiny-aya-global-8bit-mlx", "5d5dc68b477abd7d583563a515bed98604d45d52"),
    "reduce": (
        "mlx-community/c4ai-command-r7b-12-2024-4bit",
        "2ae601b3d780410caacd2148c6c96e620144f22d",
    ),
}


class Stats:
    def __init__(self) -> None:
        import mlx.core as mx

        self.mx = mx
        self.rows: list[dict] = []

    def stage(self, name: str):
        stats = self

        class _Ctx:
            def __enter__(self):
                stats.mx.reset_peak_memory()
                self.t = time.time()

            def __exit__(self, *_exc):
                stats.rows.append(
                    {
                        "stage": name,
                        "s": round(time.time() - self.t, 2),
                        "peak_gb": round(stats.mx.get_peak_memory() / 1e9, 2),
                        "active_gb": round(stats.mx.get_active_memory() / 1e9, 2),
                    }
                )

        return _Ctx()


def llm_generate(model, tokenizer, prompt: str, max_tokens: int, temp: float, seed: int = 0):
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    return generate(
        model, tokenizer, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp)
    )


def main() -> None:
    mem32k = "--mem32k" in sys.argv
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_audio.vad import load as load_vad
    from mlx_lm import load as load_lm
    from mlx_speech.generation import CohereAsrModel
    from mlx_vlm import apply_chat_template
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm import load as load_vlm

    st = Stats()
    work = Path(tempfile.mkdtemp(prefix="cohr-spike-"))
    fixtures.main(work)
    truth = json.loads((work / "truth.json").read_text())
    audio, _ = sf.read(work / "meeting.wav", dtype="float32")
    dur = len(audio) / SR
    report: dict = {"work_dir": str(work), "audio_s": round(dur, 1)}

    # --- load everything (resident policy) ---
    with st.stage("load_all"):
        vad = load_vad(MODELS["vad"][0], revision=MODELS["vad"][1])
        asr_dir = Path(snapshot_download(MODELS["asr"][0], revision=MODELS["asr"][1])) / "mlx-int8"
        asr = CohereAsrModel.from_path(asr_dir)
        vlm, vlm_proc = load_vlm(MODELS["vision"][0], revision=MODELS["vision"][1])
        aya = load_lm(MODELS["map"][0], revision=MODELS["map"][1])
        r7b = load_lm(MODELS["reduce"][0], revision=MODELS["reduce"][1])
        mx.eval([])

    # --- VAD -> chunks ---
    with st.stage("vad"):
        ts = vad.get_speech_timestamps(audio, sample_rate=SR, return_seconds=True)
        chunks = pack([(float(x["start"]), float(x["end"])) for x in ts])
    (work / "chunks.json").write_text(json.dumps({"chunks": chunks}))

    # --- Transcribe ---
    with st.stage("transcribe"):
        transcript = [
            {
                "chunk_id": i,
                "t0": a,
                "t1": b,
                "text": asr.transcribe(
                    np.ascontiguousarray(audio[int(a * SR) : int(b * SR)]),
                    sample_rate=SR,
                    language="en",
                ).text,
            }
            for i, (a, b) in enumerate(chunks)
        ]
    (work / "transcript.json").write_text(json.dumps(transcript, indent=2))

    # --- Micro Vision ---
    slides = []
    with st.stage("vision"):
        for s in truth["slides"]:
            prompt = str(apply_chat_template(vlm_proc, vlm.config, VISION_PROMPT, num_images=1))
            res = vlm_generate(
                vlm,
                vlm_proc,  # ty: ignore[invalid-argument-type]
                prompt,
                image=[str(work / f"slide_{s['slide_id']}.png")],
                max_tokens=1024,
                temperature=0.0,
            )
            obj = parse_slide(res.text) or {"title": f"Slide {s['slide_id']}", "text_md": res.text}
            slides.append({"slide_id": s["slide_id"], **obj})
    (work / "slides.json").write_text(json.dumps(slides, indent=2, ensure_ascii=False))

    def slide_text(s: dict) -> str:
        tables = s.get("tables_md") or []
        return "\n".join(
            [f"[slide {s['slide_id']}] {s.get('title', '')}", str(s.get("text_md", ""))]
            + [t if isinstance(t, str) else json.dumps(t, ensure_ascii=False) for t in tables]
        )

    # --- one segment -> Tiny Aya Map ---
    seg_text = " ".join(str(t["text"]) for t in transcript)
    with st.stage("map"):
        user = fill(
            (PROMPTS / "map.md").read_text(),
            target_language="English",
            time_range=f"00:00–{int(dur // 60):02d}:{int(dur % 60):02d}",
            slides="\n\n".join(slide_text(s) for s in slides),
            transcript=seg_text,
        )
        prompt = aya[1].apply_chat_template(
            [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False
        )
        map_raw = llm_generate(aya[0], aya[1], prompt, 1024, 0.3)
        map_obj = parse_json(map_raw)
    report["map_valid_json"] = map_obj is not None
    (work / "map.json").write_text(json.dumps(map_obj or {"raw": map_raw}, indent=2))

    # --- R7B rag reduce with citations ---
    seg_title = f"00:00–{int(dur // 60):02d}:{int(dur % 60):02d} · slides 1,2,3"
    map_notes = json.dumps(map_obj, ensure_ascii=False) if map_obj else map_raw
    docs = [{"title": seg_title, "snippet": f"Notes: {map_notes}\n\nTranscript: {seg_text}"}]
    docs += [{"title": f"slide {s['slide_id']}", "snippet": slide_text(s)} for s in slides]
    doc_targets = [("seg", 0)] + [("slide", s["slide_id"]) for s in slides]
    hf_tok = getattr(r7b[1], "_tokenizer", r7b[1])
    with st.stage("reduce"):
        prompt = hf_tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": fill((PROMPTS / "reduce.md").read_text(), target_language="English"),
                }
            ],
            documents=docs,
            chat_template="rag",
            enable_citations=True,
            tokenize=False,
            add_generation_prompt=True,
        )
        notes_raw = llm_generate(r7b[0], r7b[1], prompt, 1200, 0.0)
    (work / "notes_raw.txt").write_text(notes_raw)

    # --- citation parse + resolve ---
    resolved, dropped = [], []
    for span, ref in CITE_RE.findall(notes_raw):
        for i in (int(x) for x in re.findall(r"\d+", ref.split(":", 1)[-1])):
            (resolved if 0 <= i < len(docs) else dropped).append((span.strip()[:80], i))
    report["citations"] = len(resolved)
    report["dropped_citations"] = len(dropped)
    report["cited_targets"] = sorted({doc_targets[i] for _, i in resolved})
    report["examples"] = [f"{span} -> {doc_targets[i]}" for span, i in resolved[:5]]
    report["section_markers"] = [m for m in re.findall(r"^## [A-Z_]+", notes_raw, re.M)]

    if mem32k:
        with st.stage("reduce_32k_prompt"):
            target, filler = 32000, seg_text
            big_docs = list(docs)
            while len(hf_tok.encode(json.dumps(big_docs))) < target:
                big_docs.append({"title": f"filler {len(big_docs)}", "snippet": filler * 4})
            prompt = hf_tok.apply_chat_template(
                [{"role": "user", "content": "Summarize the documents in 3 sentences."}],
                documents=big_docs,
                chat_template="rag",
                enable_citations=True,
                tokenize=False,
                add_generation_prompt=True,
            )
            report["mem32k_prompt_tokens"] = len(hf_tok.encode(prompt))
            llm_generate(r7b[0], r7b[1], prompt, 200, 0.0)

    report["stages"] = st.rows
    report["gpu_limit_gb"] = round(
        float(mx.device_info()["max_recommended_working_set_size"]) / 1e9, 2
    )
    report["PASS"] = len(resolved) > 0
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n--- notes_raw.txt ---\n" + notes_raw)
    sys.exit(0 if report["PASS"] else 1)


if __name__ == "__main__":
    main()
