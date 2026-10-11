"""Run configuration (PLAN §3, §4, §5, §12).

One frozen pydantic model, grouped by stage so each stage hashes only its own section
(pipeline.stage_hashes). Storage paths come from the environment and never enter a hash.
"""

import hashlib
import os
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelSpec(_Section):
    repo: str
    revision: str  # pinned commit sha (PLAN §12, D6)
    subfolder: str | None = None


# Pinned in Phase 0 (NOTES.md 2026-10-10, D6).
DEFAULT_MODELS: dict[str, ModelSpec] = {
    "vad": ModelSpec(
        repo="mlx-community/silero-vad-v6",
        revision="2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06",
    ),
    "transcribe": ModelSpec(
        repo="mlx-community/cohere-transcribe-03-2026-mlx-8bit",
        revision="a0acb7f93cd32d82c4fbf801b6d8fb39d20c509f",
        subfolder="mlx-int8",
    ),
    "vision": ModelSpec(
        repo="mlx-community/North-Micro-Vision-Instruct-8bit",
        revision="26e6b3aef507899fbf86057593d99fc12418e26f",
    ),
    "map": ModelSpec(
        repo="mlx-community/tiny-aya-global-8bit-mlx",
        revision="5d5dc68b477abd7d583563a515bed98604d45d52",
    ),
    "reduce": ModelSpec(
        repo="mlx-community/c4ai-command-r7b-12-2024-4bit",
        revision="2ae601b3d780410caacd2148c6c96e620144f22d",
    ),
    "judge": ModelSpec(
        repo="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        revision="50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
    ),
}


# Target languages: Command R7B's 23 (model card, Phase 0, D7). Prompts name the language in
# English; "Other (beta)" languages (Phase 6) fall back to the code itself.
TARGET_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def language_name(code: str) -> str:
    return TARGET_LANGUAGES.get(code, code)


class IngestConfig(_Section):
    # ffmpeg select='gt(scene,…)'. A change between two white text slides scores 0.04–0.07
    # (Phase 4b-1 eval), so 0.3 kept only the first slide of a deck recording.
    scene_threshold: float = 0.02
    max_edge_px: int = 1600  # frames and slide pages, long edge
    # 16×16 phash (256 bits): an 8×8 one put consecutive text slides only 4–8 bits apart,
    # inside the old limit of 6; at 16×16 they are 38–60 apart.
    phash_size: int = 16
    dedupe_max_distance: int = 24  # Hamming distance to the last kept frame


class VadConfig(_Section):
    chunker: Literal["vad", "silence"] = "vad"
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 100
    pad_ms: int = 30
    merge_gap_s: float = 0.5
    max_chunk_s: float = 30.0  # D2
    # Runs further apart than this never share a chunk, so long non-speech stretches between
    # nearby runs are never sent to Transcribe (it makes up text from them, Phase 0).
    max_inner_gap_s: float = 2.0
    split_search_s: float = 5.0  # a run over max_chunk_s is cut at its least-speechy frame here


class VisionConfig(_Section):
    max_tokens: int = 1024
    temperature: float = 0.0


class FuseConfig(_Section):
    window_s: float = 300.0  # D12
    align_threshold: float = 0.15  # D11
    max_segment_tokens: int = 5000  # transcript + slide text per segment, for Tiny Aya (§5.4)


class MapConfig(_Section):
    max_tokens: int = 1024
    temperature: float = 0.3


class ReduceConfig(_Section):
    max_tokens: int = 2048
    temperature: float = 0.0
    max_input_tokens: int = 32_000  # above this: two-pass reduce (§4)
    citation_min_score: float = 0.1  # D19


class Config(_Section):
    source_language: str = "en"
    target_language: str = "en"
    seed: int = 0  # mx.random.seed before every generating stage (§3)
    memory_policy: Literal["resident", "sequential"] = "resident"
    source_mode: Literal["link", "copy"] = "link"
    log_level: Literal["INFO", "DEBUG"] = "INFO"
    ingest: IngestConfig = IngestConfig()
    vad: VadConfig = VadConfig()
    vision: VisionConfig = VisionConfig()
    fuse: FuseConfig = FuseConfig()
    map: MapConfig = MapConfig()
    reduce: ReduceConfig = ReduceConfig()
    models: dict[str, ModelSpec] = Field(default_factory=lambda: dict(DEFAULT_MODELS))


def runs_dir() -> Path:
    return Path(os.environ.get("COHR_RUNS_DIR", "runs"))


def hf_home() -> Path | None:
    """Model cache location; None means huggingface_hub's default (PLAN §12)."""
    value = os.environ.get("COHR_HF_HOME")
    return Path(value).expanduser() if value else None


def load_prompt(name: str) -> str:
    """Read src/cohr/prompts/<name>.md (prompt text never lives in code)."""
    return (resources.files("cohr") / "prompts" / f"{name}.md").read_text(encoding="utf-8")


def prompt_hash(name: str) -> str:
    return hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()
