"""PLAN §3, §12: config defaults, pinned revisions, prompt hashing, env paths."""

import os
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import cohr  # noqa: F401  (sets the telemetry env var on import)
from cohr import config
from cohr.config import Config


def test_telemetry_disabled_on_import() -> None:
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_every_model_is_pinned_to_a_commit() -> None:
    for name, spec in config.DEFAULT_MODELS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), name
    assert config.DEFAULT_MODELS["transcribe"].subfolder == "mlx-int8"


def test_config_is_frozen_and_strict() -> None:
    cfg = Config()
    with pytest.raises(ValidationError):
        cfg.seed = 1  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError):
        Config.model_validate({"sed": 1})


def test_config_round_trips_through_manifest_json() -> None:
    cfg = Config(target_language="ja", memory_policy="sequential")
    assert Config.model_validate(cfg.model_dump(mode="json")) == cfg


@pytest.mark.parametrize("name", ["vision", "map", "reduce"])
def test_prompts_load_and_hash(name: str) -> None:
    text = config.load_prompt(name)
    assert text.strip()
    assert config.prompt_hash(name) == config.prompt_hash(name)
    assert len(config.prompt_hash(name)) == 64


def test_storage_paths_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHR_RUNS_DIR", str(tmp_path / "r"))
    assert config.runs_dir() == tmp_path / "r"
    monkeypatch.delenv("COHR_HF_HOME", raising=False)
    assert config.hf_home() is None
    monkeypatch.setenv("COHR_HF_HOME", "~/models")
    assert config.hf_home() == Path.home() / "models"


def test_regional_map_model_replaces_map_for_its_target_only() -> None:
    # D18: map_model_by_lang swaps models["map"] for a matching target; Map's and translate's
    # "models.map" hash key moves for that language only, and adding the field moved nothing.
    from cohr import pipeline
    from cohr.config import ModelSpec

    fire = ModelSpec(repo="mlx-community/tiny-aya-fire", revision="abc", quantize_bits=8)
    by_lang = {"hi": fire}
    assert Config(target_language="hi", map_model_by_lang=by_lang).models["map"] == fire
    de = Config(target_language="de", map_model_by_lang=by_lang)
    assert de.models["map"] == Config().models["map"]
    stage = next(s for s in pipeline.default_stages() if s.name == "map")
    plain = Config(target_language="de")
    assert pipeline.config_hash(stage, de) == pipeline.config_hash(stage, plain)
    hi = Config(target_language="hi", map_model_by_lang=by_lang)
    assert pipeline.config_hash(stage, hi) != pipeline.config_hash(
        stage, Config(target_language="hi")
    )
    # unset quantize_bits isn't dumped (existing hashes); a set one is, and round-trips
    assert "quantize_bits" not in Config().models["map"].model_dump()
    assert Config.model_validate(hi.model_dump(mode="json")) == hi
