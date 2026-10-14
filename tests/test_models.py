"""PLAN §4, §12: lazy loading, memory policy, one lock, missing models, download disk check."""

import threading
from collections import namedtuple
from pathlib import Path

import pytest

from cohr import models
from cohr.config import Config, ModelSpec
from cohr.models import ModelLoadError, ModelManager, ModelsMissing
from cohr.storage import GB, DiskSpaceError
from tests.conftest import NullMemory


class Counting:
    def __init__(self) -> None:
        self.loads: list[str] = []

    def loader(self, name: str):
        def load(path: Path) -> str:
            self.loads.append(name)
            return f"{name}@{path.name}"

        return load


def manager(tmp_path: Path, policy: str = "resident", cached=("vision", "transcribe", "vad")):
    counting = Counting()
    memory = NullMemory()

    def resolve(spec: ModelSpec) -> Path:
        name = next(n for n, s in Config().models.items() if s == spec)
        if name not in cached:
            raise FileNotFoundError(name)
        return tmp_path / name

    m = ModelManager(
        Config(memory_policy=policy),  # ty: ignore[invalid-argument-type]
        loaders={n: counting.loader(n) for n in ("vad", "transcribe", "vision")},
        resolve=resolve,
        memory=memory,
    )
    return m, counting, memory


def test_resident_loads_once_and_keeps(tmp_path: Path) -> None:
    m, counting, memory = manager(tmp_path)
    with m.use("vision") as a:
        assert a == "vision@vision"
    with m.use("vision"):
        pass
    assert counting.loads == ["vision"]
    assert m.loaded() == ["vision"]
    assert memory.clears == 0
    m.release()
    assert m.loaded() == [] and memory.clears == 1


def test_sequential_releases_after_each_use(tmp_path: Path) -> None:
    m, counting, memory = manager(tmp_path, "sequential")
    for _ in range(2):
        with m.use("transcribe"):
            pass
    assert counting.loads == ["transcribe", "transcribe"]
    assert m.loaded() == [] and memory.clears == 2


def test_vad_always_released(tmp_path: Path) -> None:
    m, _, _ = manager(tmp_path)
    with m.use("vad"):
        assert m.loaded() == ["vad"]
    assert m.loaded() == []


def test_require_lists_every_missing_model(tmp_path: Path) -> None:
    m, _, _ = manager(tmp_path, cached=("vad",))
    m.require(["vad"])
    with pytest.raises(ModelsMissing, match="cohr models download") as e:
        m.require(["vad", "transcribe", "vision"])
    assert list(e.value.missing) == ["transcribe", "vision"]
    with pytest.raises(ModelsMissing), m.use("vision"):
        pass


def test_loader_errors_become_model_load_error(tmp_path: Path) -> None:
    def broken(_path: Path) -> None:
        raise ValueError("strict load: 12 missing keys")

    m = ModelManager(Config(), loaders={"vision": broken}, resolve=lambda _s: tmp_path)
    with pytest.raises(ModelLoadError, match="12 missing keys"), m.use("vision"):
        pass
    with pytest.raises(ModelLoadError, match="no loader"), m.use("map"):
        pass
    assert m.loaded() == []


def test_one_job_at_a_time(tmp_path: Path) -> None:
    m, _, _ = manager(tmp_path)
    inside = threading.Event()
    leave = threading.Event()
    entered: list[str] = []

    def job(name: str) -> None:
        with m.use("vision"):
            entered.append(name)
            inside.set()
            leave.wait(2)

    first = threading.Thread(target=job, args=("first",))
    first.start()
    inside.wait(2)
    second = threading.Thread(target=job, args=("second",))
    second.start()
    second.join(0.1)
    assert entered == ["first"]  # second waits for the lock
    leave.set()
    first.join(2)
    second.join(2)
    assert entered == ["first", "second"]


def test_runtime_resolution_never_downloads(tmp_path: Path) -> None:
    spec = Config().models["vad"]
    with pytest.raises(FileNotFoundError):
        models.local_path(spec, tmp_path / "empty-cache")


def test_cache_dir_follows_cohr_hf_home(tmp_path: Path) -> None:
    assert models.cache_dir() == tmp_path / "hf" / "hub"  # conftest sets COHR_HF_HOME


Usage = namedtuple("Usage", "total used free")


def test_download_refuses_without_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.disk_usage", lambda _p: Usage(100 * GB, 95 * GB, 5 * GB))
    fetched: list[str] = []
    specs = {n: Config().models[n] for n in ("vision", "map")}  # 4 GB each, not cached
    with pytest.raises(DiskSpaceError, match="need 10.0 GB"):
        models.download(
            specs,
            cache=tmp_path / "cache",
            size_of=lambda _s: 4 * GB,
            fetch=lambda s, _c: fetched.append(s.repo),
        )
    assert fetched == []  # nothing starts before the check passes


def test_download_fetches_only_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.disk_usage", lambda _p: Usage(100 * GB, 10 * GB, 90 * GB))
    fetched: list[str] = []
    specs = {n: Config().models[n] for n in ("vision", "map")}
    names = models.download(
        specs,
        cache=tmp_path / "c",
        size_of=lambda _s: GB,
        fetch=lambda s, _c: fetched.append(s.repo),
    )
    assert names == ["vision", "map"]
    assert fetched == [specs["vision"].repo, specs["map"].repo]


def test_download_converts_a_quantized_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # D18: a regional model with no 8-bit MLX repo is converted once, at download time.
    monkeypatch.setattr("shutil.disk_usage", lambda _p: Usage(100 * GB, 10 * GB, 90 * GB))
    spec = ModelSpec(repo="mlx-community/tiny-aya-fire", revision="abc", quantize_bits=8)
    cache = tmp_path / "c"
    with pytest.raises(FileNotFoundError, match="not converted"):
        models.local_path(spec, cache)

    def convert(s: ModelSpec, c: Path | None) -> None:
        out = models.converted_path(s, c)
        out.mkdir(parents=True)
        (out / "config.json").write_text("{}")

    names = models.download(
        {"map": spec}, cache=cache, size_of=lambda _s: GB, fetch=lambda s, c: None, convert=convert
    )
    assert names == ["map"]
    assert models.local_path(spec, cache) == cache / "cohr-converted" / (
        "mlx-community--tiny-aya-fire@abc-q8"
    )
    # converted already: nothing to fetch
    assert models.download({"map": spec}, cache=cache, size_of=lambda _s: GB) == []


def test_set_config_keeps_models_when_only_language_changes(tmp_path: Path) -> None:
    m, counting, _ = manager(tmp_path)
    with m.use("vision"):
        pass
    m.set_config(Config(target_language="de"))
    assert m.loaded() == ["vision"] and m.config.target_language == "de"
    with m.use("vision"):
        pass
    assert counting.loads == ["vision"]


def test_set_config_sequential_releases_everything(tmp_path: Path) -> None:
    m, _, _ = manager(tmp_path)
    with m.use("vision"):
        pass
    m.set_config(Config(memory_policy="sequential"))
    assert m.loaded() == []


def test_set_config_releases_a_model_whose_spec_changed(tmp_path: Path) -> None:
    m, _, _ = manager(tmp_path)
    for name in ("vision", "transcribe"):
        with m.use(name):
            pass
    other = dict(Config().models, vision=ModelSpec(repo="x/y", revision="0" * 40))
    m.set_config(Config(models=other))
    assert m.loaded() == ["transcribe"]


def test_memory_stats_without_mlx_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.delitem(sys.modules, "mlx.core", raising=False)
    mem = models.MlxMemory()
    assert mem.active() is None and mem.limit() is None and mem.peak() is None
