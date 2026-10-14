"""ModelManager (PLAN §4, §12): pinned downloads, lazy loading, memory policy, one lock.

Runtime loads never touch the network: each model is resolved from the cache with
`local_files_only=True` and loaded from its local path. Only `download()` (the explicit
`cohr models download` step) talks to the Hub, after a free-disk check.

MLX and the model packages are imported inside functions only; unit tests run without a GPU.
"""

import contextlib
import gc
import logging
import os
import shutil
import sys
import threading
from collections.abc import Callable, Generator, Iterable, Mapping
from pathlib import Path
from typing import Any

from cohr import storage
from cohr.config import Config, ModelSpec, hf_home

log = logging.getLogger("cohr")

PIPELINE_MODELS = ("vad", "transcribe", "vision", "map", "reduce")  # the judge is eval-only
DOWNLOAD_MARGIN_BYTES = 2 * storage.GB
ALWAYS_RELEASE = frozenset({"vad"})  # ~2 MB, loaded for its step only (§4)

Loader = Callable[[Path], Any]
Resolver = Callable[[ModelSpec], Path]


class ModelLoadError(RuntimeError):
    def __init__(self, name: str, message: str):
        super().__init__(message)
        self.name = name


class ModelsMissing(ModelLoadError):
    def __init__(self, missing: Mapping[str, ModelSpec]):
        listed = ", ".join(f"{n} ({s.repo}@{s.revision[:7]})" for n, s in missing.items())
        super().__init__(
            next(iter(missing)),
            f"Models not downloaded: {listed}. Run `uv run cohr models download`.",
        )
        self.missing = dict(missing)


# --- cache ---------------------------------------------------------------------------------


def cache_dir() -> Path | None:
    """`<COHR_HF_HOME>/hub` when set; None means huggingface_hub's default cache."""
    home = hf_home()
    return home / "hub" if home else None


def _effective_cache_dir() -> Path:
    from huggingface_hub import constants

    return cache_dir() or Path(constants.HF_HUB_CACHE)


def converted_path(spec: ModelSpec, cache: Path | None = None) -> Path:
    """Where `cohr models download` writes a spec's local quantized copy (D18)."""
    root = cache if cache is not None else _effective_cache_dir()
    name = f"{spec.repo.replace('/', '--')}@{spec.revision}-q{spec.quantize_bits}"
    return root / "cohr-converted" / name


def local_path(spec: ModelSpec, cache: Path | None = None) -> Path:
    """Local snapshot dir for a pinned revision (+ subfolder), or its converted copy for a
    spec with `quantize_bits`. Never downloads or converts."""
    if spec.quantize_bits:
        path = converted_path(spec, cache)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{spec.repo}@{spec.revision} is not converted ({path})")
        return path
    return _snapshot_path(spec, cache)


def _snapshot_path(spec: ModelSpec, cache: Path | None = None) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        root = snapshot_download(
            spec.repo,
            revision=spec.revision,
            cache_dir=cache if cache is not None else cache_dir(),
            local_files_only=True,
        )
    except LocalEntryNotFoundError as e:
        raise FileNotFoundError(f"{spec.repo}@{spec.revision} is not in the cache") from e
    path = Path(root) / spec.subfolder if spec.subfolder else Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is missing from the cached snapshot")
    return path


def size_on_disk(path: Path) -> int:
    """Bytes of a cached snapshot (its files are symlinks into blobs/)."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def repo_bytes(spec: ModelSpec) -> int:
    """Download size of a pinned revision, from the Hub (network)."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(spec.repo, revision=spec.revision, files_metadata=True)
    return sum(s.size or 0 for s in info.siblings or [])


def download(
    specs: Mapping[str, ModelSpec],
    *,
    cache: Path | None = None,
    size_of: Callable[[ModelSpec], int] = repo_bytes,
    fetch: Callable[[ModelSpec, Path | None], None] | None = None,
    convert: Callable[[ModelSpec, Path | None], None] | None = None,
) -> list[str]:
    """Download the specs that aren't cached yet, and convert those with `quantize_bits`.
    Refuses (DiskSpaceError) before downloading anything if free space is below their total
    size + 2 GB (§12). Returns the names fetched."""
    cache = cache if cache is not None else cache_dir()
    missing = {}
    for name, spec in specs.items():
        try:
            local_path(spec, cache)
        except FileNotFoundError:
            missing[name] = spec
    if not missing:
        return []
    needed = sum(size_of(s) for s in missing.values())
    target = cache if cache is not None else _effective_cache_dir()
    storage.check_free_space(target, needed, margin=DOWNLOAD_MARGIN_BYTES)
    fetch = fetch or _snapshot_download
    convert = convert or _convert
    for name, spec in missing.items():
        log.info("downloading %s (%s@%s)", name, spec.repo, spec.revision)
        fetch(spec, cache)
        if spec.quantize_bits:
            log.info("converting %s to %d-bit", name, spec.quantize_bits)
            convert(spec, cache)
    return list(missing)


def _snapshot_download(spec: ModelSpec, cache: Path | None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(spec.repo, revision=spec.revision, cache_dir=cache)


QUANT_GROUP_SIZE = 64  # as in the default Map model's 8-bit repo (tiny-aya-global-8bit-mlx)


def _convert(spec: ModelSpec, cache: Path | None) -> None:
    """mlx_lm.convert the cached snapshot into converted_path, via a temp dir (atomic)."""
    from mlx_lm import convert

    dest = converted_path(spec, cache)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    convert(
        hf_path=str(_snapshot_path(spec, cache)),
        mlx_path=str(tmp),
        quantize=True,
        q_bits=spec.quantize_bits,
        q_group_size=QUANT_GROUP_SIZE,
    )
    os.replace(tmp, dest)


# --- memory --------------------------------------------------------------------------------


class MlxMemory:
    """mx memory stats and cache clearing, only once something has imported MLX.

    Stages that load no model (ingest) never import MLX, so the CLI and CI stay GPU-free.
    """

    @staticmethod
    def _mx() -> Any | None:
        return sys.modules.get("mlx.core")

    def reset_peak(self) -> None:
        if mx := self._mx():
            mx.reset_peak_memory()

    def peak(self) -> int | None:
        mx = self._mx()
        return int(mx.get_peak_memory()) if mx else None

    def clear_cache(self) -> None:
        if mx := self._mx():
            mx.clear_cache()

    def active(self) -> int | None:
        mx = self._mx()
        return int(mx.get_active_memory()) if mx else None

    def limit(self) -> int | None:
        """The GPU working-set limit (26.8 GB measured on the M2 Max 32 GB, §4)."""
        mx = self._mx()
        return int(mx.device_info()["max_recommended_working_set_size"]) if mx else None


# --- manager -------------------------------------------------------------------------------


def default_loaders() -> dict[str, Loader]:
    from cohr import asr, judge, mapper, reduce, vad, vision

    return {
        "vad": vad.load,
        "transcribe": asr.load,
        "vision": vision.load,
        "map": mapper.load,
        "reduce": reduce.load,
        "judge": judge.load,  # eval only (§18)
    }


class ModelManager:
    """One per process (the UI caches it). All model use goes through `use()` under one lock,
    so only one job touches the GPU at a time (§7)."""

    def __init__(
        self,
        config: Config,
        *,
        loaders: Mapping[str, Loader] | None = None,
        resolve: Resolver | None = None,
        memory: MlxMemory | None = None,
    ):
        self.config = config
        self.lock = threading.RLock()
        self._loaders = dict(loaders) if loaders is not None else None
        self._resolve = resolve or local_path
        self.memory = memory or MlxMemory()
        self._loaded: dict[str, Any] = {}

    def spec(self, name: str) -> ModelSpec:
        try:
            return self.config.models[name]
        except KeyError:
            raise ModelLoadError(name, f"no model {name!r} in config.models") from None

    def missing(self, names: Iterable[str]) -> dict[str, ModelSpec]:
        out = {}
        for name in names:
            try:
                self._resolve(self.spec(name))
            except FileNotFoundError:
                out[name] = self.spec(name)
        return out

    def require(self, names: Iterable[str]) -> None:
        """Raise ModelsMissing listing every model that isn't downloaded."""
        if missing := self.missing(names):
            raise ModelsMissing(missing)

    def loaded(self) -> list[str]:
        return list(self._loaded)

    def set_config(self, config: Config) -> None:
        """Switch to a run's config before its job (the UI shares one manager). Releases models
        whose pinned spec changed, and everything when the policy becomes sequential."""
        with self.lock:
            old, self.config = self.config, config
            for name in self.loaded():
                if config.memory_policy == "sequential" or old.models.get(name) != (
                    config.models.get(name)
                ):
                    self.release(name)

    def _load(self, name: str) -> Any:
        loaders = self._loaders if self._loaders is not None else default_loaders()
        if name not in loaders:
            raise ModelLoadError(name, f"no loader for model {name!r}")
        spec = self.spec(name)
        try:
            path = self._resolve(spec)
        except FileNotFoundError:
            raise ModelsMissing({name: spec}) from None
        log.info("loading %s from %s", name, path)
        try:
            model = loaders[name](path)
        except Exception as e:
            raise ModelLoadError(name, f"could not load {name} ({spec.repo}): {e}") from e
        self._loaded[name] = model
        return model

    @contextlib.contextmanager
    def use(self, name: str) -> Generator[Any]:
        """Hold the lock, load on first use, release afterwards per the memory policy (§4).

        Load failures raise ModelLoadError (or ModelsMissing) before the body runs.
        """
        with self.lock:
            model = self._loaded.get(name)
            if model is None:
                model = self._load(name)
            try:
                yield model
            finally:
                del model
                if self.config.memory_policy == "sequential" or name in ALWAYS_RELEASE:
                    self.release(name)

    def release(self, name: str | None = None) -> None:
        """Unload one model (or all): drop the reference, collect, clear MLX's buffer cache."""
        with self.lock:
            for key in [name] if name else list(self._loaded):
                self._loaded.pop(key, None)
            gc.collect()
            self.memory.clear_cache()

    def reset_peak(self) -> None:
        self.memory.reset_peak()

    def peak_memory(self) -> int | None:
        return self.memory.peak()


def cached_report(config: Config, names: Iterable[str]) -> list[tuple[str, ModelSpec, int | None]]:
    """(name, spec, bytes on disk or None if not cached) for `cohr models`."""
    out = []
    for name in names:
        spec = config.models[name]
        try:
            size: int | None = size_on_disk(local_path(spec))
        except FileNotFoundError:
            size = None
        out.append((name, spec, size))
    return out
