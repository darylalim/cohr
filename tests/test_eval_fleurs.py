"""FLEURS in the full set (PLAN §18, 6e-1): sampling, streaming, scoring. No network, no models."""

import contextlib
import io
import tarfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from cohr import asr, evals, fleurs
from cohr.fleurs import Clip, FleursSpec

ROOT = Path(__file__).resolve().parents[1]

# (sentence id, file, raw text): sentence 10 is read by two speakers
ROWS = [
    (10, "a.wav", "Hello there."),
    (10, "b.wav", "Hello there."),
    (11, "c.wav", "One hundred percent sure."),
    (12, "d.wav", "Good night."),
    (13, "e.wav", "Last one."),
]


def tsv(rows: list[tuple[int, str, str]] = ROWS) -> str:
    return "".join(f"{i}\t{f}\t{t}\t{t.lower()}\tx\t16000\tMALE\n" for i, f, t in rows)


def wav_bytes(seconds: float = 0.5, sr: int = asr.SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sr), dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


def tar_gz(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        folder = tarfile.TarInfo("test")  # a directory entry first, as in FLEURS
        folder.type = tarfile.DIRTYPE
        tar.addfile(folder)
        for name in names:
            data = wav_bytes()
            info = tarfile.TarInfo(f"test/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeFs:
    def __init__(self, names: list[str]):
        self.tar = tar_gz(names)
        self.reads: list[str] = []

    def read_text(self, path: str, encoding: str | None = None) -> str:
        self.reads.append(path)
        return tsv()

    def open(self, path: str, mode: str = "rb") -> Any:
        self.reads.append(path)
        return io.BytesIO(self.tar)


class OfflineFs:
    def read_text(self, path: str, encoding: str | None = None) -> str:
        raise AssertionError("network used")

    def open(self, path: str, mode: str = "rb") -> Any:
        raise AssertionError("network used")


def spec(**kw: Any) -> FleursSpec:
    return FleursSpec.model_validate(
        {"revision": "abc", "clips": 2, "pool": 4, "languages": {"en": "en_us"}} | kw
    )


def test_full_set_holds_quick_and_lang_and_every_transcribe_language() -> None:
    full = evals.load_set("full", ROOT)
    quick, lang = evals.load_set("quick", ROOT), evals.load_set("lang", ROOT)
    assert full.items == [*quick.items, *lang.items]
    assert len({it.id for it in full.items}) == len(full.items)
    assert full.fleurs is not None
    assert sorted(full.fleurs.languages) == sorted(asr.LANGUAGES)
    assert (full.fleurs.clips, full.fleurs.pool) == (20, 100)
    assert full.fleurs.languages["zh"] == "cmn_hans_cn"


def test_parse_tsv_keys_rows_by_file_and_keeps_raw_text() -> None:
    rows = fleurs.parse_tsv(tsv())
    assert list(rows) == ["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"]
    assert rows["c.wav"] == Clip(file="c.wav", sentence_id=11, text="One hundred percent sure.")


def test_pick_takes_one_clip_per_sentence_by_seed() -> None:
    rows = fleurs.parse_tsv(tsv())
    members = ["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"]
    picked = fleurs.pick(members, rows, 4, seed=1)
    assert [c.file for c in picked] == ["a.wav", "c.wav", "d.wav", "e.wav"]  # b repeats 10
    assert fleurs.pick(members, rows, 2, seed=7) == fleurs.pick(members, rows, 2, seed=7)
    draws = {tuple(c.file for c in fleurs.pick(members, rows, 2, seed=s)) for s in range(20)}
    assert len(draws) > 1  # the seed matters
    with pytest.raises(ValueError, match="only 4 sentences"):
        fleurs.pick(members, rows, 5, seed=1)


def test_read_members_stops_after_n_files() -> None:
    out = fleurs.read_members(io.BytesIO(tar_gz(["a.wav", "b.wav", "c.wav"])), 2)
    assert list(out) == ["a.wav", "b.wav"]
    assert out["a.wav"] == wav_bytes()


def test_load_sample_streams_once_then_reuses_the_kept_clips(tmp_path: Path) -> None:
    fs = FakeFs(["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"])
    sample, folder = fleurs.load_sample(spec(), "en", tmp_path, fs=fs, log=lambda _: None)
    assert fs.reads == [
        "datasets/google/fleurs@abc/data/en_us/test.tsv",
        "datasets/google/fleurs@abc/data/en_us/audio/test.tar.gz",
    ]
    assert len(sample.items) == 2
    # drawn from the first 4 members only: e.wav is never a candidate
    assert {c.file for c in sample.items} <= {"a.wav", "c.wav", "d.wav"}
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        [*(c.file for c in sample.items), "sample.json"]
    )

    again, _ = fleurs.load_sample(spec(), "en", tmp_path, fs=OfflineFs(), log=lambda _: None)
    assert again == sample

    with pytest.raises(AssertionError, match="network"):  # another seed is another sample
        fleurs.load_sample(spec(seed=2), "en", tmp_path, fs=OfflineFs(), log=lambda _: None)
    (folder / sample.items[0].file).unlink()  # a clip gone: fetched again
    with pytest.raises(AssertionError, match="network"):
        fleurs.load_sample(spec(), "en", tmp_path, fs=OfflineFs(), log=lambda _: None)


def test_corpus_error_weights_by_reference_length() -> None:
    refs, hyps = ["red green blue gray", "pink"], ["red green blue gray", "teal"]
    assert evals.asr_corpus_error(refs, hyps, "en") == ("wer", pytest.approx(1 / 5))
    assert evals.asr_corpus_error(["今日は"], ["今日わ"], "ja") == ("cer", pytest.approx(1 / 3))
    # English number words read as digits: FLEURS spells them, Transcribe writes digits
    assert evals.asr_error("One-hundred percent.", "100%", "en") == ("wer", 0.0)


def test_fleurs_languages_follow_the_item_filter() -> None:
    s = spec(languages={"en": "en_us", "ja": "ja_jp"})
    assert evals.fleurs_languages(s, []) == ["en", "ja"]
    assert evals.fleurs_languages(s, ["fleurs"]) == ["en", "ja"]
    assert evals.fleurs_languages(s, ["fleurs-ja", "clean-en"]) == ["ja"]
    assert evals.fleurs_languages(s, ["clean-en"]) == []
    assert evals.fleurs_languages(None, []) == []


class FakeAsr:
    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.calls: list[tuple[int, str]] = []

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        self.calls.append((len(audio), language))
        return self.answers[language]


class FakeModels:
    def __init__(self, asr_: FakeAsr):
        self.asr = asr_
        self.configs: list[Any] = []

    def set_config(self, config: Any) -> None:
        self.configs.append(config)

    @contextlib.contextmanager
    def use(self, name: str) -> Generator[Any]:
        assert name == "transcribe"
        yield self.asr


def test_run_fleurs_transcribes_each_clip_whole_in_its_language(tmp_path: Path) -> None:
    fake = FakeAsr({"en": " Hello there. "})
    models = FakeModels(fake)
    fs = FakeFs(["a.wav", "c.wav", "d.wav", "e.wav"])
    rows = evals.run_fleurs(
        spec(),
        ["en"],
        root=tmp_path,
        models=models,
        overrides={"models.transcribe.revision": "pinned"},
        fs=fs,
        log=lambda _: None,
    )
    [row] = rows
    assert fake.calls == [(asr.SR // 2, "en")] * 2  # the whole 0.5 s clip, no chunking
    assert models.configs[0].source_language == "en"
    assert models.configs[0].models["transcribe"].revision == "pinned"  # overrides apply
    assert (row.language, row.metric, row.clips, row.audio_s) == ("en", "wer", 2, 1.0)
    assert [r.hypothesis for r in row.results] == ["Hello there."] * 2
    perfect = [r.error == 0.0 for r in row.results]
    assert perfect == [r.reference == "Hello there." for r in row.results]


def test_read_clip_refuses_another_sample_rate(tmp_path: Path) -> None:
    path = tmp_path / "x.wav"
    path.write_bytes(wav_bytes(sr=22050))
    with pytest.raises(ValueError, match="22050 Hz"):
        evals.read_clip(path)


def test_run_set_with_only_fleurs_reports_per_language_rows(tmp_path: Path) -> None:
    (tmp_path / "evals" / "sets").mkdir(parents=True)
    (tmp_path / "evals" / "sets" / "tiny.yaml").write_text(
        "fleurs:\n  revision: abc\n  clips: 2\n  pool: 4\n  languages: {en: en_us, ja: ja_jp}\n",
        encoding="utf-8",
    )
    tiny = evals.load_set("tiny", tmp_path).fleurs
    assert tiny is not None
    for lang in ("en", "ja"):  # samples already kept: no network
        fs = FakeFs(["a.wav", "c.wav", "d.wav", "e.wav"])
        fleurs.load_sample(tiny, lang, tmp_path, fs=fs, log=lambda _: None)
    models = FakeModels(FakeAsr({"en": "Good night.", "ja": "x"}))
    report = evals.run_set(
        "tiny", root=tmp_path, models=models, judge_=False, ask_=False, log=lambda _: None
    )
    assert report.items == []
    assert [(r.language, r.metric) for r in report.fleurs] == [("en", "wer"), ("ja", "cer")]
    assert report.summary["fleurs_cer"] == report.fleurs[1].error
    assert report.summary["fleurs_wer"] == report.fleurs[0].error
    md = evals.render_markdown(report)
    assert "## FLEURS (Transcribe only, report-only)" in md
    assert "| ja | cer |" in md
    assert evals.compare(report, report) == []  # FLEURS keys are outside the gate

    only_ja = evals.run_set(
        "tiny",
        root=tmp_path,
        items=["fleurs-ja"],
        models=models,
        judge_=False,
        ask_=False,
        log=lambda _: None,
    )
    assert [r.language for r in only_ja.fleurs] == ["ja"]


def test_reports_without_fleurs_still_load() -> None:
    baseline = evals.load_report(ROOT / "evals" / "baseline.json")
    assert baseline.fleurs == []
    assert "fleurs_wer" not in baseline.summary
